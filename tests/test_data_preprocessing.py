import pandas as pd
import numpy as np
from pathlib import Path

from src.data_preprocessing import DataPreprocessor


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 2, 3, 4],
            "amount": [100.0, 250.0, 250.0, 5000.0, np.nan],
            "device": ["mobile", "desktop", "desktop", "mobile", "tablet"],
            "country": ["US", "US", "US", "FR", "FR"],
            "is_fraud": [0, 0, 0, 1, 1],
        }
    )


def test_fit_transform_removes_missing_duplicates_and_outliers():
    df = _sample_dataframe()
    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud"])

    preprocessor = DataPreprocessor(
        target_column="is_fraud",
        scaler="minmax",
        encoder="onehot",
        numerical_imputation="median",
        categorical_imputation="most_frequent",
    )

    transformed = preprocessor.fit_transform(X, y)

    # Ensure duplicates removed (only unique rows should remain after cleaning)
    assert len(transformed) < len(df), "Duplicate rows should be dropped during preprocessing"

    # All missing values should be imputed
    assert transformed.isna().sum().sum() == 0

    # Outlier row (amount=5000) should be filtered out by the IQR rule
    assert transformed.shape[0] == 3

    # One-hot encoding should create indicator columns for categorical variables
    encoded_columns = [col for col in transformed.columns if "device" in col or "country" in col]
    assert encoded_columns, "Categorical one-hot encoded columns are missing"


def test_split_and_save_processed_data(tmp_path: Path):
    df = _sample_dataframe()

    preprocessor = DataPreprocessor(target_column="is_fraud", scaler="standard")
    split = preprocessor.split_fit_transform(df, target_column="is_fraud")

    # Verify splits
    assert not split.X_train.empty
    assert not split.X_test.empty
    assert split.y_train is not None and split.y_test is not None

    # Ensure transformed features have no missing values
    assert split.X_train.isna().sum().sum() == 0
    assert split.X_test.isna().sum().sum() == 0

    # Persist processed data and confirm structure
    destination = tmp_path / "processed.csv"
    preprocessor.save_processed_data(split, destination)
    saved = pd.read_csv(destination)

    assert set(saved["dataset"].unique()) == {"train", "test"}
    assert "dataset" in saved.columns
    assert "is_fraud" in saved.columns
