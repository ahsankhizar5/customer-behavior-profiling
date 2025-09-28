import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from src.data_pipeline import DataPipeline, DataPipelineConfig
from src.feature_engineering import FeatureEngineeringConfig


def _build_sample_dataset() -> pd.DataFrame:
    base_time = datetime(2024, 1, 1, 8, 30)
    rows = []
    for i in range(6):
        rows.append(
            {
                "transaction_id": i + 1,
                "customer_id": "cust-001" if i < 4 else "cust-002",
                "timestamp": base_time + timedelta(hours=i),
                "amount": [42.5, 120.0, 4200.0, 55.0, 75.0, 5.0][i],
                "device": ["mobile", "mobile", "mobile", "desktop", "mobile", "mobile"][i],
                "browser": ["chrome", "chrome", "chrome", "firefox", "chrome", "chrome"][i],
                "channel": ["app", "app", "app", "web", "app", "app"][i],
                "location": ["US", "US", "US", "US", "CA", "CA"][i],
                "home_location": ["US", "US", "US", "US", "US", "US"][i],
                "is_fraud": [0, 0, 1, 0, 0, 0][i],
            }
        )
    # Introduce a duplicated row and a missing value row
    rows.append(rows[1].copy())
    rows[-1]["transaction_id"] = 99
    rows.append({**rows[0], "transaction_id": 100, "amount": None})
    return pd.DataFrame(rows)


def test_data_pipeline_end_to_end(tmp_path):
    raw_df = _build_sample_dataset()
    raw_path = tmp_path / "raw.csv"
    raw_df.to_csv(raw_path, index=False)

    processed_dir = tmp_path / "processed"
    config = DataPipelineConfig(
        raw_data_path=raw_path,
        processed_dir=processed_dir,
        target_column="is_fraud",
        stratify=False,
        feature_engineering_config=FeatureEngineeringConfig(
            customer_id_col="customer_id",
            timestamp_col="timestamp",
            amount_col="amount",
            device_col="device",
            browser_col="browser",
            channel_col="channel",
            location_col="location",
            home_location_col="home_location",
        ),
    )

    pipeline = DataPipeline(config)
    result = pipeline.run()

    assert result.processed_transactions_path is not None
    assert result.processed_transactions_path.exists()
    assert result.customer_profiles_path is not None
    assert result.customer_profiles_path.exists()
    assert result.combined_split_path is not None
    assert result.combined_split_path.exists()

    processed = result.processed_transactions
    # Duplicate rows and outliers should be filtered reducing count
    assert len(processed) < len(raw_df)
    # Encoded categorical features should be present
    categorical_cols = [c for c in processed.columns if "categorical__" in c]
    assert categorical_cols, "Expected one-hot encoded categorical columns"
    assert processed.isna().sum().sum() == 0

    customer_profiles = result.customer_profiles
    assert customer_profiles is not None
    assert "customer_id" in customer_profiles.columns
    assert not customer_profiles.empty

    split = result.train_test_split
    assert split is not None
    assert not split.X_train.empty
    assert not split.X_test.empty

    # Combined split CSV includes dataset column distinguishing train/test
    split_df = pd.read_csv(result.combined_split_path)
    assert set(split_df["dataset"].unique()) == {"train", "test"}