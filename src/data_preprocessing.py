"""Data preprocessing utilities for customer behavior profiling.

This module provides a configurable data preprocessing pipeline that can:
- Remove duplicate records
- Address missing values via imputation or column dropping
- Detect and remove outliers using the IQR rule
- Scale numerical features
- Encode categorical features (one-hot or target encoding)
- Generate reproducible train/test splits
- Persist fitted preprocessing pipelines and processed datasets

The implementation is designed to work across heterogeneous behavioral datasets (transactional,
web analytics, demographics) while remaining extensible for downstream modeling tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # scikit-learn >= 1.3 provides a TargetEncoder
    from sklearn.preprocessing import TargetEncoder  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    TargetEncoder = None  # type: ignore

set_config(transform_output="pandas")

ScalerChoice = Union[str, None]
EncoderChoice = Union[str, None]


@dataclass
class TrainTestSplit:
    """Container for train/test split outputs."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: Optional[pd.Series]
    y_test: Optional[pd.Series]


class DataPreprocessor:
    """Configurable preprocessing pipeline for heterogeneous customer datasets."""

    def __init__(
        self,
        *,
        numerical_features: Optional[Sequence[str]] = None,
        categorical_features: Optional[Sequence[str]] = None,
        target_column: Optional[str] = None,
        numerical_imputation: str = "median",
        categorical_imputation: str = "most_frequent",
        scaler: ScalerChoice = "standard",
        encoder: EncoderChoice = "onehot",
        missing_threshold: Optional[float] = 0.95,
        drop_duplicates: bool = True,
        outlier_method: Optional[str] = "iqr",
        outlier_iqr_multiplier: float = 1.5,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.numerical_features = list(numerical_features) if numerical_features else None
        self.categorical_features = (
            list(categorical_features) if categorical_features else None
        )
        self.target_column = target_column
        self.numerical_imputation = numerical_imputation
        self.categorical_imputation = categorical_imputation
        self.scaler_choice = scaler
        self.encoder_choice = encoder
        self.missing_threshold = missing_threshold
        self.drop_duplicates = drop_duplicates
        self.outlier_method = outlier_method
        self.outlier_iqr_multiplier = outlier_iqr_multiplier
        self.test_size = test_size
        self.random_state = random_state

        self.pipeline_: Optional[Pipeline] = None
        self.columns_to_drop_: Sequence[str] = []
        self.numeric_features_: Sequence[str] = []
        self.categorical_features_: Sequence[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataPreprocessor":
        """Fit preprocessing pipeline on the provided feature frame."""
        X_clean, y_clean = self._clean_dataframe(X, y, training=True)
        self._fit_pipeline(X_clean, y_clean)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using the fitted pipeline."""

        if self.pipeline_ is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        X_clean, _ = self._clean_dataframe(X, training=False)
        transformed = self.pipeline_.transform(X_clean)
        return self._ensure_dataframe(transformed, X_clean)

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Convenience method to fit and transform data in one shot."""
        X_clean, y_clean = self._clean_dataframe(X, y, training=True)
        transformed = self._fit_pipeline(X_clean, y_clean, return_transformed=True)
        return transformed

    def split_fit_transform(
        self,
        data: pd.DataFrame,
        *,
        target_column: Optional[str] = None,
        stratify: bool = True,
    ) -> TrainTestSplit:
        """Split dataset, fit preprocessor on training data, and return transformed splits."""

        target_column = target_column or self.target_column
        if target_column is None:
            raise ValueError("target_column must be provided for supervised splits.")

        y = data[target_column]
        X = data.drop(columns=[target_column])

        X_clean, y_clean = self._clean_dataframe(X, y, training=True)
        stratify_y = y_clean if stratify else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean,
                y_clean,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify_y,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean,
                y_clean,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=None,
            )

        self.fit(X_train, y_train)
        X_train_t = self.transform(X_train)
        X_test_t = self.transform(X_test)

        return TrainTestSplit(X_train_t, X_test_t, y_train, y_test)

    def save_preprocessor(self, path: Union[str, Path]) -> Path:
        """Persist the fitted preprocessing pipeline to disk."""

        if self.pipeline_ is None:
            raise RuntimeError("Cannot save an unfitted preprocessor.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline_, path)
        return path

    def save_processed_data(
        self,
        data: Union[pd.DataFrame, TrainTestSplit],
        destination: Union[str, Path],
        *,
        include_target: bool = True,
    ) -> Path:
        """Save processed dataset or train/test split to a CSV file."""

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, TrainTestSplit):
            combined = self._combine_split(data, include_target=include_target)
            combined.to_csv(destination, index=False)
        else:
            data.to_csv(destination, index=False)

        return destination

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _fit_pipeline(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        *,
        return_transformed: bool = False,
    ) -> Optional[pd.DataFrame]:
        self.numeric_features_ = self._infer_numeric_features(X)
        self.categorical_features_ = self._infer_categorical_features(X)
        self.pipeline_ = self._build_pipeline()

        if return_transformed:
            transformed = self.pipeline_.fit_transform(X, y)
            return self._ensure_dataframe(transformed, X)

        self.pipeline_.fit(X, y)
        return None

    def _clean_dataframe(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        *,
        training: bool,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = X.copy()
        target = y.copy() if y is not None else None

        if training and self.missing_threshold is not None:
            missing_ratio = df.isna().mean()
            columns_to_drop = missing_ratio[missing_ratio >= self.missing_threshold].index
            df = df.drop(columns=columns_to_drop)
            self.columns_to_drop_ = list(columns_to_drop)
        elif not training and self.columns_to_drop_:
            df = df.drop(columns=[c for c in self.columns_to_drop_ if c in df.columns])

        if training and self.outlier_method == "iqr":
            df, target = self._remove_outliers_iqr(df, target)

        if self.drop_duplicates and training:
            if target is not None:
                target_name = target.name or "target"
                combined = df.copy()
                combined[target_name] = target.loc[df.index]
                combined = combined.drop_duplicates()
                target = combined[target_name]
                df = combined.drop(columns=[target_name])
            else:
                df = df.drop_duplicates()

        return df, target

    def _infer_numeric_features(self, df: pd.DataFrame) -> Sequence[str]:
        if self.numerical_features is not None:
            return [c for c in self.numerical_features if c in df.columns]
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def _infer_categorical_features(self, df: pd.DataFrame) -> Sequence[str]:
        if self.categorical_features is not None:
            return [c for c in self.categorical_features if c in df.columns]
        return df.select_dtypes(exclude=[np.number]).columns.tolist()

    def _build_pipeline(self) -> Pipeline:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy=self.numerical_imputation)),
        ]
        if self.scaler_choice == "standard":
            numeric_steps.append(("scaler", StandardScaler()))
        elif self.scaler_choice == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            numeric_steps.append(("scaler", MinMaxScaler()))

        categorical_steps = [
            ("imputer", SimpleImputer(strategy=self.categorical_imputation, fill_value="missing")),
        ]

        if self.encoder_choice == "onehot":
            categorical_steps.append(
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                )
            )
        elif self.encoder_choice == "target":
            if TargetEncoder is None:
                raise ImportError(
                    "TargetEncoder requires scikit-learn >= 1.3. "
                    "Install or upgrade scikit-learn to use target encoding."
                )
            categorical_steps.append(("encoder", TargetEncoder(handle_unknown="value")))
        elif self.encoder_choice is not None:
            raise ValueError(f"Unsupported encoder: {self.encoder_choice}")

        transformers = []
        if self.numeric_features_:
            transformers.append(("numeric", Pipeline(numeric_steps), self.numeric_features_))
        if self.categorical_features_:
            transformers.append(
                ("categorical", Pipeline(categorical_steps), self.categorical_features_)
            )

        return Pipeline(
            steps=[
                (
                    "features",
                    ColumnTransformer(transformers=transformers, remainder="drop"),
                )
            ]
        )

    def _remove_outliers_iqr(
        self, df: pd.DataFrame, target: Optional[pd.Series]
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        numeric_features = (
            self.numeric_features_ or self._infer_numeric_features(df)
        )

        if not numeric_features:
            return df, target

        numeric_df = df[numeric_features]
        q1 = numeric_df.quantile(0.25)
        q3 = numeric_df.quantile(0.75)
        iqr = q3 - q1
        threshold = self.outlier_iqr_multiplier
        mask = ~((numeric_df < (q1 - threshold * iqr)) | (numeric_df > (q3 + threshold * iqr))).any(axis=1)

        df_filtered = df.loc[mask]
        target_filtered = target.loc[mask] if target is not None else None
        return df_filtered, target_filtered

    def _combine_split(
        self, split: TrainTestSplit, *, include_target: bool
    ) -> pd.DataFrame:
        train = split.X_train.copy()
        test = split.X_test.copy()

        train["dataset"] = "train"
        test["dataset"] = "test"

        frames = [train, test]

        if include_target and split.y_train is not None and split.y_test is not None:
            train_target = split.y_train.copy()
            test_target = split.y_test.copy()
            train_target.name = train_target.name or "target"
            test_target.name = test_target.name or train_target.name
            train[train_target.name] = train_target
            test[test_target.name] = test_target

        return pd.concat(frames).reset_index(drop=True)

    def _ensure_dataframe(self, data, X_reference: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline_ is None:
            raise RuntimeError("Preprocessor pipeline is not fitted.")
        if isinstance(data, pd.DataFrame):
            return data
        feature_names = self.pipeline_.get_feature_names_out()
        return pd.DataFrame(data, columns=feature_names, index=X_reference.index)


def load_dataset(path: Union[str, Path], **read_kwargs: Dict) -> pd.DataFrame:
    """Load a dataset from CSV or JSON using pandas."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, **read_kwargs)
    if path.suffix.lower() == ".json":
        return pd.read_json(path, **read_kwargs)

    raise ValueError(f"Unsupported file format for {path}")


def save_dataframe(df: pd.DataFrame, destination: Union[str, Path]) -> Path:
    """Persist a DataFrame to CSV under the data/processed directory by default."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination_path, index=False)
    return destination_path


__all__ = ["DataPreprocessor", "TrainTestSplit", "load_dataset", "save_dataframe"]
