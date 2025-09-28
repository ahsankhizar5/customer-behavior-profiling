"""Data pipeline utilities orchestrating collection, preprocessing, and feature engineering."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from .data_preprocessing import (
    DataPreprocessor,
    TrainTestSplit,
    load_dataset,
)
from .feature_engineering import FeatureEngineer, FeatureEngineeringConfig

PathLike = Union[str, Path]


@dataclass
class DataPipelineConfig:
    """Configuration describing raw data sources and preprocessing behaviour."""

    raw_data_path: PathLike
    processed_dir: PathLike = Path("data/processed")
    target_column: Optional[str] = "is_fraud"
    stratify: bool = True
    transactions_output: str = "transactions_processed.csv"
    customer_profiles_output: str = "customer_profiles.csv"
    combined_split_output: str = "transactions_split.csv"
    save_transactions: bool = True
    save_customer_profiles: bool = True
    save_split: bool = True
    preprocessor_kwargs: Dict[str, Any] = field(default_factory=dict)
    feature_engineering_config: FeatureEngineeringConfig = field(
        default_factory=FeatureEngineeringConfig
    )

    def __post_init__(self) -> None:
        self.raw_data_path = Path(self.raw_data_path)
        self.processed_dir = Path(self.processed_dir)


@dataclass
class DataPipelineResult:
    """Container for artifacts produced by the data pipeline."""

    raw_data: pd.DataFrame
    cleaned_data: pd.DataFrame
    processed_transactions: pd.DataFrame
    customer_profiles: Optional[pd.DataFrame]
    train_test_split: Optional[TrainTestSplit]
    processed_transactions_path: Optional[Path]
    customer_profiles_path: Optional[Path]
    combined_split_path: Optional[Path]


class DataPipeline:
    """Execute the end-to-end data preparation workflow."""

    def __init__(
        self,
        config: DataPipelineConfig,
        *,
        preprocessor: Optional[DataPreprocessor] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
    ) -> None:
        self.config = config
        self.preprocessor = preprocessor or DataPreprocessor(
            target_column=config.target_column, **config.preprocessor_kwargs
        )
        self.feature_engineer = feature_engineer or FeatureEngineer(
            config=config.feature_engineering_config
        )

    def run(self) -> DataPipelineResult:
        cfg = self.config
        raw_df = load_dataset(cfg.raw_data_path)
        processed_dir = cfg.processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)

        target_col = cfg.target_column
        if target_col and target_col in raw_df.columns:
            y = raw_df[target_col]
            X = raw_df.drop(columns=[target_col])
        else:
            y = None
            X = raw_df

        cleaned_X, cleaned_y = self.preprocessor.clean_dataframe(X, y, training=True)
        cleaned_df = cleaned_X.copy()
        if cleaned_y is not None and target_col is not None:
            cleaned_df[target_col] = cleaned_y

        # Fit the preprocessor on the cleaned data and transform it.
        self.preprocessor.fit(cleaned_X, cleaned_y)
        processed_features = self.preprocessor.transform(cleaned_X)
        if cleaned_y is not None and target_col is not None:
            processed_features[target_col] = cleaned_y.loc[processed_features.index]

        # Feature engineering using the cleaned dataset (pre-encoding).
        try:
            enriched_transactions = self.feature_engineer.engineer_features(cleaned_df)
            customer_profiles = self.feature_engineer.aggregate_customer_profiles(
                enriched_transactions
            )
        except KeyError:
            # If required columns are missing we skip feature engineering but still
            # output the processed features for modelling.
            enriched_transactions = cleaned_df
            customer_profiles = None

        # Generate and persist train/test split separately using a fresh preprocessor
        split: Optional[TrainTestSplit] = None
        combined_split_path: Optional[Path] = None
        if target_col and target_col in raw_df.columns and cfg.save_split:
            split_preprocessor = DataPreprocessor(
                target_column=target_col, **cfg.preprocessor_kwargs
            )
            split = split_preprocessor.split_fit_transform(
                raw_df, target_column=target_col, stratify=cfg.stratify
            )
            combined_split_path = processed_dir / cfg.combined_split_output
            split_preprocessor.save_processed_data(
                split, combined_split_path, include_target=True
            )

        processed_transactions_path: Optional[Path] = None
        if cfg.save_transactions:
            processed_transactions_path = processed_dir / cfg.transactions_output
            processed_features.to_csv(processed_transactions_path, index=False)

        customer_profiles_path: Optional[Path] = None
        if cfg.save_customer_profiles and customer_profiles is not None:
            customer_profiles_path = processed_dir / cfg.customer_profiles_output
            customer_profiles.to_csv(customer_profiles_path, index=False)

        return DataPipelineResult(
            raw_data=raw_df,
            cleaned_data=enriched_transactions,
            processed_transactions=processed_features,
            customer_profiles=customer_profiles,
            train_test_split=split,
            processed_transactions_path=processed_transactions_path,
            customer_profiles_path=customer_profiles_path,
            combined_split_path=combined_split_path,
        )


__all__ = ["DataPipeline", "DataPipelineConfig", "DataPipelineResult"]


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entrypoint for running the data pipeline on a raw dataset."""

    parser = argparse.ArgumentParser(description="Run the customer behaviour data pipeline.")
    parser.add_argument("--raw", required=True, help="Path to the raw CSV/JSON dataset.")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory where processed datasets will be written.",
    )
    parser.add_argument(
        "--target-column",
        default="is_fraud",
        help="Target column name. Use an empty string to skip supervised split.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Disable generation of the train/test split artifact.",
    )

    args = parser.parse_args(argv)

    target_column = args.target_column or None
    config = DataPipelineConfig(
        raw_data_path=args.raw,
        processed_dir=args.processed_dir,
        target_column=target_column,
        save_split=not args.no_split,
    )

    pipeline = DataPipeline(config)
    result = pipeline.run()

    print("Processed transactions:", result.processed_transactions_path)
    if result.customer_profiles_path:
        print("Customer profiles:", result.customer_profiles_path)
    if result.combined_split_path:
        print("Train/Test split:", result.combined_split_path)


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    main()
