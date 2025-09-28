"""Feature engineering utilities for customer behavior profiling.

This module enriches transactional datasets with behavioral features suited for
fraud detection workflows. It provides:

- A configurable :class:`FeatureEngineer` class for transaction-level feature generation.
- Aggregation helpers to build customer-level profiles summarising historical behaviour.

The engineered features cover:

* Temporal characteristics (hour-of-day, weekday/weekend flags, recency gaps).
* Frequency and velocity metrics across multiple rolling windows.
* Monetary statistics including rolling sums/means and log-scaled values.
* Device/browser fingerprints and change indicators.
* Geo-location consistency flags between habitual and observed locations.

All operations are implemented with pandas to preserve compatibility with the
preprocessing pipeline and downstream scikit-learn models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Configuration describing column names and rolling windows.

    Parameters
    ----------
    customer_id_col:
        Column identifying the customer/account entity.
    timestamp_col:
        Column containing event timestamps; coerced to ``datetime64[ns]``.
    amount_col:
        Transaction amount column (numeric). Optional if monetary features are not required.
    device_col / browser_col / channel_col:
        Columns representing device/browser identifiers for fingerprinting. Missing columns are ignored.
    location_col / home_location_col:
        Columns used to evaluate geo-consistency.
    velocity_windows:
        Time-window strings understood by pandas (e.g. ``"1H"``, ``"1D"``) used for rolling
        transaction counts.
    amount_windows:
        Windows for rolling monetary statistics.
    """

    customer_id_col: str = "customer_id"
    timestamp_col: str = "timestamp"
    amount_col: Optional[str] = "amount"
    device_col: Optional[str] = "device"
    browser_col: Optional[str] = "browser"
    channel_col: Optional[str] = "channel"
    location_col: Optional[str] = "location"
    home_location_col: Optional[str] = "home_location"
    velocity_windows: Sequence[str] = ("1h", "1d", "7d")
    amount_windows: Sequence[str] = ("1d", "7d", "30d")


class FeatureEngineer:
    """Generate transaction-level and aggregated behavioural features."""

    def __init__(self, config: FeatureEngineeringConfig | None = None) -> None:
        self.config = config or FeatureEngineeringConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a feature-enriched copy of ``data``.

        Parameters
        ----------
        data:
            Transaction-level DataFrame containing at least the configured
            ``customer_id_col`` and ``timestamp_col``.
        """

        cfg = self.config
        required = {cfg.customer_id_col, cfg.timestamp_col}
        missing = required - set(data.columns)
        if missing:
            raise KeyError(f"Input data missing required columns: {', '.join(sorted(missing))}")

        df = data.copy()
        df["__orig_index"] = np.arange(len(df))
        df = df.sort_values([cfg.customer_id_col, cfg.timestamp_col]).reset_index(drop=True)

        df[cfg.timestamp_col] = self._ensure_datetime(df[cfg.timestamp_col])

        self._add_time_features(df)
        self._add_velocity_features(df)
        if cfg.amount_col and cfg.amount_col in df.columns:
            self._add_amount_features(df)
        self._add_device_fingerprint_features(df)
        self._add_geo_features(df)

        df = df.sort_values("__orig_index").drop(columns=["__orig_index"])
        df.reset_index(drop=True, inplace=True)
        return df

    def aggregate_customer_profiles(self, features: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transaction features into customer-level profiles."""

        cfg = self.config
        if cfg.customer_id_col not in features.columns:
            raise KeyError(
                f"Features DataFrame must contain '{cfg.customer_id_col}' for aggregation."
            )

        df = features.copy()
        df[cfg.timestamp_col] = self._ensure_datetime(df[cfg.timestamp_col])

        aggregates: list[pd.DataFrame] = []

        # Average & max transaction amount per customer
        if cfg.amount_col and cfg.amount_col in df.columns:
            amount_stats = (
                df.groupby(cfg.customer_id_col)[cfg.amount_col]
                .agg(amount_avg="mean", amount_max="max", amount_std="std", amount_sum="sum")
                .fillna({"amount_std": 0.0})
            )
            aggregates.append(amount_stats)

        # Frequency statistics (daily/weekly/monthly counts)
        resample_specs = {
            "daily": "D",
            "weekly": "W",
            "monthly": "ME",
        }
        for label, rule in resample_specs.items():
            period_counts = (
                df.set_index(cfg.timestamp_col)
                .groupby(cfg.customer_id_col)
                .resample(rule)
                .size()
                .rename("count")
                .reset_index(level=0)
            )
            stats = (
                period_counts.groupby(cfg.customer_id_col)["count"]
                .agg(
                    **{
                        f"txn_{label}_mean": "mean",
                        f"txn_{label}_max": "max",
                        f"txn_{label}_std": "std",
                    }
                )
                .fillna({f"txn_{label}_std": 0.0})
            )
            aggregates.append(stats)

        # Velocity statistics from rolling counts if present
        velocity_cols = [c for c in df.columns if c.startswith("txn_count_")]
        if velocity_cols:
            agg_map = {col: ["mean", "max"] for col in velocity_cols}
            velocity_stats = df.groupby(cfg.customer_id_col).agg(agg_map)
            # Flatten MultiIndex columns
            velocity_stats.columns = ["_".join(col).strip("_") for col in velocity_stats.columns]
            aggregates.append(velocity_stats)

        # Geo consistency rates
        if "location_matches_home" in df.columns:
            geo_stats = (
                df.groupby(cfg.customer_id_col)["location_matches_home"]
                .agg(geo_match_rate="mean")
            )
            aggregates.append(geo_stats)

        # Device/browser diversity
        diversity_cols: Dict[str, str] = {}
        if cfg.device_col and cfg.device_col in df.columns:
            diversity_cols[cfg.device_col] = "unique_device_count"
        if cfg.browser_col and cfg.browser_col in df.columns:
            diversity_cols[cfg.browser_col] = "unique_browser_count"
        if diversity_cols:
            diversity = (
                df.groupby(cfg.customer_id_col)
                .agg({col: "nunique" for col in diversity_cols})
                .rename(columns=diversity_cols)
            )
            aggregates.append(diversity)

        if not aggregates:
            return df[[cfg.customer_id_col]].drop_duplicates().reset_index(drop=True)

        profile = aggregates[0]
        for stats in aggregates[1:]:
            profile = profile.join(stats, how="outer")

        return profile.reset_index()

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------
    def _ensure_datetime(self, series: pd.Series) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        if pd.api.types.is_numeric_dtype(series):
            # Assume seconds resolution if numeric values provided
            return pd.to_datetime(series, unit="s", errors="coerce")
        return pd.to_datetime(series, errors="coerce")

    def _add_time_features(self, df: pd.DataFrame) -> None:
        cfg = self.config
        ts = df[cfg.timestamp_col]
        df["txn_hour"] = ts.dt.hour
        df["txn_weekday"] = ts.dt.weekday
        df["txn_is_weekend"] = ts.dt.weekday.isin([5, 6]).astype(int)
        df["txn_month"] = ts.dt.month
        df["txn_day_of_month"] = ts.dt.day
        df["txn_is_night"] = ts.dt.hour.isin(list(range(0, 6))).astype(int)

        # Recency features
        df["time_since_last_txn_sec"] = (
            df.groupby(self.config.customer_id_col)[cfg.timestamp_col]
            .diff()
            .dt.total_seconds()
            .fillna(-1)
        )

    def _add_velocity_features(self, df: pd.DataFrame) -> None:
        cfg = self.config
        customer_col = cfg.customer_id_col
        ts_col = cfg.timestamp_col

        df["__ones"] = 1.0

        for window in cfg.velocity_windows:
            normalized_window = window.lower()
            col_name = f"txn_count_{normalized_window}"
            df[col_name] = (
                df.groupby(customer_col, group_keys=False)
                .apply(
                    lambda g: self._rolling_window_sum(
                        g, value_col="__ones", window=normalized_window
                    )
                )
            )

        df.drop(columns=["__ones"], inplace=True)

    def _add_amount_features(self, df: pd.DataFrame) -> None:
        cfg = self.config
        amount_col = cfg.amount_col
        assert amount_col is not None

        df[f"{amount_col}_log1p"] = np.log1p(df[amount_col].clip(lower=0))

        for window in cfg.amount_windows:
            normalized_window = window.lower()
            base = f"amount_{normalized_window}"
            df[f"{base}_sum"] = (
                df.groupby(cfg.customer_id_col, group_keys=False)
                .apply(
                    lambda g: self._rolling_window_sum(
                        g, value_col=amount_col, window=normalized_window
                    )
                )
            )
            df[f"{base}_mean"] = (
                df.groupby(cfg.customer_id_col, group_keys=False)
                .apply(
                    lambda g: self._rolling_window_mean(
                        g, value_col=amount_col, window=normalized_window
                    )
                )
            )
            df[f"{base}_max"] = (
                df.groupby(cfg.customer_id_col, group_keys=False)
                .apply(
                    lambda g: self._rolling_window_max(
                        g, value_col=amount_col, window=normalized_window
                    )
                )
            )

    def _add_device_fingerprint_features(self, df: pd.DataFrame) -> None:
        cfg = self.config
        components = [
            col
            for col in [cfg.device_col, cfg.browser_col, cfg.channel_col]
            if col and col in df.columns
        ]
        if components:
            df["device_fingerprint"] = df[components].astype(str).agg("|".join, axis=1)

        # Change flags relative to previous transaction per customer
        for col in [cfg.device_col, cfg.browser_col, cfg.channel_col]:
            if col and col in df.columns:
                df[f"{col}_changed"] = (
                    df.groupby(cfg.customer_id_col)[col]
                    .apply(self._flag_changes)
                    .reset_index(level=0, drop=True)
                )

    def _add_geo_features(self, df: pd.DataFrame) -> None:
        cfg = self.config
        if cfg.location_col and cfg.location_col in df.columns:
            df[cfg.location_col] = df[cfg.location_col].astype(str)
        if cfg.home_location_col and cfg.home_location_col in df.columns:
            df[cfg.home_location_col] = df[cfg.home_location_col].astype(str)

        if (
            cfg.location_col
            and cfg.home_location_col
            and cfg.location_col in df.columns
            and cfg.home_location_col in df.columns
        ):
            df["location_matches_home"] = (
                df[cfg.location_col].str.lower() == df[cfg.home_location_col].str.lower()
            ).astype(int)
        elif cfg.location_col and cfg.location_col in df.columns:
            df["location_matches_home"] = np.nan

    # ------------------------------------------------------------------
    # Rolling helpers
    # ------------------------------------------------------------------
    def _rolling_window_sum(
        self, group: pd.DataFrame, *, value_col: str, window: str
    ) -> pd.Series:
        series = (
            group.set_index(self.config.timestamp_col)[value_col]
            .rolling(window=window, min_periods=1)
            .sum()
        )
        series = series.reset_index(drop=True)
        series.index = group.index
        return series

    def _rolling_window_mean(
        self, group: pd.DataFrame, *, value_col: str, window: str
    ) -> pd.Series:
        series = (
            group.set_index(self.config.timestamp_col)[value_col]
            .rolling(window=window, min_periods=1)
            .mean()
        )
        series = series.reset_index(drop=True)
        series.index = group.index
        return series

    def _rolling_window_max(
        self, group: pd.DataFrame, *, value_col: str, window: str
    ) -> pd.Series:
        series = (
            group.set_index(self.config.timestamp_col)[value_col]
            .rolling(window=window, min_periods=1)
            .max()
        )
        series = series.reset_index(drop=True)
        series.index = group.index
        return series

    @staticmethod
    def _flag_changes(series: pd.Series) -> pd.Series:
        changed = series.ne(series.shift(1))
        if not changed.empty:
            changed.iloc[0] = False
        return changed.astype(int)


__all__ = ["FeatureEngineeringConfig", "FeatureEngineer"]
