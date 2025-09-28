import pandas as pd
import numpy as np

from src.feature_engineering import FeatureEngineer, FeatureEngineeringConfig


def _sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [
                "c1",
                "c1",
                "c1",
                "c2",
                "c2",
            ],
            "timestamp": [
                "2023-01-01 10:00:00",
                "2023-01-01 10:30:00",
                "2023-01-02 11:00:00",
                "2023-01-01 09:00:00",
                "2023-01-08 09:30:00",
            ],
            "amount": [50.0, 75.0, 200.0, 40.0, 500.0],
            "device": ["mobile", "mobile", "desktop", "desktop", "desktop"],
            "browser": ["chrome", "chrome", "firefox", "firefox", "edge"],
            "channel": ["web", "web", "web", "mobile", "mobile"],
            "location": ["US", "US", "FR", "US", "US"],
            "home_location": ["US", "US", "US", "US", "US"],
        }
    )


def test_engineer_features_generates_temporal_and_velocity_fields():
    df = _sample_transactions()
    engineer = FeatureEngineer(FeatureEngineeringConfig())

    features = engineer.engineer_features(df)

    expected_columns = {
        "txn_hour",
        "txn_weekday",
        "txn_is_weekend",
        "txn_count_1h",
        "txn_count_1d",
        "amount_1d_sum",
        "amount_7d_mean",
        "device_fingerprint",
        "device_changed",
        "browser_changed",
        "location_matches_home",
    }

    assert expected_columns.issubset(features.columns)

    first_row = features.iloc[0]
    assert first_row["txn_hour"] == 10
    assert first_row["txn_is_weekend"] == 1
    assert first_row["txn_count_1h"] == 1
    assert first_row["time_since_last_txn_sec"] == -1
    assert np.isclose(first_row["amount_1d_sum"], 50.0)
    assert first_row["device_changed"] == 0

    second_row = features.iloc[1]
    assert second_row["txn_count_1h"] == 2  # two txns within 1 hour window
    assert np.isclose(second_row["amount_1d_sum"], 125.0)
    assert second_row["device_changed"] == 0

    third_row = features.iloc[2]
    assert third_row["device_changed"] == 1  # switched device
    assert third_row["browser_changed"] == 1
    assert third_row["location_matches_home"] == 0


def test_aggregate_customer_profiles_computes_behavioral_summaries():
    df = _sample_transactions()
    engineer = FeatureEngineer()
    features = engineer.engineer_features(df)

    profiles = engineer.aggregate_customer_profiles(features)
    profile_c1 = profiles.loc[profiles["customer_id"] == "c1"].iloc[0]

    assert np.isclose(profile_c1["amount_avg"], (50.0 + 75.0 + 200.0) / 3)
    assert profile_c1["amount_max"] == 200.0
    assert profile_c1["txn_daily_max"] == 2  # two transactions on 2023-01-01
    assert profile_c1["unique_device_count"] == 2
    assert np.isclose(profile_c1["geo_match_rate"], 2 / 3, atol=1e-6)
