import numpy as np
import pandas as pd

from src.anomaly_detection import (
    AnomalyDetectionResult,
    AutoencoderConfig,
    AutoencoderResult,
    score_anomaly_model,
    train_autoencoder,
    train_isolation_forest,
    train_one_class_svm,
)


def _synthetic_data(seed: int = 123, anomalies: int = 5):
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, size=(120, 6))
    outliers = rng.normal(6, 1, size=(anomalies, 6))
    data = np.vstack([normal, outliers])
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(data.shape[1])])


def test_isolation_forest_detects_outliers():
    df = _synthetic_data()
    result = train_isolation_forest(df, contamination=0.05, random_state=7)
    assert isinstance(result, AnomalyDetectionResult)
    # Ensure at least some anomalies are detected
    assert result.labels.sum() >= 3
    new_scores = score_anomaly_model(result, df.iloc[:10])
    assert new_scores.shape == (10,)


def test_one_class_svm_runs():
    df = _synthetic_data()
    result = train_one_class_svm(df, nu=0.1)
    assert isinstance(result, AnomalyDetectionResult)
    assert result.labels.shape[0] == len(df)
    # Re-scoring should work with the stored scaler
    new_scores = score_anomaly_model(result, df.tail(8))
    assert new_scores.shape == (8,)


def test_autoencoder_training_and_scoring():
    df = _synthetic_data(anomalies=8)
    config = AutoencoderConfig(hidden_units=(16,), latent_dim=8, epochs=5, batch_size=16, contamination=0.1)
    result = train_autoencoder(df, config=config)
    assert isinstance(result, AutoencoderResult)
    assert result.threshold > 0
    assert result.labels.sum() >= 5
    new_scores = score_anomaly_model(result, df.head(12))
    assert new_scores.shape == (12,)