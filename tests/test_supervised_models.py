import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.anomaly_detection import train_isolation_forest
from src.supervised_models import (
    HybridFraudPipeline,
    augment_with_anomaly_scores,
    run_supervised_suite,
)


def _classification_dataset(random_state: int = 42):
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_clusters_per_class=2,
        class_sep=1.5,
        weights=[0.85, 0.15],
        random_state=random_state,
    )
    columns = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), y


def test_supervised_suite_trains_models():
    X, y = _classification_dataset()
    X_train, X_test = X.iloc[:200], X.iloc[200:]
    y_train, y_test = y[:200], y[200:]

    results = run_supervised_suite(
        X_train,
        y_train,
        X_test,
        y_test,
        include_xgboost=False,
        include_lightgbm=False,
    )

    assert set(results.keys()) == {"logistic_regression", "random_forest", "mlp_classifier"}
    for result in results.values():
        assert result.probabilities.shape[0] == len(X_test)
        assert 0.0 <= result.metrics["accuracy"] <= 1.0


def test_hybrid_pipeline_augments_scores():
    X, y = _classification_dataset(random_state=21)
    X_train, X_test = X.iloc[:180], X.iloc[180:]
    y_train, y_test = y[:180], y[180:]

    def detector(data):
        return train_isolation_forest(data, contamination=0.1, random_state=21)

    pipeline = HybridFraudPipeline(anomaly_detector=detector)
    pipeline.fit(X_train, y_train)

    augmented = augment_with_anomaly_scores(X_test, np.ones(len(X_test)))
    assert "anomaly_score" in augmented.columns

    proba = pipeline.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    preds = pipeline.predict(X_test)
    assert preds.shape == (len(X_test),)