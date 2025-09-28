import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.evaluation import (
    EvaluationConfig,
    compute_metrics,
    cross_validate_model,
    evaluate_model,
    plot_confusion_matrix,
    plot_curves,
)


def _sample_classification_data(random_state: int = 123):
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        class_sep=1.2,
        weights=[0.85, 0.15],
        random_state=random_state,
    )
    columns = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), y


def test_compute_metrics_returns_expected_keys():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_scores = np.array([0.2, 0.8, 0.3, 0.4, 0.9])

    metrics = compute_metrics(y_true, y_pred, y_scores)
    assert {"precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy"} == set(metrics.keys())
    assert metrics["precision"] >= 0


def test_plot_helpers_generate_figures():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_scores = np.array([0.2, 0.9, 0.3, 0.4])

    fig_cm, _ = plot_confusion_matrix(y_true, y_pred)
    (roc_fig, _), (pr_fig, _) = plot_curves(y_true, y_scores)

    assert fig_cm is not None
    assert roc_fig is not None
    assert pr_fig is not None



def test_evaluate_model_creates_reports(tmp_path: Path):
    X, y = _sample_classification_data()
    model = LogisticRegression(max_iter=200, class_weight="balanced")
    model.fit(X, y)

    config = EvaluationConfig(report_dir=tmp_path)
    result = evaluate_model(model, X, y, config=config, model_name="lr")

    expected_files = {
        "confusion_matrix",
        "roc_curve",
        "pr_curve",
        "classification_report",
        "metrics",
    }
    assert expected_files.issubset(result.report_paths.keys())
    for path in result.report_paths.values():
        assert Path(path).exists()

    with open(result.report_paths["metrics"], "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
        assert "precision" in metrics


def test_evaluate_model_handles_external_validation(tmp_path: Path):
    X, y = _sample_classification_data(random_state=321)
    model = LogisticRegression(max_iter=200, class_weight="balanced")
    model.fit(X, y)

    X_ext, y_ext = _sample_classification_data(random_state=999)
    config = EvaluationConfig(report_dir=tmp_path)
    result = evaluate_model(
        model,
        X,
        y,
        config=config,
        X_external=X_ext,
        y_external=y_ext,
        model_name="external",
    )

    assert "external_metrics" in result.report_paths
    assert Path(result.report_paths["external_metrics"]).exists()


def test_cross_validate_model_returns_scores():
    X, y = _sample_classification_data()
    model = LogisticRegression(max_iter=200, class_weight="balanced")
    scores = cross_validate_model(model, X, y, config=EvaluationConfig(cv_folds=3))

    assert "accuracy" in scores
    assert "roc_auc" in scores
