"""Evaluation utilities for fraud detection models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	accuracy_score,
	classification_report,
	confusion_matrix,
	precision_recall_curve,
	precision_recall_fscore_support,
	recall_score,
	roc_auc_score,
	roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class EvaluationConfig:
	report_dir: Path = Path("reports")
	confusion_matrix_filename: str = "confusion_matrix.png"
	roc_curve_filename: str = "roc_curve.png"
	pr_curve_filename: str = "pr_curve.png"
	classification_report_filename: str = "classification_report.csv"
	metrics_filename: str = "metrics_summary.json"
	cv_folds: int = 5
	metrics: Tuple[str, ...] = ("precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy")


@dataclass
class EvaluationResult:
	metrics: Dict[str, float]
	classification_report: pd.DataFrame
	roc_curve: Optional[Dict[str, np.ndarray]] = None
	pr_curve: Optional[Dict[str, np.ndarray]] = None
	confusion_matrix: Optional[np.ndarray] = None
	cross_validation_scores: Optional[Dict[str, float]] = None
	report_paths: Dict[str, Path] = field(default_factory=dict)


def _ensure_dir(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int], y_scores: Iterable[float]) -> Dict[str, float]:
	y_true_arr = np.asarray(y_true)
	y_pred_arr = np.asarray(y_pred)
	y_scores_arr = np.asarray(y_scores)

	precision, recall, f1, _ = precision_recall_fscore_support(
		y_true_arr, y_pred_arr, average="binary", zero_division=0
	)
	roc_auc = roc_auc_score(y_true_arr, y_scores_arr) if len(np.unique(y_true_arr)) > 1 else float("nan")
	precision_curve, recall_curve, _ = precision_recall_curve(y_true_arr, y_scores_arr)
	pr_auc = np.trapezoid(precision_curve, recall_curve)
	accuracy = accuracy_score(y_true_arr, y_pred_arr)

	return {
		"precision": float(precision),
		"recall": float(recall),
		"f1": float(f1),
		"roc_auc": float(roc_auc),
		"pr_auc": float(pr_auc),
		"accuracy": float(accuracy),
	}


def plot_confusion_matrix(
	y_true: Iterable[int],
	y_pred: Iterable[int],
	labels: Optional[List[str]] = None,
	*,
	ax: Optional[plt.Axes] = None,
	show: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
	cm = confusion_matrix(y_true, y_pred)
	display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
	fig, ax = plt.subplots(figsize=(6, 5)) if ax is None else (ax.figure, ax)
	display.plot(ax=ax, cmap="Blues", colorbar=False)
	ax.set_title("Confusion Matrix")
	if show:
		plt.show()
	return fig, ax


def plot_curves(
	y_true: Iterable[int], y_scores: Iterable[float], *, show: bool = False
) -> Tuple[Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, plt.Axes]]:
	y_true_arr = np.asarray(y_true)
	y_scores_arr = np.asarray(y_scores)
	fpr, tpr, _ = roc_curve(y_true_arr, y_scores_arr)
	precision, recall, _ = precision_recall_curve(y_true_arr, y_scores_arr)

	roc_fig, roc_ax = plt.subplots(figsize=(6, 5))
	roc_ax.plot(fpr, tpr, label="ROC Curve")
	roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
	roc_ax.set_xlabel("False Positive Rate")
	roc_ax.set_ylabel("True Positive Rate")
	roc_ax.set_title("ROC Curve")
	roc_ax.legend()

	pr_fig, pr_ax = plt.subplots(figsize=(6, 5))
	pr_ax.plot(recall, precision, label="PR Curve")
	pr_ax.set_xlabel("Recall")
	pr_ax.set_ylabel("Precision")
	pr_ax.set_title("Precision-Recall Curve")
	pr_ax.legend()

	if show:
		plt.show()

	return (roc_fig, roc_ax), (pr_fig, pr_ax)


def _classification_report_df(y_true: Iterable[int], y_pred: Iterable[int]) -> pd.DataFrame:
	report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
	return pd.DataFrame(report_dict).transpose()


def cross_validate_model(
	model: ClassifierMixin,
	X: pd.DataFrame | np.ndarray,
	y: Iterable[int],
	*,
	config: EvaluationConfig,
) -> Dict[str, float]:
	skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=42)
	metrics = {
		"accuracy": "accuracy",
		"precision": "precision",
		"recall": "recall",
		"f1": "f1",
	}

	scores = {}
	for metric_name, scoring in metrics.items():
		score = cross_val_score(model, X, y, cv=skf, scoring=scoring)
		scores[metric_name] = float(np.mean(score))

	roc_scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
	scores["roc_auc"] = float(np.mean(roc_scores))

	return scores


def evaluate_model(
	model: ClassifierMixin,
	X: pd.DataFrame | np.ndarray,
	y: Iterable[int],
	*,
	config: Optional[EvaluationConfig] = None,
	labels: Optional[List[str]] = None,
	X_external: Optional[pd.DataFrame | np.ndarray] = None,
	y_external: Optional[Iterable[int]] = None,
	model_name: str = "model",
) -> EvaluationResult:
	config = config or EvaluationConfig()
	report_dir = _ensure_dir(Path(config.report_dir) / model_name)

	probabilities = (
		model.predict_proba(X)[:, 1]
		if hasattr(model, "predict_proba")
		else model.decision_function(X)
	)
	predictions = (probabilities >= 0.5).astype(int)

	metrics = compute_metrics(y, predictions, probabilities)
	report_df = _classification_report_df(y, predictions)
	fpr, tpr, _ = roc_curve(y, probabilities)
	precision, recall, _ = precision_recall_curve(y, probabilities)
	cm = confusion_matrix(y, predictions)

	roc_auc = metrics.get("roc_auc")
	pr_auc = metrics.get("pr_auc")

	(_roc_fig, _), (_pr_fig, _) = plot_curves(y, probabilities)
	roc_path = report_dir / config.roc_curve_filename
	pr_path = report_dir / config.pr_curve_filename
	_conf_fig, _ = plot_confusion_matrix(y, predictions, labels=labels)
	conf_path = report_dir / config.confusion_matrix_filename

	_conf_fig.savefig(conf_path, bbox_inches="tight")
	_roc_fig.savefig(roc_path, bbox_inches="tight")
	_pr_fig.savefig(pr_path, bbox_inches="tight")
	plt.close(_conf_fig)
	plt.close(_roc_fig)
	plt.close(_pr_fig)

	report_path = report_dir / config.classification_report_filename
	report_df.to_csv(report_path)

	metrics_path = report_dir / config.metrics_filename
	pd.Series(metrics).to_json(metrics_path, indent=2)

	cv_scores = cross_validate_model(model, X, y, config=config)

	report_paths = {
		"confusion_matrix": conf_path,
		"roc_curve": roc_path,
		"pr_curve": pr_path,
		"classification_report": report_path,
		"metrics": metrics_path,
	}

	external_result: Optional[Dict[str, float]] = None
	if X_external is not None and y_external is not None:
		prob_ext = (
			model.predict_proba(X_external)[:, 1]
			if hasattr(model, "predict_proba")
			else model.decision_function(X_external)
		)
		pred_ext = (prob_ext >= 0.5).astype(int)
		external_result = compute_metrics(y_external, pred_ext, prob_ext)
		external_path = report_dir / "external_metrics.json"
		pd.Series(external_result).to_json(external_path, indent=2)
		report_paths["external_metrics"] = external_path

	return EvaluationResult(
		metrics=metrics,
		classification_report=report_df,
		roc_curve={"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc},
		pr_curve={"precision": precision, "recall": recall, "pr_auc": pr_auc},
		confusion_matrix=cm,
		cross_validation_scores=cv_scores,
		report_paths=report_paths,
	)


__all__ = [
	"EvaluationConfig",
	"EvaluationResult",
	"compute_metrics",
	"plot_confusion_matrix",
	"plot_curves",
	"cross_validate_model",
	"evaluate_model",
]
