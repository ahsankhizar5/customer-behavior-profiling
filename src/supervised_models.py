"""Supervised learning and hybrid fraud models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neural_network import MLPClassifier

from .anomaly_detection import AnomalyDetectionResult, score_anomaly_model

try:  # Optional dependency
	import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
	xgb = None  # type: ignore

try:  # Optional dependency
	import lightgbm as lgb
except ImportError:  # pragma: no cover - optional dependency
	lgb = None  # type: ignore


def _ensure_dataframe(data: pd.DataFrame | np.ndarray | Iterable[Iterable[float]]) -> pd.DataFrame:
	if isinstance(data, pd.DataFrame):
		return data.copy()
	array = np.asarray(list(data)) if not isinstance(data, np.ndarray) else data
	columns = [f"feature_{i}" for i in range(array.shape[1])]
	return pd.DataFrame(array, columns=columns)


def _predict_proba_safe(model: Any, X: pd.DataFrame) -> np.ndarray:
	if hasattr(model, "predict_proba"):
		return model.predict_proba(X)[:, 1]
	if hasattr(model, "decision_function"):
		decision = model.decision_function(X)
		return 1.0 / (1.0 + np.exp(-decision))
	preds = model.predict(X)
	return preds.astype(float)


def _evaluate_model(model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
	proba = _predict_proba_safe(model, X_test)
	preds = (proba >= 0.5).astype(int)
	accuracy = accuracy_score(y_test, preds)
	precision, recall, f1, _ = precision_recall_fscore_support(
		y_test, preds, average="binary", zero_division=0
	)
	roc_auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")

	return {
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"roc_auc": roc_auc,
	}


@dataclass
class SupervisedModelResult:
	model: Any
	metrics: Dict[str, float]
	predictions: np.ndarray
	probabilities: np.ndarray


def train_logistic_regression(
	X_train: pd.DataFrame | np.ndarray,
	y_train: np.ndarray,
	*,
	max_iter: int = 1000,
) -> LogisticRegression:
	model = LogisticRegression(max_iter=max_iter, class_weight="balanced")
	model.fit(X_train, y_train)
	return model


def train_random_forest(
	X_train: pd.DataFrame | np.ndarray,
	y_train: np.ndarray,
	*,
	n_estimators: int = 200,
	max_depth: Optional[int] = None,
	random_state: int = 42,
) -> RandomForestClassifier:
	model = RandomForestClassifier(
		n_estimators=n_estimators,
		max_depth=max_depth,
		random_state=random_state,
		class_weight="balanced_subsample",
	)
	model.fit(X_train, y_train)
	return model


def train_xgboost(
	X_train: pd.DataFrame | np.ndarray,
	y_train: np.ndarray,
	*,
	random_state: int = 42,
	n_estimators: int = 200,
) -> Any:
	if xgb is None:  # pragma: no cover - optional dependency
		raise ImportError("xgboost is not installed; install it to use the XGBoost classifier")

	model = xgb.XGBClassifier(
		n_estimators=n_estimators,
		learning_rate=0.1,
		max_depth=4,
		subsample=0.8,
		colsample_bytree=0.8,
		random_state=random_state,
		eval_metric="logloss",
		use_label_encoder=False,
	)
	model.fit(X_train, y_train)
	return model


def train_lightgbm(
	X_train: pd.DataFrame | np.ndarray,
	y_train: np.ndarray,
	*,
	random_state: int = 42,
	n_estimators: int = 200,
) -> Any:
	if lgb is None:  # pragma: no cover - optional dependency
		raise ImportError("lightgbm is not installed; install it to use the LightGBM classifier")

	model = lgb.LGBMClassifier(
		n_estimators=n_estimators,
		learning_rate=0.05,
		num_leaves=31,
		subsample=0.8,
		colsample_bytree=0.8,
		random_state=random_state,
	)
	model.fit(X_train, y_train)
	return model


def train_mlp(
	X_train: pd.DataFrame | np.ndarray,
	y_train: np.ndarray,
	*,
	hidden_layer_sizes: tuple[int, ...] = (64, 32),
	random_state: int = 42,
) -> MLPClassifier:
	model = MLPClassifier(
		hidden_layer_sizes=hidden_layer_sizes,
		activation="relu",
		solver="adam",
		max_iter=300,
		random_state=random_state,
	)
	model.fit(X_train, y_train)
	return model


def run_supervised_suite(
	X_train: pd.DataFrame | np.ndarray,
	y_train: np.ndarray,
	X_test: pd.DataFrame | np.ndarray,
	y_test: np.ndarray,
	*,
	include_xgboost: bool = True,
	include_lightgbm: bool = True,
) -> Dict[str, SupervisedModelResult]:
	"""Train a suite of supervised models and report metrics."""

	X_train_df = _ensure_dataframe(X_train)
	X_test_df = _ensure_dataframe(X_test)

	results: Dict[str, SupervisedModelResult] = {}

	models: Dict[str, Any] = {
		"logistic_regression": train_logistic_regression(X_train_df, y_train),
		"random_forest": train_random_forest(X_train_df, y_train),
		"mlp_classifier": train_mlp(X_train_df, y_train),
	}

	if include_xgboost and xgb is not None:
		models["xgboost"] = train_xgboost(X_train_df, y_train)
	if include_lightgbm and lgb is not None:
		models["lightgbm"] = train_lightgbm(X_train_df, y_train)

	for name, model in models.items():
		proba = _predict_proba_safe(model, X_test_df)
		metrics = _evaluate_model(model, X_test_df, y_test)
		results[name] = SupervisedModelResult(
			model=model,
			metrics=metrics,
			predictions=(proba >= 0.5).astype(int),
			probabilities=proba,
		)

	return results


def augment_with_anomaly_scores(
	X: pd.DataFrame | np.ndarray,
	scores: np.ndarray,
	column_name: str = "anomaly_score",
) -> pd.DataFrame:
	df = _ensure_dataframe(X)
	df[column_name] = scores
	return df


class HybridFraudPipeline:
	"""Pipeline that fuses unsupervised anomaly scores with supervised learning."""

	def __init__(
		self,
		anomaly_detector: Callable[[pd.DataFrame | np.ndarray], AnomalyDetectionResult],
		supervised_factory: Callable[[], Any] | None = None,
		*,
		anomaly_column: str = "anomaly_score",
	) -> None:
		self.anomaly_detector = anomaly_detector
		self.supervised_factory = supervised_factory or (
			lambda: LogisticRegression(max_iter=1000, class_weight="balanced")
		)
		self.anomaly_column = anomaly_column

		self.anomaly_result: Optional[AnomalyDetectionResult] = None
		self.supervised_model: Optional[Any] = None

	def fit(self, X_train: pd.DataFrame | np.ndarray, y_train: np.ndarray) -> "HybridFraudPipeline":
		X_train_df = _ensure_dataframe(X_train)
		self.anomaly_result = self.anomaly_detector(X_train_df)

		scores = self.anomaly_result.scores
		augmented = augment_with_anomaly_scores(X_train_df, scores, column_name=self.anomaly_column)

		factory = self.supervised_factory
		model = factory()
		model.fit(augmented, y_train)

		self.supervised_model = model
		return self

	def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
		if self.anomaly_result is None or self.supervised_model is None:
			raise RuntimeError("HybridFraudPipeline must be fitted before inference.")

		scores = score_anomaly_model(self.anomaly_result, X)
		augmented = augment_with_anomaly_scores(X, scores, column_name=self.anomaly_column)

		if hasattr(self.supervised_model, "predict_proba"):
			return self.supervised_model.predict_proba(augmented)
		if hasattr(self.supervised_model, "decision_function"):
			decision = self.supervised_model.decision_function(augmented)
			return np.column_stack((1 - decision, decision))
		preds = self.supervised_model.predict(augmented)
		return np.column_stack((1 - preds, preds))

	def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
		proba = self.predict_proba(X)
		return (proba[:, 1] >= 0.5).astype(int)


__all__ = [
	"SupervisedModelResult",
	"train_logistic_regression",
	"train_random_forest",
	"train_xgboost",
	"train_lightgbm",
	"train_mlp",
	"run_supervised_suite",
	"augment_with_anomaly_scores",
	"HybridFraudPipeline",
]
