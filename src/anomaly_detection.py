"""Anomaly detection strategies for fraud risk scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def _to_array(data: pd.DataFrame | np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
	if isinstance(data, pd.DataFrame):
		return data.values
	return np.asarray(data, dtype=np.float32)


def _standardise(data: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
	scaler = StandardScaler()
	scaled = scaler.fit_transform(data)
	return scaled, scaler


@dataclass
class AnomalyDetectionResult:
	scores: np.ndarray
	labels: np.ndarray
	model: Any


@dataclass
class AutoencoderConfig:
	hidden_units: Tuple[int, ...] = (32, 16)
	latent_dim: int = 8
	dropout: float = 0.0
	epochs: int = 20
	batch_size: int = 32
	learning_rate: float = 1e-3
	contamination: float = 0.05


@dataclass
class AutoencoderResult(AnomalyDetectionResult):
	threshold: float
	scaler: StandardScaler
	reconstruction_errors: np.ndarray
	autoencoder: Model


def train_isolation_forest(
	data: pd.DataFrame | np.ndarray,
	*,
	contamination: float = 0.05,
	random_state: int = 42,
) -> AnomalyDetectionResult:
	"""Fit an Isolation Forest and return anomaly scores + labels."""

	array = _to_array(data)
	model = IsolationForest(contamination=contamination, random_state=random_state)
	model.fit(array)

	decision_scores = model.decision_function(array)
	# Higher decision_function => more normal. Convert to anomaly scores where higher = more anomalous.
	anomaly_scores = -decision_scores
	labels = (model.predict(array) == -1).astype(int)
	return AnomalyDetectionResult(scores=anomaly_scores, labels=labels, model=model)


def train_one_class_svm(
	data: pd.DataFrame | np.ndarray,
	*,
	kernel: str = "rbf",
	nu: float = 0.05,
	gamma: str | float = "scale",
) -> AnomalyDetectionResult:
	"""Fit a One-Class SVM for outlier detection."""

	array = _to_array(data)
	array, scaler = _standardise(array)
	model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
	model.fit(array)

	decision_scores = model.decision_function(array)
	anomaly_scores = -decision_scores
	labels = (model.predict(array) == -1).astype(int)

	# Store scaler with the model for downstream use.
	model.scaler_ = scaler  # type: ignore[attr-defined]
	return AnomalyDetectionResult(scores=anomaly_scores, labels=labels, model=model)


def _build_autoencoder(input_dim: int, config: AutoencoderConfig) -> Model:
	model = Sequential(name="fraud_autoencoder")
	model.add(Input(shape=(input_dim,)))
	for units in config.hidden_units:
		model.add(Dense(units, activation="relu"))
		if config.dropout > 0:
			model.add(Dropout(config.dropout))
	model.add(Dense(config.latent_dim, activation="relu", name="latent"))
	for units in reversed(config.hidden_units):
		model.add(Dense(units, activation="relu"))
		if config.dropout > 0:
			model.add(Dropout(config.dropout))
	model.add(Dense(input_dim, activation="linear"))
	return model


def train_autoencoder(
	data: pd.DataFrame | np.ndarray,
	*,
	config: AutoencoderConfig | None = None,
) -> AutoencoderResult:
	"""Train a simple dense autoencoder for anomaly detection."""

	cfg = config or AutoencoderConfig()
	array = _to_array(data)
	array, scaler = _standardise(array)
	input_dim = array.shape[1]

	autoencoder = _build_autoencoder(input_dim, cfg)
	autoencoder.compile(optimizer=Adam(cfg.learning_rate), loss="mse")
	autoencoder.fit(
		array,
		array,
		epochs=cfg.epochs,
		batch_size=min(cfg.batch_size, array.shape[0]),
		shuffle=True,
		verbose=0,
	)

	reconstructed = autoencoder.predict(array, verbose=0)
	reconstruction_errors = np.mean(np.square(array - reconstructed), axis=1)
	threshold = np.percentile(reconstruction_errors, 100 * (1 - cfg.contamination))
	labels = (reconstruction_errors > threshold).astype(int)

	return AutoencoderResult(
		scores=reconstruction_errors,
		labels=labels,
		model=autoencoder,
		threshold=threshold,
		scaler=scaler,
		reconstruction_errors=reconstruction_errors,
		autoencoder=autoencoder,
	)


def score_autoencoder(
	result: AutoencoderResult,
	new_data: pd.DataFrame | np.ndarray,
) -> np.ndarray:
	"""Compute reconstruction errors for new samples using a fitted autoencoder."""

	array = _to_array(new_data)
	scaled = result.scaler.transform(array)
	reconstructed = result.autoencoder.predict(scaled, verbose=0)
	errors = np.mean(np.square(scaled - reconstructed), axis=1)
	return errors


def score_anomaly_model(
	result: AnomalyDetectionResult,
	new_data: pd.DataFrame | np.ndarray,
) -> np.ndarray:
	"""Generic scorer for any anomaly detector result."""

	if isinstance(result, AutoencoderResult):
		return score_autoencoder(result, new_data)

	model = result.model
	array = _to_array(new_data)
	if hasattr(model, "scaler_"):
		array = model.scaler_.transform(array)  # type: ignore[attr-defined]

	if hasattr(model, "decision_function"):
		return -model.decision_function(array)
	if hasattr(model, "score_samples"):
		return -model.score_samples(array)

	raise NotImplementedError(
		f"Unable to compute anomaly scores for model of type {type(model).__name__}"
	)


__all__ = [
	"AnomalyDetectionResult",
	"AutoencoderConfig",
	"AutoencoderResult",
	"train_isolation_forest",
	"train_one_class_svm",
	"train_autoencoder",
	"score_autoencoder",
	"score_anomaly_model",
]
