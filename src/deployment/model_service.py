"""Model serving utilities for fraud detection deployment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

from ..anomaly_detection import AnomalyDetectionResult, score_anomaly_model, train_isolation_forest
from ..data_preprocessing import DataPreprocessor, load_dataset
from ..supervised_models import train_logistic_regression
from .security import EncryptionManager


@dataclass
class FraudPrediction:
    fraud_risk: float
    anomaly_score: float
    explanations: List[str]
    metadata: Dict[str, Any]


class PredictionVault:
    """Encrypted persistence layer for prediction audit logs."""

    def __init__(self, destination: Path, encryption: EncryptionManager) -> None:
        self.destination = destination
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.encryption = encryption

    def append(self, record: Dict[str, Any]) -> None:
        token = self.encryption.encrypt(json.dumps(record).encode("utf-8"))
        with self.destination.open("a", encoding="utf-8") as handle:
            handle.write(token + "\n")


class FraudModelService:
    """End-to-end inference service wrapping preprocessing, modelling, and anomaly scoring."""

    def __init__(
        self,
        *,
        training_data_path: Path | str = Path("data/processed/transactions_split.csv"),
        target_column: str = "is_fraud",
        model_path: Path | str = Path("artifacts/model.joblib"),
        preprocessor_path: Path | str = Path("artifacts/preprocessor.joblib"),
        anomaly_path: Path | str = Path("artifacts/anomaly.joblib"),
        batch_size: int = 512,
        preprocessor: Optional[DataPreprocessor] = None,
        classifier: Optional[Any] = None,
        anomaly_result: Optional[AnomalyDetectionResult] = None,
    ) -> None:
        self.training_data_path = Path(training_data_path)
        self.target_column = target_column
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.anomaly_path = Path(anomaly_path)
        self.batch_size = batch_size

        self.preprocessor = preprocessor
        self.classifier = classifier
        self.anomaly_result = anomaly_result

        self._feature_means: Optional[pd.Series] = None
        self._feature_names: Optional[List[str]] = None

        if self.preprocessor is None or self.classifier is None or self.anomaly_result is None:
            self._bootstrap()
        else:
            self._feature_names = list(self.classifier.feature_names_in_)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_transaction(self, payload: Dict[str, Any]) -> FraudPrediction:
        df = pd.DataFrame([payload])
        features = self._transform(df)
        fraud_risk = float(self.classifier.predict_proba(features)[0, 1])
        anomaly_score = float(score_anomaly_model(self.anomaly_result, features)[0])
        explanations = self._feature_contributions(features.iloc[0])
        return FraudPrediction(
            fraud_risk=fraud_risk,
            anomaly_score=anomaly_score,
            explanations=explanations,
            metadata={"feature_vector": features.iloc[0].to_dict()},
        )

    def predict_batch(self, records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        batch: List[Dict[str, Any]] = []
        for payload in records:
            batch.append(payload)
            if len(batch) >= self.batch_size:
                frames.append(self._predict_chunk(batch))
                batch = []
        if batch:
            frames.append(self._predict_chunk(batch))
        if not frames:
            return pd.DataFrame(columns=["fraud_risk", "anomaly_score", "explanations"])
        return pd.concat(frames, ignore_index=True)

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Public helper for dashboards requiring direct access to model features."""

        return self._transform(df)

    def anomaly_scores(self, features: pd.DataFrame) -> np.ndarray:
        """Compute anomaly scores for already transformed feature frames."""

        return score_anomaly_model(self.anomaly_result, features)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _bootstrap(self) -> None:
        if self._load_artifacts():
            return

        data = load_dataset(self.training_data_path)
        if self.target_column not in data.columns:
            raise KeyError(
                f"Training dataset must include target column '{self.target_column}'."
            )

        train_df = data.copy()
        if "dataset" in train_df.columns:
            train_df = train_df[train_df["dataset"].str.lower() == "train"].drop(columns=["dataset"])
            if train_df.empty:
                train_df = data.copy().drop(columns=[c for c in ["dataset"] if c in data.columns])

        y = train_df[self.target_column]
        X = train_df.drop(columns=[self.target_column])

        self.preprocessor = DataPreprocessor(target_column=self.target_column)
        self.preprocessor.fit(X, y)
        features = self.preprocessor.transform(X)

        self.classifier = train_logistic_regression(features, y.to_numpy())
        self.anomaly_result = train_isolation_forest(features)

        self._feature_names = list(features.columns)
        self._feature_means = features.mean()

        self._persist_artifacts(features)

    def _persist_artifacts(self, features: pd.DataFrame) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, self.model_path)
        joblib.dump(self.preprocessor, self.preprocessor_path)
        joblib.dump(self.anomaly_result, self.anomaly_path)
        summary_path = self.model_path.with_suffix(".summary.json")
        summary = {
            "feature_means": features.mean().to_dict(),
            "feature_names": list(features.columns),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _load_artifacts(self) -> bool:
        if not (self.model_path.exists() and self.preprocessor_path.exists() and self.anomaly_path.exists()):
            return False
        self.classifier = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.anomaly_result = joblib.load(self.anomaly_path)

        summary_path = self.model_path.with_suffix(".summary.json")
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self._feature_means = pd.Series(summary.get("feature_means", {}))
            self._feature_names = summary.get("feature_names")
        elif hasattr(self.classifier, "feature_names_in_"):
            self._feature_names = list(self.classifier.feature_names_in_)
        return True

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise RuntimeError("Model service not initialised correctly: missing preprocessor.")
        features = self.preprocessor.transform(df)
        if self._feature_names is not None:
            missing = set(self._feature_names) - set(features.columns)
            for column in missing:
                features[column] = 0.0
            features = features[self._feature_names]
        return features

    def _predict_chunk(self, batch: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(batch)
        features = self._transform(df)
        fraud_risk = self.classifier.predict_proba(features)[:, 1]
        anomaly_scores = score_anomaly_model(self.anomaly_result, features)
        explanations = [self._feature_contributions(features.iloc[i]) for i in range(len(features))]
        result = df.copy()
        result["fraud_risk"] = fraud_risk
        result["anomaly_score"] = anomaly_scores
        result["explanations"] = explanations
        return result

    def _feature_contributions(self, row: pd.Series, top_k: int = 3) -> List[str]:
        if not hasattr(self.classifier, "coef_"):
            return ["Explanation unavailable for this classifier type."]

        coef = np.asarray(self.classifier.coef_)[0]
        values = row.to_numpy(dtype=float)
        contributions = values * coef
        top_indices = np.argsort(np.abs(contributions))[::-1][:top_k]
        explanations = []
        for idx in top_indices:
            feature_name = row.index[idx]
            delta = contributions[idx]
            explanations.append(f"Feature '{feature_name}' contributed {delta:.3f} to the logit.")
        if self._feature_means is not None:
            anomaly_offset = row - self._feature_means.reindex(row.index).fillna(0)
            max_feature = anomaly_offset.abs().idxmax()
            explanations.append(
                f"Largest deviation from baseline observed on '{max_feature}' (Δ={anomaly_offset[max_feature]:.3f})."
            )
        return explanations


__all__ = ["FraudModelService", "FraudPrediction", "PredictionVault"]
