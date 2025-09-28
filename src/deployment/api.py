"""FastAPI service exposing fraud risk scoring endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .model_service import FraudModelService, PredictionVault
from .security import EncryptionManager, JWTAuthManager, JWTSettings


class TokenRequest(BaseModel):
    api_key: str = Field(..., description="Pre-shared API key for service-to-service auth")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PredictRequest(BaseModel):
    transactions: List[Dict[str, Any]] = Field(..., description="List of transaction payloads")
    encrypt_results: bool = Field(False, description="Persist encrypted audit log entries")
    alert_threshold: float = Field(0.5, ge=0.0, le=1.0)


class PredictionPayload(BaseModel):
    fraud_risk: float
    anomaly_score: float
    explanations: List[str]
    transaction: Dict[str, Any]


class BatchSummary(BaseModel):
    total: int
    alerts: int
    alert_threshold: float


class PredictResponse(BaseModel):
    predictions: List[PredictionPayload]
    summary: BatchSummary


def build_app() -> FastAPI:
    jwt_secret = os.getenv("JWT_SECRET")
    if not jwt_secret:
        raise RuntimeError("JWT_SECRET must be configured before starting the API service.")

    jwt_settings = JWTSettings(secret_key=jwt_secret)
    jwt_manager = JWTAuthManager(jwt_settings)

    aes_manager = EncryptionManager()
    audit_path = Path(os.getenv("PREDICTION_LOG", "reports/predictions.enc"))

    batch_size_raw = os.getenv("BATCH_SIZE", "512")
    try:
        batch_size = int(batch_size_raw)
    except ValueError:
        batch_size = 512

    artifact_dir = Path(os.getenv("MODEL_ARTIFACT_DIR", "artifacts"))

    model_service = FraudModelService(
        training_data_path=Path(os.getenv("TRAINING_DATA", "data/processed/transactions_split.csv")),
        target_column=os.getenv("TARGET_COLUMN", "is_fraud"),
        batch_size=batch_size,
        model_path=artifact_dir / os.getenv("MODEL_FILENAME", "model.joblib"),
        preprocessor_path=artifact_dir / os.getenv("PREPROCESSOR_FILENAME", "preprocessor.joblib"),
        anomaly_path=artifact_dir / os.getenv("ANOMALY_FILENAME", "anomaly.joblib"),
    )
    vault = PredictionVault(audit_path, aes_manager)

    security = HTTPBearer(auto_error=False)
    api_key_secret = os.getenv("API_KEY_SECRET")

    app = FastAPI(title="Fraud Detection Service", version="1.0.0")
    app.add_middleware(HTTPSRedirectMiddleware)

    def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
        if credentials is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing credentials")
        token = credentials.credentials
        try:
            return jwt_manager.verify_token(token)
        except PermissionError as exc:  # pragma: no cover - fastapi will convert
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    @app.get("/health", tags=["system"])
    def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/token", response_model=TokenResponse, tags=["auth"])
    def issue_token(request: TokenRequest) -> TokenResponse:
        if not api_key_secret:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="API key not configured")
        if request.api_key != api_key_secret:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        token = jwt_manager.create_token(subject="service-client", scopes=["predict"])
        return TokenResponse(access_token=token)

    @app.post("/predict", response_model=PredictResponse, tags=["prediction"])
    def predict(
        request: PredictRequest,
        background_tasks: BackgroundTasks,
        _claims: Dict[str, Any] = Depends(authenticate),
    ) -> PredictResponse:
        if not request.transactions:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No transactions provided")
        try:
            df = model_service.predict_batch(request.transactions)
        except Exception as exc:  # pragma: no cover - bubbled as 400
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        predictions: List[PredictionPayload] = []
        alert_count = 0
        for record in df.to_dict(orient="records"):
            fraud_risk = float(record.pop("fraud_risk"))
            anomaly_score = float(record.pop("anomaly_score"))
            explanations = record.pop("explanations")
            transaction = {k: v for k, v in record.items() if k not in {"features", "metadata"}}
            predictions.append(
                PredictionPayload(
                    fraud_risk=fraud_risk,
                    anomaly_score=anomaly_score,
                    explanations=explanations,
                    transaction=transaction,
                )
            )
            if fraud_risk >= request.alert_threshold:
                alert_count += 1

        if request.encrypt_results:
            for prediction in predictions:
                background_tasks.add_task(
                    vault.append,
                    {
                        "fraud_risk": prediction.fraud_risk,
                        "anomaly_score": prediction.anomaly_score,
                        "explanations": prediction.explanations,
                        "transaction": prediction.transaction,
                    },
                )

        summary = BatchSummary(
            total=len(predictions),
            alerts=alert_count,
            alert_threshold=request.alert_threshold,
        )
        return PredictResponse(predictions=predictions, summary=summary)

    return app


try:  # pragma: no cover - executed during module import
    app = build_app()
except RuntimeError:
    # Allows importing the module for configuration inspection without all env vars.
    app = FastAPI(title="Fraud Detection Service", version="1.0.0")

__all__ = ["app", "build_app", "PredictRequest", "PredictResponse"]
