import base64
import os
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.deployment.api import build_app
from src.deployment.security import EncryptionManager, JWTAuthManager, JWTSettings


def test_encryption_roundtrip():
    key = EncryptionManager.generate_key()
    manager = EncryptionManager(key=base64.urlsafe_b64decode(key))
    payload = {"fraud_risk": 0.82}
    token = manager.encrypt_json(payload)
    restored = manager.decrypt_json(token)
    assert restored == payload


def test_jwt_manager_issue_and_verify():
    settings = JWTSettings(secret_key="secret", issuer="test-service")
    manager = JWTAuthManager(settings)
    token = manager.create_token("user-123", scopes=["predict"])
    claims = manager.verify_token(token)
    assert claims["sub"] == "user-123"
    assert "predict" in claims.get("scopes", [])


def test_predict_endpoint_with_auth(tmp_path, monkeypatch):
    # Create a lightweight training dataset
    df = pd.DataFrame(
        {
            "amount": [50, 75, 500, 1200, 45, 900],
            "channel": ["web", "mobile", "web", "web", "mobile", "mobile"],
            "device": ["android", "ios", "android", "android", "ios", "ios"],
            "location": ["US", "US", "CA", "US", "US", "MX"],
            "home_location": ["US", "US", "US", "US", "US", "US"],
            "is_fraud": [0, 0, 1, 1, 0, 1],
            "dataset": ["train", "train", "train", "train", "test", "test"],
        }
    )
    train_path = tmp_path / "train.csv"
    df.to_csv(train_path, index=False)

    monkeypatch.setenv("TRAINING_DATA", str(train_path))
    monkeypatch.setenv("TARGET_COLUMN", "is_fraud")
    monkeypatch.setenv("JWT_SECRET", "super-secret")
    monkeypatch.setenv("API_KEY_SECRET", "service-key")
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "predictions.enc"))
    monkeypatch.setenv("MODEL_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    app = build_app()
    client = TestClient(app)

    token_resp = client.post("/token", json={"api_key": "service-key"})
    assert token_resp.status_code == 200
    token = token_resp.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "transactions": [
            {
                "amount": 250,
                "channel": "web",
                "device": "android",
                "location": "US",
                "home_location": "US",
            }
        ],
        "encrypt_results": True,
        "alert_threshold": 0.4,
    }
    response = client.post("/predict", json=payload, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total"] == 1
    assert len(body["predictions"]) == 1
    assert "explanations" in body["predictions"][0]

    # Ensure encrypted audit log was written
    log_path = Path(os.getenv("PREDICTION_LOG"))
    assert log_path.exists()
    encrypted_records = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert encrypted_records

    # Missing token should be rejected
    invalid = client.post("/predict", json=payload)
    assert invalid.status_code == 401
