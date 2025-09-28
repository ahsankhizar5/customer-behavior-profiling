# customer-behavior-profiling
AI-powered Customer Behavior Profiling system for fraud detection. Implements data collection, preprocessing, behavior profiling with clustering, and anomaly detection using supervised and unsupervised ML models.

## Environment setup

You’ll find a project-local virtual environment at `.venv/`. To activate it in PowerShell:

```powershell
cd "c:\Users\ahsan\Downloads\Fraud Detection System\customer-behavior-profiling"
.\.venv\Scripts\Activate
```

Key commands once the environment is active:

- Install/refresh dependencies (already installed):

	```powershell
	python -m pip install -r requirements.txt
	```

- Run the automated test suite:

	```powershell
	python -m pytest
	```

Deactivate the environment anytime with `deactivate`.

## Data pipeline

Step two of the project is implemented through `src/data_pipeline.py`. The module wires
data collection (CSV/JSON loaders), preprocessing, feature engineering, and artifact
persistence. Run it from the command line after activating the virtual environment:

```powershell
python -m src.data_pipeline --raw data/creditcard.csv --processed-dir data/processed
```

Key outputs under `data/processed/`:

- `transactions_processed.csv` – model-ready features with scaling/encoding applied.
- `customer_profiles.csv` – aggregated customer fingerprints derived from engineered features.
- `transactions_split.csv` – combined train/test split annotated with a `dataset` column.

Adjust the `--raw` argument to point at your dataset (CSV or JSON, e.g. files under `data/`). Use `--no-split` to
skip generating the train/test artifact or `--target-column ""` for unsupervised workflows.

## Machine learning models

Step three introduces clustering, anomaly detection, supervised learning, and a hybrid fraud-risk pipeline.
Key modules:

- `src/clustering.py` – PCA/t-SNE/UMAP projection helpers plus K-Means, DBSCAN, and hierarchical clustering
	(with linkage matrices for dendrograms).
- `src/anomaly_detection.py` – Isolation Forest, One-Class SVM, and dense autoencoder trainers with unified scoring
	utilities.
- `src/supervised_models.py` – Logistic Regression, Random Forest, MLP, optional XGBoost/LightGBM boosters, and a
	`HybridFraudPipeline` class that appends anomaly scores to supervised features.

Example: train clustering + hybrid classifier on an engineered dataset (after running the data pipeline):

```python
import pandas as pd
from src.anomaly_detection import train_isolation_forest
from src.supervised_models import HybridFraudPipeline

features = pd.read_csv("data/processed/transactions_processed.csv")
labels = features.pop("is_fraud")

def detector(data):
		return train_isolation_forest(data, contamination=0.05)

pipeline = HybridFraudPipeline(anomaly_detector=detector)
pipeline.fit(features, labels)

fraud_risk = pipeline.predict_proba(features)[:, 1]
```

Automated checks cover the new algorithms via `pytest` (see `tests/test_clustering.py`,
`tests/test_anomaly_detection.py`, and `tests/test_supervised_models.py`).

## Model validation and reporting

Step four adds a dedicated evaluation toolkit in `src/evaluation.py` for validating any binary fraud model. Calling
`evaluate_model` will:

- compute precision, recall, F1, ROC-AUC, PR-AUC, and accuracy
- persist a CSV classification report plus JSON metric summary
- generate confusion matrix, ROC, and Precision–Recall plots
- run stratified cross-validation (default 5 folds) and expose the averaged scores
- optionally score an external holdout dataset and log the metrics separately

Artifacts land under `reports/<model_name>/` with predictable filenames. Example usage after training a model:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation import evaluate_model

train = pd.read_csv("data/processed/transactions_split.csv")
train = train[train["dataset"] == "train"]

X_train = train.drop(columns=["is_fraud", "dataset"])
y_train = train["is_fraud"]

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

result = evaluate_model(model, X_train, y_train, model_name="logreg_baseline")
print(result.metrics)
print(result.cross_validation_scores)
print(result.report_paths)
```

To benchmark on an untouched dataset, pass `X_external`/`y_external` into `evaluate_model`; the function will save
`external_metrics.json` alongside the standard reports. Visualization helpers (`plot_confusion_matrix` and `plot_curves`)
are available for custom notebooks as well.

Automated tests cover the evaluation workflow too—see `tests/test_evaluation.py`.

## Deployment & Integration

Step five introduces production-grade delivery via a FastAPI service and a Streamlit dashboard, wrapped in a security
layer that enforces JWT authentication and AES-256 encryption.

### FastAPI fraud scoring service

- Entrypoint: `src/deployment/api.py`
- Endpoint: `POST /predict` (protected by JWT)
	- Accepts a list of raw transaction JSON payloads (`transactions`), computes fraud risk scores, anomaly scores, and
		provides the top contributing features as human-readable explanations.
	- Supports batch processing via an internal streaming pipeline (`FraudModelService.predict_batch`) so large datasets
		are processed chunk-by-chunk.<br>
	- Optional `encrypt_results=true` flag persists an encrypted audit log (`reports/predictions.enc`) using AES-256-GCM.
- Authentication: `POST /token` exchanges a pre-shared API key for a JWT access token. All prediction requests must
	include `Authorization: Bearer <token>` headers.
- HTTPS enforcement: the service ships with Starlette's `HTTPSRedirectMiddleware`; run behind a TLS-terminating proxy
	or launch uvicorn with certificates to ensure in-transit encryption.

Run locally (after setting environment variables below):

```powershell
cd "c:\Users\ahsan\Downloads\Fraud Detection System\customer-behavior-profiling"
$env:TRAINING_DATA="data/processed/transactions_split.csv"
$env:TARGET_COLUMN="is_fraud"
$env:JWT_SECRET="<64-character-secret>"
$env:API_KEY_SECRET="<api-key>"
$env:AES_KEY="$(python -c "from src.deployment.security import EncryptionManager; print(EncryptionManager.generate_key())")"
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000 --ssl-keyfile path\to\dev.key --ssl-certfile path\to\dev.crt
```

Recommended TLS configuration: delegate certificates to a reverse proxy (NGINX/Traefik) or provide the file paths above
when starting uvicorn. The redirect middleware guarantees clients upgrade to HTTPS.

### Streamlit fraud analyst dashboard

- Entrypoint: `streamlit_app.py`
- Capabilities:
	- Upload CSV datasets and obtain batch predictions in seconds.
	- Visualize cluster embeddings (PCA) coloured by fraud probability and scaled by anomaly scores.
	- Highlight fraud alerts with textual rationales drawn from the logistic regression coefficients and deviation metrics.
- Uses the same `FraudModelService` under the hood, ensuring consistency between the API and dashboard outputs.

Launch the dashboard:

```powershell
cd "c:\Users\ahsan\Downloads\Fraud Detection System\customer-behavior-profiling"
streamlit run streamlit_app.py
```

Upload recent transactions (CSV) to see live predictions, anomaly distributions, and alert summaries.

### Security & data protection

- **JWT authentication** – managed by `JWTAuthManager`. Secrets configured via `JWT_SECRET` and optionally
	`API_KEY_SECRET` for issuing access tokens.
- **AES-256 encryption at rest** – `EncryptionManager` wraps audit logging (and any other payloads) in AES-GCM. Generate
	keys via `EncryptionManager.generate_key()` and set the value as `AES_KEY` (URL-safe base64 string).
- **HTTPS in transit** – enforced through middleware; run behind TLS in production to protect the REST interface.
- **Batch scalability** – `FraudModelService` processes payloads in configurable batches (default 512). Tweak via
	`BATCH_SIZE` environment variable when constructing the service.

Additional configuration knobs:

| Environment variable | Purpose | Default |
| --- | --- | --- |
| `TRAINING_DATA` | Path to labelled dataset used for bootstrapping the service | `data/processed/transactions_split.csv` |
| `TARGET_COLUMN` | Target column for fraud labels | `is_fraud` |
| `PREDICTION_LOG` | Destination for encrypted audit entries | `reports/predictions.enc` |
| `BATCH_SIZE` | Batch size for inference chunks | `512` |

See `tests/test_deployment.py` for reference usage patterns and security regressions.
