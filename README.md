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
