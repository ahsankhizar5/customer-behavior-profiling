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
