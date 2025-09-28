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
