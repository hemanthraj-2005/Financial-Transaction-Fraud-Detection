# Financial Transaction Fraud Detection Using Graph Analytics and Machine Learning

Fraud is rare (well under 1% of transactions) but costly. Row-by-row models miss fraud that only shows up in the
*relationships* between accounts: layering chains, mule accounts and circular rings. This project models
transactions as a directed, weighted graph and combines graph-derived account features with transaction features
to score transactions and accounts, with human-readable reasons, in a Streamlit dashboard.

> **This is batch analytics on a simulated dataset, not a real-time or production fraud system.**

## Highlights

- PySpark (local mode) for cleaning, aggregation and account features, with a Pandas fallback for small data
- Directed, weighted transaction graph (NetworkX): degree, weighted PageRank, approximate betweenness,
  pass-through ratio, reciprocal partners, cycle groups
- Louvain community detection (greedy-modularity fallback) with a label-free suspicion score
- Leakage-safe ML: time-based split, graph features from strictly earlier time blocks, ablation study
- Logistic Regression, Random Forest and XGBoost, compared with PR-AUC, precision, recall, F1 and ROC-AUC
- 0-100 risk score with bands, reason codes, and account-level risk
- SQLite storage and a Streamlit dashboard (6 pages + Investigation view)
- Tests with pytest

## Architecture

```
Raw data (PaySim / synthetic) -> PySpark or Pandas (clean, features, aggregates) -> Parquet
    -> Graph layer (NetworkX): metrics + communities
    -> ML layer (scikit-learn, XGBoost): time split, snapshot features, models, ablation
    -> Risk scoring: 0-100 score, reason codes, account risk -> SQLite
    -> Streamlit + Plotly dashboard (reads precomputed results only)
```

## Setup

Requirements: Python 3.10+, and **JDK 17** if you want the Spark engine (`java -version`).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

The main dataset is **PaySim** (Kaggle: "Synthetic Financial Datasets For Fraud Detection", ~6.3M rows).
Download it manually from https://www.kaggle.com/datasets/ealaxi/paysim1 and put the CSV in `data/raw/`,
or run `python -m src.data.download` (needs a Kaggle API token).

PaySim is a *simulation* derived from real mobile-money logs, not raw bank data. Fraud occurs only in TRANSFER and
CASH_OUT, and its balance columns reveal the label, which is why they are excluded from the main model.

If PaySim is missing, the pipeline can fall back to a clearly labelled **synthetic development dataset**. Results
from synthetic data must never be reported; the pipeline and dashboard show a warning whenever it is used.

## Run everything

```bash
python run_pipeline.py                                   # all steps (PaySim if present, else synthetic)
python run_pipeline.py --source synthetic                # development data only
python run_pipeline.py --source paysim --max-tx 150000 --blocks 6   # laptop-friendly real run
python run_pipeline.py --from ml                         # re-run from a step onwards
python run_pipeline.py --dry-run                         # show the plan only
streamlit run dashboard/app.py                           # open the dashboard
```

The full PaySim file is large. The graph and ML steps use the first N transactions by time (`--max-tx`, default
300,000 for ML and 500,000 for the graph) to keep a laptop run feasible, and the run reports when it truncates.
Louvain and PageRank run once per time block, so start small.

## Steps one by one

| Step | Command | Main outputs |
|---|---|---|
| Preprocess | `python -m src.preprocessing.pipeline` | `data/processed/transactions_clean`, `account_aggregates` |
| EDA | `python -m src.analytics.eda` | `reports/eda/*.html` |
| Graph | `python -m src.graph.pipeline` | `graph_edges`, `graph_metrics`, `reports/graph_report.json` |
| Communities | `python -m src.graph.community_pipeline` | `community_membership`, `community_stats` |
| ML | `python -m src.models.pipeline` | `models/best_model.joblib`, `reports/model_comparison.csv`, `ml_report.json` |
| Score | `python -m src.models.scoring_pipeline` | `data/processed/fraud.db`, `reports/risk_report.json` |

## Tests

```bash
pytest
```

## How leakage is controlled

- Time-based split by step: fit (first part), validation (threshold and model selection), test (untouched).
- Graph features for a transaction come from a graph built on strictly earlier time blocks only.
- The only label-based feature (community fraud rate) uses labels from earlier blocks and from the fit window only.
- Balance columns are excluded from the primary model and shown only in an ablation; `isFlaggedFraud` is never used.
- Scalers are fitted on the fit window only; the model is chosen on validation PR-AUC, never on the test window.

## Project structure

```
config.py  run_pipeline.py  requirements.txt  pytest.ini  README.md
src/
  data/           synthetic_generator, loader, download
  preprocessing/  schema, cleaning, diagnostics, scaling, spark_pipeline, pipeline
  features/       transaction_features, account_aggregates, time_split, graph_features, feature_sets
  analytics/      fraud_stats, eda
  graph/          construction, metrics, pipeline, communities, community_stats, community_pipeline
  models/         train, pipeline, explain, risk, scoring_pipeline
  db/             database, queries, analytics
  utils/          errors, logging_utils, seed, validation, io_utils
dashboard/        app.py, loaders.py, data.py, charts.py, ui.py, pages/
tests/            one test module per phase
data/  models/  reports/  docs/
```

## Limitations

- Batch analytics on a simulated dataset; not real-time and not validated on real bank data.
- Betweenness is approximate (sampled); NetworkX keeps the graph in memory, so very large graphs need
  GraphFrames/GraphX or a graph database in production.
- Risk bands and the community suspicion score are project-defined heuristics, not validated probabilities.
- Reason codes describe unusual features; they are not causal explanations or proof of fraud.
- Fraud labels are assumed known right after each block; real systems have label delays and concept drift.
- Account-level "fraud" means "touches a fraud transaction", so it includes victims as well as mules.

## Troubleshooting

- **Spark errors:** install JDK 17 and restart the terminal. The pipeline falls back to Pandas with a warning.
- **pandas 3 + PySpark warnings:** harmless deprecation notices. `pip install "pandas<3"` silences them.
- **Dashboard says the database is missing:** run `python run_pipeline.py` (or the score step) first.
- **Memory problems:** lower `--max-tx` and `--blocks`, or set `FRAUD_SPARK_MEMORY=6g`.