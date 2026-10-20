# Financial Transaction Fraud Detection Using Graph Analytics and Machine Learning

## Abstract
Financial fraud is rare but costly, and models that score each transaction in isolation miss fraud that appears only in
the relationships between accounts. This project represents transactions as a directed, weighted graph and combines
graph-derived account features with transaction-level features to detect fraudulent transactions and suspicious
accounts. A PySpark pipeline cleans the PaySim dataset and builds account features; a NetworkX layer computes degree,
PageRank, approximate betweenness and communities; and Logistic Regression, Random Forest and XGBoost are compared
using fraud-appropriate metrics on a leakage-safe, time-based split. Outputs are a 0-100 risk score with reason codes,
stored in SQLite and explored in a Streamlit dashboard. [One sentence with your headline test-window result, e.g.
"The selected model reached PR-AUC X on the test window."]

## 1. Introduction
Fraud accounts for well under 1% of transactions but causes disproportionate losses. Rule-based systems and row-by-row
machine learning inspect each transaction on its own. Fraud schemes such as layering chains, mule accounts and
circular rings are structural: they are visible only when accounts and the money flowing between them are modelled
together. Graph analytics supplies exactly this view.

## 2. Problem statement
Detect fraudulent transactions and suspicious accounts in a large transaction dataset by combining transaction
features with graph features, while avoiding data leakage and explaining each alert in plain language.

## 3. Objectives
1. Build a reproducible pipeline from raw transactions to risk scores.
2. Use PySpark for large-scale cleaning, aggregation and account features.
3. Build a transaction graph and compute degree, betweenness and PageRank.
4. Detect communities and flag suspicious ones.
5. Train and compare Logistic Regression, Random Forest and XGBoost with a leakage-safe, time-based split.
6. Produce a 0-100 risk score with reason codes.
7. Deliver a dashboard with an account investigation view.

## 4. Background
- **Class imbalance.** With fraud near 0.13%, a model that always predicts "legitimate" is about 99.87% accurate, so
  accuracy is meaningless. Precision, recall, F1 and especially PR-AUC are used.
- **Graph features.** Degree measures how many counterparties an account has; PageRank measures importance in the
  money flow; betweenness measures how often an account lies on paths between others; pass-through ratio compares money
  in and out (mule behaviour); strongly connected groups indicate circular flows.
- **Community detection.** Louvain optimises modularity to find groups of accounts that trade mostly among themselves.
- **Data leakage.** Features or splits that let information from the future or from the label reach the model give
  unrealistically good results. Time-based splitting and snapshot features prevent this.

## 5. Dataset
PaySim (Kaggle, about 6.3 million rows, 743 hourly steps) is a simulator calibrated on real mobile-money logs. It has
sender and receiver account IDs, a time step, a transaction type and a fraud label, which makes graph construction and
a time-based split possible. Limitations: it is simulated; fraud occurs only in TRANSFER and CASH_OUT; its balance
columns reveal the label; and the built-in `isFlaggedFraud` is a rule-based flag, so it is excluded.
A clearly labelled synthetic generator is used only for development and testing, never for reported results.

## 6. System design
Raw data -> PySpark/Pandas preprocessing -> Parquet -> graph layer (NetworkX) -> ML layer -> risk scoring ->
SQLite -> Streamlit dashboard. The dashboard reads precomputed outputs only, so it responds in seconds. The pipeline
runs end to end with `python run_pipeline.py`.

## 7. Methodology
**7.1 Preprocessing.** Explicit-schema loading; nine validity rules (nulls, negative amounts, invalid step/type/label,
negative balances, self-transfers, blank IDs); duplicate removal; time features; past-only account history features.
**7.2 Graph construction.** Accounts are nodes; each sender-receiver pair is one directed edge with total amount and
transaction count. Metrics: degree, weighted PageRank, sampled betweenness, pass-through ratio, reciprocal partners,
strongly connected component sizes.
**7.3 Communities.** Louvain on the undirected, log-amount-weighted projection with a greedy-modularity fallback.
A label-free suspicion score ranks communities by internal flow share, density, pass-through behaviour and cycles.
**7.4 Leakage-safe features.** The time range is cut into blocks; a transaction's graph and community features come from
a graph built on earlier blocks only. The only label-based feature, a smoothed community fraud rate, uses labels from
earlier blocks within the fit window.
**7.5 Models.** Logistic Regression (class weights), Random Forest (balanced subsample) and XGBoost (scale_pos_weight).
The decision threshold maximises F1 on the validation window; the model is selected by validation PR-AUC.
**7.6 Ablation.** The selected model is retrained on three feature sets: transaction only, transaction + graph, and
transaction + graph + balance columns (leaky), to measure what the graph adds and to demonstrate leakage.
**7.7 Risk score and reasons.** The model probability is mapped piecewise-linearly to 0-100, anchored so that the decision
threshold equals 61. Bands: Low 0-30, Medium 31-60, High 61-80, Critical 81-100 (project-defined). Reason codes list the
features that are unusually high compared with legitimate fit-window activity, ranked by model importance. An account's
risk is the maximum risk of its transactions.

## 8. Implementation
Python 3.13, Pandas, NumPy, PyArrow, PySpark (local mode), NetworkX, python-louvain, scikit-learn, XGBoost, Plotly,
Streamlit and SQLite. The code is modular (data, preprocessing, features, graph, models, db, dashboard) and covered by
pytest. Random seeds and paths are configuration-driven for reproducibility.

## 9. Results
Report only results from the real PaySim run (not synthetic data). Use `reports/ml_report.json`,
`reports/model_comparison.csv` and the dashboard.

| Model (feature set tx_graph) | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | [fill] | [fill] | [fill] | [fill] | [fill] |
| Random Forest | [fill] | [fill] | [fill] | [fill] | [fill] |
| XGBoost | [fill] | [fill] | [fill] | [fill] | [fill] |

Ablation for the selected model: tx_only [PR-AUC], tx_graph [PR-AUC], tx_graph_balance [PR-AUC, inflated by leakage].
Also include: dataset size and fraud rate, the split boundaries, graph size (nodes, edges), number of communities and
flagged communities, and 2-3 dashboard screenshots (Overview, Suspicious Accounts, Investigation).
Discuss honestly whether the graph features improved over tx_only on the test window; on PaySim the gain may be small
because most sender accounts appear only once.

## 10. Discussion and limitations
Batch analytics on simulated data; not real-time and not validated on real bank data. Betweenness is approximate.
NetworkX is memory-bound; production scale would need GraphFrames/GraphX or a graph database. Risk bands and the
community score are heuristics. Reason codes are descriptive, not causal. Labels are assumed to arrive immediately.
Account-level "fraud" includes victims as well as mules.

## 11. Conclusion and future work
The project shows an end-to-end, reproducible way to combine graph analytics with machine learning for fraud detection
under a leakage-safe evaluation. Future work: streaming scoring, label-delay handling and drift monitoring, graph neural
networks, scalable graph engines, calibration of risk bands, and analyst-feedback loops.

## References (verify the details before citing)
- Lopez-Rojas, E., Elmir, A., Axelsson, S. (2016). PaySim: A financial mobile money simulator for fraud detection.
- Blondel, V. et al. (2008). Fast unfolding of communities in large networks. J. Stat. Mech.
- Brin, S., Page, L. (1998). The anatomy of a large-scale hypertextual web search engine.
- Brandes, U., Pich, C. (2007). Centrality estimation in large networks.
- Breiman, L. (2001). Random forests. Machine Learning.
- Chen, T., Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
- Saito, T., Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating
  binary classifiers on imbalanced datasets. PLoS ONE.