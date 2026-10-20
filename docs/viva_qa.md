# Viva questions and short answers

**1. What problem does the project solve?**
It detects fraudulent transactions and suspicious accounts by looking at relationships between accounts, not just single
transactions, and explains each alert with reason codes.

**2. Why graph analytics?**
Schemes like layering chains, mule accounts and circular rings are structural. They are invisible to row-by-row models
but visible in degree, pass-through behaviour, cycles and communities.

**3. Why PaySim, and what are its limits?**
It has sender/receiver IDs (needed for a graph), a time column (needed for a time split), types and fraud labels, and it
is large enough to justify Spark. It is a simulation, fraud occurs only in TRANSFER and CASH_OUT, and its balance
columns reveal the label.

**4. Why PySpark if you run on a laptop? What changes on a cluster?**
Spark runs in local mode with the same code that would run on a cluster. On a cluster you change the master URL and read
from distributed storage. Pandas would hit single-machine memory limits on very large data.

**5. Why NetworkX instead of Neo4j or GraphFrames?**
It is simple, runs on a laptop and keeps the demo reproducible. It is memory-bound, so at production scale I would move to
GraphFrames/GraphX or a graph database.

**6. Which graph metrics did you use and why?**
Degree (counterparties), weighted PageRank (importance in the money flow), approximate betweenness (bridging), pass-through
ratio (money in vs out, mule-like), reciprocal partners and strongly connected groups (circular flow).

**7. Why is betweenness approximate?**
Exact betweenness is roughly O(V*E), infeasible on large graphs. I sample start nodes (k pivots) on the largest connected
component and document it.

**8. What is Louvain, and what is the fallback?**
Louvain greedily optimises modularity to find communities. If python-louvain is unavailable or fails, the pipeline uses
NetworkX greedy modularity and records the reason.

**9. Is the community suspicion score a probability?**
No. It is a label-free, project-defined heuristic that ranks communities relative to each other.

**10. Why not accuracy?**
With about 0.13% fraud, always predicting "legitimate" is about 99.87% accurate. I use precision, recall, F1, ROC-AUC and
mainly PR-AUC.

**11. How did you handle class imbalance?**
Class weights (Logistic Regression, Random Forest) and scale_pos_weight (XGBoost), with no resampling. Threshold tuning is
done on a validation window.

**12. Why a time-based split?**
Deployment predicts the future from the past. A random split lets future information leak into training and inflates scores.

**13. What is data leakage and where could it occur here?**
Information from the future or from the label reaching the features. Risks here: graph metrics computed on all data,
community fraud rates, balance columns, `isFlaggedFraud`, and scalers fitted on all data.

**14. How do you prevent leakage?**
Time-based split; graph and community features from strictly earlier time blocks; label-based features only from earlier
blocks in the fit window; balance columns and `isFlaggedFraud` excluded; scaler fitted on the fit window; model chosen on
validation, not test. Tests check that changing future data or labels does not change earlier features.

**15. Explain the snapshot (block) idea.**
The time range is split into blocks. A transaction in block b gets its graph features from a graph of blocks before b, so it
never sees its own block or later ones. Block 0 has no history and is dropped.

**16. Why exclude the balance features? What does the ablation show?**
In PaySim, fraud empties the sender's account, so balances reveal the label. The ablation retrains with and without them
to show the inflated score honestly.

**17. How did you choose the model and the threshold?**
The model with the best validation PR-AUC; the threshold maximises F1 on the validation window. Test numbers are reported
once.

**18. Did the graph features help?**
[Answer from your own ablation: compare tx_only vs tx_graph on the test window.] On PaySim the gain can be small because
most senders appear only once, so I report whatever the ablation shows.

**19. How is the 0-100 risk score computed? Are the bands validated?**
A piecewise-linear mapping of the model probability, anchored so the decision threshold equals 61. Bands are
project-defined and calibratable, not scientifically validated.

**20. How are reason codes generated? Are they causal?**
For Medium+ transactions, rules fire when a feature is above the 95th percentile of legitimate fit-window activity (or a
rare flag is set), ranked by model importance. They describe unusual features; they are not causal or proof of fraud.

**21. How is account risk defined, and what is its limit?**
The maximum risk of any transaction touching the account. "Fraud account" in evaluation means touching a fraud
transaction, so it includes victims as well as mules.

**22. Why SQLite?**
It is enough for this scale, needs no server, and lets the dashboard read precomputed results quickly. PostgreSQL would be
overkill here.

**23. Is this real-time?**
No. It is batch analytics. Real-time scoring would need streaming ingestion and incremental graph updates.

**24. How do you ensure reproducibility?**
Fixed seeds, config-driven paths, one-command pipeline (`run_pipeline.py`) and a pytest suite.

**25. Why a synthetic fallback dataset?**
So the code and tests run before PaySim is downloaded. It is labelled everywhere, falls back only in "auto" mode, and its
results are never reported.

**26. What testing did you do?**
Unit tests for cleaning, features, graph metrics, communities, leakage safety, models, risk scoring, database queries,
charts and orchestration; a Spark-vs-Pandas parity test; an end-to-end test of the ML and scoring pipelines.

**27. What are the main limitations?**
Simulated data, batch only, approximate betweenness, in-memory graph, heuristic bands and community score, assumed
immediate labels, descriptive (non-causal) reasons.

**28. What would you do in production?**
Streaming scoring, label-delay handling, drift monitoring, scalable graph processing, calibrated risk bands, analyst
feedback loops and maybe graph neural networks.