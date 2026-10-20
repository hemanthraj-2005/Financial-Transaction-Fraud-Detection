"""Model definitions, threshold tuning and evaluation.

Imbalance handling: class weights / scale_pos_weight (no resampling).
Metrics: precision, recall, F1, ROC-AUC, PR-AUC (primary) and the confusion matrix.
Accuracy is not reported: predicting "legitimate" always already scores ~99%.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import RANDOM_SEED
from src.utils.errors import FraudProjectError
from src.utils.logging_utils import get_logger
from src.utils.validation import require_dependency

logger = get_logger("train")

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
LABEL = "isFraud"


def signed_log1p(x):
    """Stateless transform that tames heavy-tailed amounts and counts (keeps the sign)."""
    return np.sign(x) * np.log1p(np.abs(x))


def to_matrix(df: pd.DataFrame, columns) -> np.ndarray:
    matrix = df[list(columns)].to_numpy(dtype=np.float64)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def build_model(name: str, scale_pos_weight: float = 1.0, seed: int = RANDOM_SEED):
    if name == "logistic_regression":
        # scaler is part of the Pipeline, so it is fitted on the training data only
        return Pipeline(
            [
                ("log", FunctionTransformer(signed_log1p, validate=False)),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        )
    if name == "xgboost":
        require_dependency("xgboost", "xgboost", "the XGBoost model")
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=float(scale_pos_weight), tree_method="hist", eval_metric="aucpr",
            n_jobs=-1, random_state=seed, verbosity=0,
        )
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")


def tune_threshold(y_true, scores) -> tuple[float, float]:
    """Threshold that maximises F1 on the validation window. Returns (threshold, f1)."""
    y = np.asarray(y_true).astype(int)
    if y.sum() == 0:
        raise FraudProjectError("The validation window has no fraud cases, so a threshold cannot be tuned.")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    p, r = precision[:-1], recall[:-1]
    denominator = p + r
    f1 = np.divide(2 * p * r, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def evaluate(y_true, scores, threshold: float) -> dict:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    has_both = 0 < y.sum() < len(y)
    return {
        "n_rows": int(len(y)),
        "n_fraud": int(y.sum()),
        "threshold": float(threshold),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "roc_auc": round(float(roc_auc_score(y, s)), 6) if has_both else None,
        "pr_auc": round(float(average_precision_score(y, s)), 6) if y.sum() > 0 else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "alert_rate_pct": round(100.0 * float(pred.mean()), 4) if len(pred) else 0.0,
    }


def feature_importance(model, name: str, columns: list[str], top: int = 15) -> list[dict]:
    """Logistic regression: |standardised coefficient|. Trees: built-in importance."""
    if name == "logistic_regression":
        values = np.abs(model.named_steps["clf"].coef_[0])
    else:
        values = np.asarray(model.feature_importances_, dtype=float)
    series = pd.Series(values, index=list(columns)).sort_values(ascending=False).head(top)
    return [{"feature": f, "importance": round(float(v), 6)} for f, v in series.items()]


@dataclass
class FittedModel:
    name: str
    model: object
    columns: list
    threshold: float
    fit_seconds: float
    val: dict
    test: dict
    importance: list = field(default_factory=list)


def fit_and_evaluate(
    name: str, columns: list[str], fit: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, seed: int = RANDOM_SEED
) -> FittedModel:
    """Fit on the fit window, tune the threshold on validation, evaluate on validation and test."""
    y_fit = fit[LABEL].to_numpy(dtype=int)
    n_pos = int(y_fit.sum())
    if n_pos == 0:
        raise FraudProjectError("The fit window has no fraud cases, so a model cannot be trained.")
    ratio = (len(y_fit) - n_pos) / n_pos

    model = build_model(name, scale_pos_weight=ratio, seed=seed)
    started = time.perf_counter()
    model.fit(to_matrix(fit, columns), y_fit)
    fit_seconds = time.perf_counter() - started

    p_val = model.predict_proba(to_matrix(val, columns))[:, 1]
    p_test = model.predict_proba(to_matrix(test, columns))[:, 1]
    y_val = val[LABEL].to_numpy(dtype=int)
    y_test = test[LABEL].to_numpy(dtype=int)
    threshold, _ = tune_threshold(y_val, p_val)

    logger.info("%-20s %2d features | fit %.1fs | val PR-AUC %s", name, len(columns), fit_seconds,
                evaluate(y_val, p_val, threshold)["pr_auc"])
    return FittedModel(
        name=name,
        model=model,
        columns=list(columns),
        threshold=threshold,
        fit_seconds=fit_seconds,
        val=evaluate(y_val, p_val, threshold),
        test=evaluate(y_test, p_test, threshold),
        importance=feature_importance(model, name, columns),
    )


def select_best(results: list[FittedModel]) -> FittedModel:
    """Pick the model with the best VALIDATION PR-AUC (ties: higher validation F1). Never uses test."""
    return max(results, key=lambda r: (r.val["pr_auc"] or 0.0, r.val["f1"]))