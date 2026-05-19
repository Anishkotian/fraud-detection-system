"""
train.py — Model Training Module
==================================
Trains three models and saves them as .pkl files.

WHY THREE MODELS?
-----------------
1. Logistic Regression  → Linear, fast, interpretable baseline. Good for understanding
                          which features push toward fraud. Easy to explain in interviews.

2. Random Forest        → Ensemble of decision trees, handles non-linear patterns,
                          naturally resistant to overfitting, gives feature importances.
                          Usually best performer for structured fraud data.

3. Isolation Forest     → Unsupervised ANOMALY DETECTION. Doesn't need labels.
                          Useful when you have very few fraud samples or want to
                          detect completely new fraud patterns not seen in training.

CLASSIFICATION vs ANOMALY DETECTION:
- Classification (LR, RF): Learn from labeled fraud/legit examples → supervised
- Anomaly Detection (IF): Assumes fraud is "rare and different" → unsupervised
  Isolation Forest isolates anomalies using random splits; rare points need fewer
  splits to isolate → lower "anomaly score".
"""

import os
import pickle
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, f1_score
)

logger = logging.getLogger(__name__)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_logistic_regression(X_train, y_train):
    """
    Logistic Regression — linear probabilistic classifier.
    - class_weight='balanced': re-weights loss for fraud vs legit (extra safety)
    - max_iter=1000: more iterations for convergence on complex data
    - C=0.1: regularization strength (prevents overfitting)
    """
    logger.info("Training Logistic Regression...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=0.1,
        random_state=42,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete.")
    return model


def train_random_forest(X_train, y_train):
    """
    Random Forest — ensemble of 200 decision trees.
    - n_estimators=200: more trees → more stable predictions
    - max_depth=15: prevents individual trees from memorizing noise
    - class_weight='balanced': handles any residual imbalance post-SMOTE
    - n_jobs=-1: use all CPU cores
    """
    logger.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest training complete.")
    return model


def train_isolation_forest(X_train):
    """
    Isolation Forest — unsupervised anomaly detector.
    - contamination=0.05: expected fraud rate (5%)
    - n_estimators=100: number of isolation trees
    - Doesn't use labels → purely learns what 'normal' looks like
    """
    logger.info("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)  # No y_train — unsupervised!
    logger.info("Isolation Forest training complete.")
    return model


def evaluate_classifier(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a supervised classifier on the test set.
    Returns metrics dict for dashboard comparison.

    WHY PRECISION AND RECALL?
    --------------------------
    Accuracy is misleading on imbalanced data (95% baseline).
    - Precision = Of predicted frauds, how many are actually fraud?
      Low precision → too many false alarms (annoyed customers)
    - Recall = Of actual frauds, how many did we catch?
      Low recall → missed frauds → financial losses
    - F1 = Harmonic mean of Precision & Recall → balanced metric
    - ROC-AUC → overall discrimination ability at all thresholds

    FALSE POSITIVES vs FALSE NEGATIVES in fraud:
    - FP (predict fraud, actually legit): customer card blocked → friction, complaints
    - FN (predict legit, actually fraud): bank loses money → the worse outcome
    → Banks usually optimize for high Recall at the cost of some Precision.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "model_name": model_name,
        "accuracy": round(report["accuracy"], 4),
        "precision": round(report.get("1", {}).get("precision", 0), 4),
        "recall": round(report.get("1", {}).get("recall", 0), 4),
        "f1_score": round(report.get("1", {}).get("f1-score", 0), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_proba)
        metrics.update({
            "roc_auc": round(auc, 4),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_thresholds": roc_thresholds.tolist(),
            "precision_curve": precision_curve.tolist(),
            "recall_curve": recall_curve.tolist(),
        })
        metrics["y_proba"] = y_proba.tolist()

    logger.info(f"{model_name} | Acc={metrics['accuracy']} | Prec={metrics['precision']} | Rec={metrics['recall']} | F1={metrics['f1_score']}")
    return metrics


def evaluate_isolation_forest(model, X_test, y_test) -> dict:
    """
    Evaluate Isolation Forest by converting its -1/1 predictions to 0/1 labels.
    Isolation Forest returns: -1 for anomaly (fraud), +1 for normal (legit)
    We flip to: 1 = fraud, 0 = legit to match our label convention.
    """
    raw_preds = model.predict(X_test)
    y_pred = np.where(raw_preds == -1, 1, 0)  # -1 → fraud (1), 1 → legit (0)

    # Anomaly scores (lower = more anomalous = more likely fraud)
    scores = model.decision_function(X_test)
    # Flip scores so higher = more fraud-like
    fraud_scores = -scores

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, fraud_scores)
        fpr, tpr, roc_thresholds = roc_curve(y_test, fraud_scores)
    except Exception:
        auc = 0.5
        fpr, tpr, roc_thresholds = [0, 1], [0, 1], [1, 0]

    metrics = {
        "model_name": "Isolation Forest",
        "accuracy": round(report["accuracy"], 4),
        "precision": round(report.get("1", {}).get("precision", 0), 4),
        "recall": round(report.get("1", {}).get("recall", 0), 4),
        "f1_score": round(report.get("1", {}).get("f1-score", 0), 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "fpr": fpr if isinstance(fpr, list) else fpr.tolist(),
        "tpr": tpr if isinstance(tpr, list) else tpr.tolist(),
        "roc_thresholds": roc_thresholds if isinstance(roc_thresholds, list) else roc_thresholds.tolist(),
        "y_proba": fraud_scores.tolist(),
    }
    logger.info(f"Isolation Forest | Acc={metrics['accuracy']} | Prec={metrics['precision']} | Rec={metrics['recall']} | F1={metrics['f1_score']}")
    return metrics


def get_feature_importances(rf_model, feature_names: list) -> dict:
    """
    Extract feature importances from Random Forest.
    Shows which features most influence fraud classification.
    Higher importance = more discriminative power.
    """
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    return {
        "features": [feature_names[i] for i in sorted_idx],
        "importances": [float(importances[i]) for i in sorted_idx]
    }


def save_model(model, filename: str):
    """Serialize model to disk using pickle."""
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved → {path}")


def load_model(filename: str):
    """Load a pickled model from disk."""
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


def train_all(X_train, X_test, y_train, y_test, feature_names: list) -> dict:
    """
    Master function: train all three models, evaluate, save, return results.
    Called by the Streamlit app when user clicks "Train Models".
    """
    results = {}

    # 1. Logistic Regression
    lr = train_logistic_regression(X_train, y_train)
    results["Logistic Regression"] = evaluate_classifier(lr, X_test, y_test, "Logistic Regression")
    save_model(lr, "logistic_model.pkl")

    # 2. Random Forest
    rf = train_random_forest(X_train, y_train)
    results["Random Forest"] = evaluate_classifier(rf, X_test, y_test, "Random Forest")
    results["feature_importances"] = get_feature_importances(rf, feature_names)
    save_model(rf, "random_forest.pkl")

    # 3. Isolation Forest (trained on all X_train, no labels)
    iso = train_isolation_forest(X_train)
    results["Isolation Forest"] = evaluate_isolation_forest(iso, X_test, y_test)
    save_model(iso, "isolation_forest.pkl")

    return results
