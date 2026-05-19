"""
predict.py — Prediction & Risk Scoring Module
===============================================
Handles single transaction and batch prediction with risk tagging.
"""

import numpy as np
import pandas as pd
import pickle
import os
import logging
from preprocessing import preprocess_single_transaction, preprocess_pipeline

logger = logging.getLogger(__name__)

MODELS_DIR = "models"

# Risk thresholds — tunable by threshold slider in UI
# WHY THRESHOLDS MATTER:
# Default 0.5 cutoff isn't always optimal.
# Banks can lower threshold (e.g., 0.3) to catch more fraud (higher recall)
# at the cost of more false positives. A slider lets analysts tune this trade-off.
DEFAULT_THRESHOLD = 0.5


def get_risk_level(probability: float) -> tuple[str, str]:
    """
    Map fraud probability to a human-readable risk level.
    Used for color-coded risk badges in the UI.
    
    Returns: (risk_label, color_hex)
    """
    if probability < 0.3:
        return "🟢 Low Risk", "#22c55e"
    elif probability < 0.6:
        return "🟡 Medium Risk", "#f59e0b"
    else:
        return "🔴 High Risk", "#ef4444"


def predict_single(
    transaction: dict,
    model,
    scaler,
    encoders: dict,
    feature_cols: list,
    threshold: float = DEFAULT_THRESHOLD
) -> dict:
    """
    Predict fraud probability for a single transaction.
    Returns a rich result dict with probability, risk level, and raw scores.
    """
    X = preprocess_single_transaction(transaction, scaler, encoders, feature_cols)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0][1]
    else:
        # Isolation Forest: use decision_function (flip sign)
        score = model.decision_function(X)[0]
        # Normalize to [0, 1] range roughly
        proba = float(np.clip(1 / (1 + np.exp(score * 2)), 0, 1))

    is_fraud = int(proba >= threshold)
    risk_label, risk_color = get_risk_level(proba)

    return {
        "fraud_probability": round(float(proba), 4),
        "is_fraud": is_fraud,
        "risk_level": risk_label,
        "risk_color": risk_color,
        "threshold_used": threshold,
        "verdict": "⚠️ FRAUD DETECTED" if is_fraud else "✅ LEGITIMATE"
    }


def predict_batch(
    df: pd.DataFrame,
    model,
    scaler,
    encoders: dict,
    feature_cols: list,
    threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """
    Batch prediction on an uploaded CSV.
    Adds 'fraud_probability', 'predicted_fraud', 'risk_level' columns.
    Returns the enriched DataFrame for download.
    """
    results = []
    for _, row in df.iterrows():
        transaction = row.to_dict()
        # Remove target column if present (user might upload labeled data)
        transaction.pop("is_fraud", None)
        try:
            result = predict_single(transaction, model, scaler, encoders, feature_cols, threshold)
            results.append({
                "fraud_probability": result["fraud_probability"],
                "predicted_fraud": result["is_fraud"],
                "risk_level": result["risk_level"],
                "verdict": result["verdict"]
            })
        except Exception as e:
            logger.warning(f"Prediction error on row: {e}")
            results.append({
                "fraud_probability": 0.0,
                "predicted_fraud": 0,
                "risk_level": "Unknown",
                "verdict": "Error"
            })

    result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    return result_df


def predict_with_all_models(
    transaction: dict,
    models: dict,
    scaler,
    encoders: dict,
    feature_cols: list,
    threshold: float = DEFAULT_THRESHOLD
) -> dict:
    """
    Run a transaction through all loaded models and return a comparison.
    Useful for ensemble decision-making and showing model agreement.
    """
    all_results = {}
    fraud_votes = 0
    for model_name, model in models.items():
        result = predict_single(transaction, model, scaler, encoders, feature_cols, threshold)
        all_results[model_name] = result
        if result["is_fraud"]:
            fraud_votes += 1

    # Ensemble verdict: majority vote
    ensemble_fraud = fraud_votes >= 2
    return {
        "individual": all_results,
        "ensemble_fraud": ensemble_fraud,
        "fraud_votes": fraud_votes,
        "total_models": len(models),
        "ensemble_verdict": "⚠️ FRAUD (Ensemble)" if ensemble_fraud else "✅ LEGITIMATE (Ensemble)"
    }


def load_all_models() -> dict:
    """Load all saved models from disk. Returns empty dict if not yet trained."""
    model_files = {
        "Logistic Regression": "logistic_model.pkl",
        "Random Forest": "random_forest.pkl",
        "Isolation Forest": "isolation_forest.pkl"
    }
    loaded = {}
    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded[name] = pickle.load(f)
            logger.info(f"Loaded model: {name}")
        else:
            logger.warning(f"Model not found: {path}")
    return loaded
