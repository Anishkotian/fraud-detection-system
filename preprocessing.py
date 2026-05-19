"""
preprocessing.py — Data Preprocessing Pipeline
================================================
Handles all data cleaning, encoding, scaling, and SMOTE balancing.

WHY PREPROCESSING MATTERS IN FRAUD DETECTION:
- Raw transaction data has mixed types (numbers + categories)
- ML models require all-numeric input
- Features like 'transaction_amount' can be 1000x larger than 'is_foreign_transaction',
  which biases distance-based models → StandardScaler fixes this
- 5% fraud rate means models can cheat by predicting all-legit → SMOTE fixes this
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Expected categorical columns that need encoding
CATEGORICAL_COLS = ["merchant_category", "location", "device_type", "transaction_type"]

# Numeric columns used for scaling
NUMERIC_COLS = [
    "transaction_amount", "transaction_time", "account_age_days",
    "num_transactions_today", "distance_from_home_km", "is_foreign_transaction"
]

TARGET_COL = "is_fraud"


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and do basic sanity checks."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values with sensible defaults:
    - Numeric: median (robust to outliers)
    - Categorical: mode (most frequent category)
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"  Filled numeric '{col}' with median={median_val:.2f}")
            else:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"  Filled categorical '{col}' with mode='{mode_val}'")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    Returns encoded df + a dict of encoders for inverse transform / new data.
    """
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info(f"  Encoded '{col}': {list(le.classes_)}")
    return df, encoders


def get_feature_columns(df: pd.DataFrame) -> list:
    """Dynamically determine which columns to use as features (exclude target)."""
    exclude = [TARGET_COL]
    return [c for c in df.columns if c not in exclude]


def preprocess_pipeline(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline:
    1. Handle missing values
    2. Encode categoricals
    3. Split features / target
    4. Train-test split
    5. Scale numeric features
    6. Apply SMOTE to training set

    WHY SMOTE (Synthetic Minority Over-sampling TEchnique)?
    --------------------------------------------------------
    With 95% legit / 5% fraud, a model that always predicts "legit" hits 95% accuracy
    but catches ZERO frauds — useless. SMOTE creates synthetic fraud samples by
    interpolating between existing fraud cases in feature space, giving the model
    enough fraud examples to learn meaningful patterns without simply duplicating rows.

    Returns: X_train, X_test, y_train, y_test, scaler, encoders, feature_names
    """
    logger.info("Starting preprocessing pipeline...")

    df = handle_missing_values(df)
    df, encoders = encode_categoricals(df)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[TARGET_COL]

    logger.info(f"Features: {feature_cols}")
    logger.info(f"Class distribution — Legit: {(y==0).sum()}, Fraud: {(y==1).sum()}")

    # Train-test split BEFORE SMOTE to prevent data leakage
    # (test set must reflect real-world imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numeric features — StandardScaler: mean=0, std=1
    # WHY: Logistic Regression is sensitive to feature magnitude;
    # Random Forest is not, but scaling doesn't hurt it either.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # fit only on train → no leakage

    # Apply SMOTE only to training data
    logger.info(f"Before SMOTE — Train fraud: {y_train.sum()}, legit: {(y_train==0).sum()}")
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    logger.info(f"After SMOTE  — Train fraud: {y_train_balanced.sum()}, legit: {(y_train_balanced==0).sum()}")

    return (
        X_train_balanced, X_test_scaled,
        y_train_balanced, y_test,
        scaler, encoders, feature_cols
    )


def preprocess_single_transaction(transaction: dict, scaler: StandardScaler, encoders: dict, feature_cols: list) -> np.ndarray:
    """
    Transform a single transaction dict into a scaled feature vector for prediction.
    Used by the real-time prediction form.
    """
    df = pd.DataFrame([transaction])

    for col, le in encoders.items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            # Handle unseen labels gracefully
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                df[col] = 0  # fallback to first class
                logger.warning(f"Unseen label '{val}' for '{col}', defaulting to 0")

    # Ensure correct column order
    df = df.reindex(columns=feature_cols, fill_value=0)
    scaled = scaler.transform(df)
    return scaled


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict for dashboard display."""
    fraud_count = int(df[TARGET_COL].sum()) if TARGET_COL in df.columns else 0
    total = len(df)
    return {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "legit_count": total - fraud_count,
        "fraud_rate": round(fraud_count / total * 100, 2) if total > 0 else 0,
        "missing_values": int(df.isnull().sum().sum()),
        "features": list(df.columns),
        "numeric_stats": df.describe().to_dict()
    }
