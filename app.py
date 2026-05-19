"""
app.py — Fraud Transaction Detection System
=============================================
Main Streamlit application. Run with: streamlit run app.py

Navigation pages:
  1. 🏠 Home / Dashboard
  2. 📊 Data Explorer
  3. 🤖 Model Training
  4. 🔍 Real-Time Prediction
  5. 📦 Batch Prediction
  6. 📚 Interview Prep
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import logging
import time

from preprocessing import preprocess_pipeline, get_data_summary, load_data
from train import train_all
from predict import predict_single, predict_batch, load_all_models, get_risk_level
from utils import (
    fraud_pie_chart, amount_distribution, fraud_by_category,
    correlation_heatmap, fraud_by_hour, roc_curve_plot,
    confusion_matrix_heatmap, feature_importance_chart,
    model_comparison_chart, probability_gauge, scatter_amount_vs_risk
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg: #0a0f1e;
    --card: #111827;
    --card2: #1a2235;
    --border: #1e3a5f;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --primary: #3b82f6;
    --primary-glow: rgba(59,130,246,0.2);
    --fraud: #ef4444;
    --legit: #22c55e;
    --warn: #f59e0b;
    --purple: #8b5cf6;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a0f1e 100%) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .stRadio > label {
    color: var(--text) !important;
    font-size: 15px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 26px !important; font-weight: 700 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--purple)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px var(--primary-glow) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px var(--primary-glow) !important;
}

/* Cards */
.fraud-card {
    background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}
.legit-card {
    background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.05));
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}
.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}
.section-header {
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
    border-left: 4px solid var(--primary);
    padding-left: 12px;
    margin: 24px 0 16px 0;
}
.badge-fraud {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 13px;
    font-weight: 600;
}
.badge-legit {
    display: inline-block;
    background: rgba(34,197,94,0.15);
    color: #22c55e;
    border: 1px solid rgba(34,197,94,0.4);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 13px;
    font-weight: 600;
}
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 20px 0;
}
.kpi-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.kpi-value { font-size: 32px; font-weight: 700; color: var(--primary); }
.kpi-label { font-size: 12px; color: var(--muted); margin-top: 4px; }
.alert-fraud {
    background: linear-gradient(90deg, rgba(239,68,68,0.2), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.5);
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}
.stSelectbox > div, .stNumberInput > div, .stSlider > div {
    background: var(--card2) !important;
}
code, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--card2) !important;
}
.stDataFrame { background: var(--card) !important; border-radius: 8px; }
.interview-q {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--purple);
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
}
.interview-q h4 { color: var(--purple); margin: 0 0 8px 0; }
.interview-q p { color: var(--text); margin: 0; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "df": None, "trained": False, "results": None,
        "scaler": None, "encoders": None, "feature_cols": None,
        "X_train": None, "X_test": None, "y_train": None, "y_test": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px 0;'>
        <div style='font-size:42px;'>🛡️</div>
        <div style='font-size:18px; font-weight:700; color:#3b82f6;'>FraudGuard AI</div>
        <div style='font-size:11px; color:#64748b;'>Transaction Detection System</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigate", [
        "🏠 Dashboard",
        "📊 Data Explorer",
        "🤖 Model Training",
        "🔍 Real-Time Prediction",
        "📦 Batch Prediction",
        "📚 Interview Prep"
    ])

    st.divider()

    # Dataset upload in sidebar
    st.markdown("**📂 Load Dataset**")
    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"✅ Loaded {len(df):,} rows")
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.df is None:
        if st.button("📥 Use Sample Dataset"):
            if os.path.exists("data/transactions.csv"):
                st.session_state.df = pd.read_csv("data/transactions.csv")
                st.success(f"✅ Loaded sample dataset")
            else:
                st.error("Sample dataset not found. Run: `python data/generate_data.py`")

    if st.session_state.df is not None:
        df = st.session_state.df
        fraud_count = df["is_fraud"].sum() if "is_fraud" in df.columns else 0
        total = len(df)
        st.markdown(f"""
        <div style='background:#1e293b;border-radius:8px;padding:10px 12px;margin-top:8px;font-size:12px;'>
            <div>📄 <b>{total:,}</b> transactions</div>
            <div style='color:#ef4444;'>⚠️ <b>{fraud_count:,}</b> fraudulent ({fraud_count/total*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    if st.session_state.trained:
        st.success("✅ Models Trained")
    else:
        st.warning("⚙️ Models not trained yet")

    st.markdown("""
    <div style='font-size:10px; color:#475569; text-align:center; padding-top:10px;'>
    Built with ❤️ using Streamlit + scikit-learn
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <h1 style='font-size:32px; font-weight:800; margin-bottom:4px;'>
        🛡️ Fraud Transaction Detection
        <span style='font-size:14px; color:#64748b; font-weight:400;'> — Real-Time ML Dashboard</span>
    </h1>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.markdown("""
        <div class='info-card' style='text-align:center; padding:60px 40px;'>
            <div style='font-size:64px; margin-bottom:16px;'>📂</div>
            <h3 style='color:#3b82f6;'>No Dataset Loaded</h3>
            <p style='color:#94a3b8;'>Upload a CSV file or click "Use Sample Dataset" in the sidebar to get started.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    df = st.session_state.df
    summary = get_data_summary(df)

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Transactions", f"{summary['total_transactions']:,}")
    col2.metric("Fraud Cases", f"{summary['fraud_count']:,}", delta=f"{summary['fraud_rate']}% rate", delta_color="inverse")
    col3.metric("Legitimate", f"{summary['legit_count']:,}")
    col4.metric("Features", f"{len(summary['features'])}")
    col5.metric("Missing Values", f"{summary['missing_values']}")

    st.markdown("---")

    # Row 1: Pie + Amount Dist
    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(fraud_pie_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(amount_distribution(df), use_container_width=True)

    # Row 2: Hourly + Category
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fraud_by_hour(df), use_container_width=True)
    with col2:
        cat_cols = [c for c in ["merchant_category", "device_type", "transaction_type", "location"] if c in df.columns]
        selected_cat = st.selectbox("Category column", cat_cols, label_visibility="collapsed")
        st.plotly_chart(fraud_by_category(df, selected_cat), use_container_width=True)

    # Row 3: Correlation heatmap
    st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    st.plotly_chart(correlation_heatmap(df), use_container_width=True)

    # Model comparison (if trained)
    if st.session_state.trained and st.session_state.results:
        st.markdown('<div class="section-header">Model Performance Overview</div>', unsafe_allow_html=True)
        st.plotly_chart(model_comparison_chart(st.session_state.results), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown("<h2 class='section-header'>📊 Data Explorer</h2>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please load a dataset first.")
        st.stop()

    df = st.session_state.df

    tab1, tab2, tab3 = st.tabs(["📋 Dataset Preview", "📈 Statistics", "🔍 Deep Dive"])

    with tab1:
        col1, col2 = st.columns([3, 1])
        with col2:
            n_rows = st.slider("Rows to show", 5, 100, 20)
        with col1:
            filter_fraud = st.selectbox("Filter by", ["All", "Fraud Only", "Legitimate Only"])

        display_df = df.copy()
        if filter_fraud == "Fraud Only" and "is_fraud" in df.columns:
            display_df = df[df["is_fraud"] == 1]
        elif filter_fraud == "Legitimate Only" and "is_fraud" in df.columns:
            display_df = df[df["is_fraud"] == 0]

        st.dataframe(display_df.head(n_rows), use_container_width=True, height=400)
        st.caption(f"Showing {min(n_rows, len(display_df))} of {len(display_df):,} rows")

    with tab2:
        st.markdown("**Numeric Feature Statistics**")
        numeric_df = df.select_dtypes(include=[np.number])
        stats = numeric_df.describe().T
        stats["missing"] = df.isnull().sum()
        stats["fraud_mean"] = df[df["is_fraud"]==1][numeric_df.columns].mean() if "is_fraud" in df.columns else None
        st.dataframe(stats.style.background_gradient(cmap="Blues", subset=["mean", "std"]), use_container_width=True)

        st.markdown("**Categorical Feature Value Counts**")
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if cat_cols:
            sel = st.selectbox("Select categorical column", cat_cols)
            vc = df[sel].value_counts().reset_index()
            vc.columns = ["Value", "Count"]
            vc["Fraud Rate (%)"] = [
                round(df[df[sel]==v]["is_fraud"].mean()*100, 2) if "is_fraud" in df.columns else 0
                for v in vc["Value"]
            ]
            st.dataframe(vc, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fraud_by_category(df, "device_type" if "device_type" in df.columns else df.columns[0]), use_container_width=True)
        with col2:
            st.plotly_chart(fraud_by_category(df, "transaction_type" if "transaction_type" in df.columns else df.columns[0]), use_container_width=True)

        if "transaction_amount" in df.columns and "is_fraud" in df.columns:
            import plotly.express as px
            fig = px.box(df, x="is_fraud", y="transaction_amount", color="is_fraud",
                        color_discrete_map={0: "#22c55e", 1: "#ef4444"},
                        labels={"is_fraud": "Is Fraud", "transaction_amount": "Amount ($)"},
                        title="Transaction Amount by Class (Box Plot)")
            fig.update_layout(paper_bgcolor="#0a0f1e", plot_bgcolor="#111827",
                             font=dict(color="#e2e8f0"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.markdown("<h2 class='section-header'>🤖 Model Training & Evaluation</h2>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please load a dataset first.")
        st.stop()

    df = st.session_state.df

    # Config
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05, help="Fraction of data held out for testing")
    with col2:
        random_state = st.number_input("Random seed", value=42, step=1)
    with col3:
        st.metric("Training Samples (est.)", f"{int(len(df) * (1-test_size)):,}")

    st.markdown("""
    <div class='info-card'>
        <b>⚙️ Pipeline Preview:</b> Missing value imputation → Label encoding → Train/test split →
        StandardScaler → <span style='color:#3b82f6;'>SMOTE balancing</span> → Train 3 models → Evaluate → Save .pkl
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Train All Models", use_container_width=True):
        if "is_fraud" not in df.columns:
            st.error("Dataset must contain an 'is_fraud' column.")
        else:
            with st.spinner("Training in progress... (this may take 30–60 seconds)"):
                try:
                    progress = st.progress(0, text="Preprocessing data...")
                    time.sleep(0.5)

                    X_train, X_test, y_train, y_test, scaler, encoders, feature_cols = preprocess_pipeline(
                        df, test_size=test_size, random_state=int(random_state)
                    )
                    progress.progress(30, text="Data ready. Training Logistic Regression...")
                    time.sleep(0.3)

                    results = train_all(X_train, X_test, y_train, y_test, feature_cols)
                    progress.progress(90, text="Saving models...")
                    time.sleep(0.3)

                    st.session_state.update({
                        "trained": True, "results": results,
                        "scaler": scaler, "encoders": encoders, "feature_cols": feature_cols,
                        "X_train": X_train, "X_test": X_test,
                        "y_train": y_train, "y_test": y_test
                    })
                    progress.progress(100, text="Done!")
                    st.success("✅ All models trained and saved!")
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    logger.exception(e)

    if not st.session_state.trained:
        st.stop()

    results = st.session_state.results

    # Metrics table
    st.markdown('<div class="section-header">📊 Performance Metrics</div>', unsafe_allow_html=True)
    model_names = [k for k in results if k != "feature_importances"]
    metrics_data = {
        "Model": model_names,
        "Accuracy": [results[m].get("accuracy", 0) for m in model_names],
        "Precision": [results[m].get("precision", 0) for m in model_names],
        "Recall": [results[m].get("recall", 0) for m in model_names],
        "F1-Score": [results[m].get("f1_score", 0) for m in model_names],
        "ROC-AUC": [results[m].get("roc_auc", 0) for m in model_names],
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(
        metrics_df.style.highlight_max(subset=["Accuracy","Precision","Recall","F1-Score","ROC-AUC"],
                                       color="#1e3a5f").format({c: "{:.4f}" for c in metrics_df.columns[1:]}),
        use_container_width=True
    )

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(model_comparison_chart(results), use_container_width=True)
    with col2:
        st.plotly_chart(roc_curve_plot(results), use_container_width=True)

    # Confusion matrices
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    cm_cols = st.columns(3)
    for i, model_name in enumerate(model_names):
        with cm_cols[i]:
            cm_data = results[model_name].get("confusion_matrix", [[0,0],[0,0]])
            st.plotly_chart(confusion_matrix_heatmap(cm_data, model_name), use_container_width=True)

    # Feature importances
    if "feature_importances" in results:
        st.markdown('<div class="section-header">Feature Importances (Random Forest)</div>', unsafe_allow_html=True)
        st.plotly_chart(feature_importance_chart(results["feature_importances"]), use_container_width=True)
        st.info("💡 Features with higher importance scores have more influence on fraud classification.")

    # Per-model detailed report
    st.markdown('<div class="section-header">Detailed Classification Reports</div>', unsafe_allow_html=True)
    sel_model = st.selectbox("Select model", model_names)
    if sel_model in results:
        report = results[sel_model].get("classification_report", {})
        rows = []
        for label, vals in report.items():
            if isinstance(vals, dict):
                rows.append({
                    "Class": "Legitimate" if label == "0" else "Fraud" if label == "1" else label,
                    "Precision": round(vals.get("precision", 0), 4),
                    "Recall": round(vals.get("recall", 0), 4),
                    "F1-Score": round(vals.get("f1-score", 0), 4),
                    "Support": int(vals.get("support", 0))
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: REAL-TIME PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Real-Time Prediction":
    st.markdown("<h2 class='section-header'>🔍 Real-Time Fraud Prediction</h2>", unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("⚠️ Please train models first (go to Model Training page).")
        st.stop()

    models = load_all_models()
    if not models:
        st.error("No saved models found. Please train first.")
        st.stop()

    # Config
    col1, col2 = st.columns([2, 1])
    with col2:
        selected_model = st.selectbox("Select Model", list(models.keys()))
        threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05,
                             help="Lower = catch more fraud (higher recall, more false alarms). Higher = fewer alerts (fewer false alarms, may miss fraud).")
        st.markdown(f"""
        <div class='info-card' style='font-size:12px;'>
            <b>Threshold = {threshold}</b><br>
            Probability ≥ {threshold} → Flagged as fraud<br>
            <span style='color:#f59e0b;'>↓ threshold = ↑ recall, ↓ precision</span>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        st.markdown("**Enter Transaction Details:**")

        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Transaction Amount ($)", 0.0, 100000.0, 150.0, step=10.0)
            tx_time = st.slider("Transaction Hour (0–23)", 0, 23, 14)
            acct_age = st.number_input("Account Age (days)", 0, 5000, 365)
            num_today = st.number_input("# Transactions Today", 1, 50, 2)
        with c2:
            distance = st.number_input("Distance from Home (km)", 0.0, 10000.0, 5.0)
            is_foreign = st.selectbox("Foreign Transaction?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            merchant_cat = st.selectbox("Merchant Category",
                ["grocery", "electronics", "restaurant", "travel", "clothing", "pharmacy", "entertainment", "fuel"])
            tx_type = st.selectbox("Transaction Type", ["online", "in_store", "ATM", "contactless"])
        location = st.selectbox("Location",
            ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego"])
        device_type = st.selectbox("Device Type", ["mobile", "desktop", "tablet", "POS_terminal"])

    transaction = {
        "transaction_amount": amount,
        "transaction_time": tx_time,
        "merchant_category": merchant_cat,
        "location": location,
        "device_type": device_type,
        "transaction_type": tx_type,
        "account_age_days": acct_age,
        "num_transactions_today": num_today,
        "distance_from_home_km": distance,
        "is_foreign_transaction": is_foreign
    }

    if st.button("🔎 Analyze Transaction", use_container_width=True):
        with st.spinner("Analyzing..."):
            time.sleep(0.4)
            model = models[selected_model]
            result = predict_single(
                transaction, model,
                st.session_state.scaler,
                st.session_state.encoders,
                st.session_state.feature_cols,
                threshold
            )

        prob = result["fraud_probability"]
        is_fraud = result["is_fraud"]

        # Alert banner
        if is_fraud:
            st.markdown(f"""
            <div class='alert-fraud'>
                <span style='font-size:20px;'>⚠️ FRAUD ALERT</span><br>
                <span style='font-size:14px; color:#fca5a5;'>High-risk transaction flagged by {selected_model}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='legit-card'>
                <span style='font-size:20px;'>✅ TRANSACTION APPROVED</span><br>
                <span style='font-size:14px; color:#86efac;'>Classified as legitimate by {selected_model}</span>
            </div>
            """, unsafe_allow_html=True)

        # Gauge + details
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(probability_gauge(prob, selected_model), use_container_width=True)
        with col2:
            st.markdown("**Prediction Details:**")
            st.markdown(f"""
            <div class='info-card'>
                <table style='width:100%; font-size:14px;'>
                    <tr><td style='color:#94a3b8;'>Model</td><td><b>{selected_model}</b></td></tr>
                    <tr><td style='color:#94a3b8;'>Fraud Probability</td><td><b style='color:{"#ef4444" if is_fraud else "#22c55e"};'>{prob*100:.2f}%</b></td></tr>
                    <tr><td style='color:#94a3b8;'>Risk Level</td><td>{result['risk_level']}</td></tr>
                    <tr><td style='color:#94a3b8;'>Verdict</td><td><b>{result['verdict']}</b></td></tr>
                    <tr><td style='color:#94a3b8;'>Threshold Used</td><td>{threshold}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        # All models comparison
        st.markdown("**All Models Prediction:**")
        model_cols = st.columns(len(models))
        for i, (mname, mobj) in enumerate(models.items()):
            r = predict_single(transaction, mobj, st.session_state.scaler,
                              st.session_state.encoders, st.session_state.feature_cols, threshold)
            with model_cols[i]:
                color = "#ef4444" if r["is_fraud"] else "#22c55e"
                st.markdown(f"""
                <div class='info-card' style='text-align:center;'>
                    <div style='font-size:12px; color:#94a3b8;'>{mname}</div>
                    <div style='font-size:24px; font-weight:700; color:{color};'>{r["fraud_probability"]*100:.1f}%</div>
                    <div style='font-size:12px;'>{r["risk_level"]}</div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Batch Prediction":
    st.markdown("<h2 class='section-header'>📦 Batch Prediction</h2>", unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("⚠️ Please train models first.")
        st.stop()

    models = load_all_models()
    if not models:
        st.error("No saved models found. Please train first.")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
    with col2:
        batch_model = st.selectbox("Model for batch prediction", list(models.keys()))
        batch_threshold = st.slider("Threshold", 0.1, 0.9, 0.5, 0.05, key="batch_thresh")

    if st.session_state.df is not None:
        if st.button("🔁 Run on Current Dataset (sample 500)", use_container_width=True):
            sample = st.session_state.df.sample(min(500, len(st.session_state.df))).copy()
            sample.drop(columns=["is_fraud"], errors="ignore", inplace=True)
            with st.spinner("Predicting..."):
                result_df = predict_batch(
                    sample, models[batch_model],
                    st.session_state.scaler, st.session_state.encoders,
                    st.session_state.feature_cols, batch_threshold
                )
            st.session_state["batch_results"] = result_df

    if batch_file is not None:
        with st.spinner("Processing..."):
            upload_df = pd.read_csv(batch_file)
            upload_df.drop(columns=["is_fraud"], errors="ignore", inplace=True)
            result_df = predict_batch(
                upload_df, models[batch_model],
                st.session_state.scaler, st.session_state.encoders,
                st.session_state.feature_cols, batch_threshold
            )
        st.session_state["batch_results"] = result_df
        st.success(f"✅ Predicted {len(result_df)} transactions")

    if "batch_results" in st.session_state and st.session_state["batch_results"] is not None:
        result_df = st.session_state["batch_results"]

        # KPIs
        fraud_preds = result_df["predicted_fraud"].sum() if "predicted_fraud" in result_df.columns else 0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Analyzed", f"{len(result_df):,}")
        col2.metric("Predicted Fraud", f"{fraud_preds:,}", delta=f"{fraud_preds/len(result_df)*100:.1f}%", delta_color="inverse")
        col3.metric("Predicted Legit", f"{len(result_df)-fraud_preds:,}")
        col4.metric("High Risk", f"{(result_df.get('risk_level','')=='🔴 High Risk').sum():,}" if 'risk_level' in result_df.columns else "N/A")

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(scatter_amount_vs_risk(result_df), use_container_width=True)
        with col2:
            if "risk_level" in result_df.columns:
                import plotly.express as px
                risk_counts = result_df["risk_level"].value_counts().reset_index()
                risk_counts.columns = ["Risk Level", "Count"]
                color_map = {"🔴 High Risk": "#ef4444", "🟡 Medium Risk": "#f59e0b", "🟢 Low Risk": "#22c55e"}
                fig = px.bar(risk_counts, x="Risk Level", y="Count",
                            color="Risk Level", color_discrete_map=color_map,
                            title="Risk Level Distribution")
                fig.update_layout(paper_bgcolor="#0a0f1e", plot_bgcolor="#111827",
                                 font=dict(color="#e2e8f0"), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Results table
        st.markdown("**Prediction Results (Top 50):**")
        display_cols = [c for c in result_df.columns if c in [
            "transaction_amount", "merchant_category", "transaction_type",
            "fraud_probability", "predicted_fraud", "risk_level", "verdict"
        ]]
        st.dataframe(result_df[display_cols].head(50), use_container_width=True)

        # Download button
        csv_bytes = result_df.to_csv(index=False).encode()
        st.download_button(
            "📥 Download Full Predictions CSV",
            data=csv_bytes,
            file_name="fraud_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6: INTERVIEW PREP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📚 Interview Prep":
    st.markdown("<h2 class='section-header'>📚 Interview Preparation Guide</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-card'>
    This section covers the ML concepts behind the project — perfect for explaining it in interviews.
    Understanding <em>why</em> you made each choice matters more than memorizing code.
    </div>
    """, unsafe_allow_html=True)

    qa_list = [
        ("Q1: Explain this project in 2 minutes.",
         """This is an end-to-end Machine Learning system that detects fraudulent banking transactions.
Given a dataset of transactions with features like amount, time, location, and merchant category,
we train three models — Logistic Regression (baseline), Random Forest (best performer), and
Isolation Forest (anomaly detector). The system handles real-world challenges like severe class
imbalance (only 5% fraud), provides probability scores instead of just yes/no labels, and lets
analysts tune the decision threshold based on their risk tolerance."""),

        ("Q2: Why is fraud detection difficult?",
         """Three core challenges:
(1) Extreme class imbalance — 99% legitimate, 1% fraud in production. Naive models predict everything
as legit and hit 99% accuracy while being useless.
(2) Adversarial adaptation — fraudsters actively change behavior to evade detection systems.
(3) Precision-Recall tradeoff — catching all fraud causes too many false alarms (blocking real customers),
while minimizing false alarms means missing fraud. Banks must pick a balance based on business impact."""),

        ("Q3: Why does class imbalance matter? How do you fix it?",
         """A model trained on 95% legit / 5% fraud data can score 95% accuracy by predicting everything
as 'legit' — but its recall for fraud is ZERO. We use SMOTE (Synthetic Minority Oversampling Technique),
which creates synthetic fraud samples by interpolating between existing fraud examples in feature space.
Unlike simple oversampling (copy-paste), SMOTE forces the model to learn the fraud distribution, not
just memorize it. Important: SMOTE is applied ONLY to training data. Test set retains original imbalance
to simulate real-world performance."""),

        ("Q4: Why Precision and Recall over Accuracy?",
         """Accuracy = (TP+TN)/(All). On imbalanced data this is misleading.
Precision = TP/(TP+FP) — 'Of all we flagged as fraud, what fraction was actually fraud?'
Low precision = too many false positives = legitimate customers get blocked = complaints.
Recall = TP/(TP+FN) — 'Of all actual frauds, what fraction did we catch?'
Low recall = missed frauds = direct financial losses.
F1-Score = harmonic mean of Precision & Recall — single metric when both matter.
Banks typically prefer HIGH RECALL (catch all fraud) even at cost of precision."""),

        ("Q5: Difference between Classification and Anomaly Detection?",
         """Classification (Logistic Regression, Random Forest): Supervised. Learns from labeled fraud/legit
examples. Needs labeled training data. Performs well when fraud patterns are consistent.

Anomaly Detection (Isolation Forest): Unsupervised. Learns what 'normal' looks like, flags deviations.
No labels needed. Great for detecting NEW types of fraud never seen before.
In practice, banks use both — classification for known fraud patterns, anomaly detection as a second
layer for novel attack vectors."""),

        ("Q6: Why use Random Forest for fraud detection?",
         """Random Forest excels here because:
(1) Handles non-linear relationships — fraud has complex interaction patterns.
(2) Resistant to overfitting via bagging and feature randomization.
(3) Naturally handles mixed feature types (numeric + categorical).
(4) Provides feature importances — critical for regulatory explainability (banks must explain decisions).
(5) Robust to outliers — fraudulent transactions are extreme by nature.
Typically achieves 95%+ ROC-AUC on structured fraud datasets."""),

        ("Q7: What are real-world banking applications of this system?",
         """(1) Credit card fraud detection (Visa, Mastercard) — real-time scoring on every swipe.
(2) Account takeover prevention — unusual login + transaction patterns.
(3) AML (Anti-Money Laundering) — pattern detection across transaction networks.
(4) Insurance fraud — claims with anomalous patterns.
(5) E-commerce chargeback prevention — flag high-risk orders before fulfillment.
At production scale (Visa processes 65,000 TPS), models must score in < 100ms."""),

        ("Q8: What would you improve in production?",
         """(1) XGBoost/LightGBM — gradient boosting typically outperforms Random Forest on tabular data.
(2) Feature engineering — velocity features (5 transactions in 10 minutes), graph features (transaction network).
(3) Time-series awareness — use transaction sequences with LSTMs or transformers.
(4) Online learning — retrain incrementally as fraud patterns evolve (using river or scikit-multiflow).
(5) Model monitoring — detect data drift and model degradation with tools like Evidently AI.
(6) Explainability — use SHAP values for individual prediction explanations (regulatory requirement).
(7) A/B testing framework — safely deploy new models with canary rollouts.""")
    ]

    for q, a in qa_list:
        with st.expander(q):
            st.markdown(f"""<div class='interview-q'><p>{a}</p></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">🏗️ Project Architecture</div>', unsafe_allow_html=True)
    st.code("""
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend (app.py)               │
│  Dashboard │ Explorer │ Training │ Prediction │ Batch │ Prep │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
  preprocessing.py   train.py     predict.py
  • Missing values  • Logistic R  • Single txn
  • Label encoding  • Rand Forest • Batch CSV
  • StandardScaler  • Iso Forest  • Risk tags
  • SMOTE balance   • Evaluation  • Thresholds
         │             │              │
         └─────────────┴──────────────┘
                       │
                  utils.py
              (All Plotly charts)
                       │
                 models/ (pkl)
    logistic_model.pkl | random_forest.pkl | isolation_forest.pkl
    """, language="text")

    st.markdown('<div class="section-header">📦 Tech Stack</div>', unsafe_allow_html=True)
    tech_cols = st.columns(4)
    techs = [
        ("Frontend", "Streamlit", "Real-time UI with charts and forms"),
        ("ML Core", "scikit-learn", "Logistic Regression, Random Forest, Isolation Forest"),
        ("Balancing", "imbalanced-learn", "SMOTE for class imbalance correction"),
        ("Charts", "Plotly", "Interactive fraud analytics dashboard")
    ]
    for i, (layer, tech, desc) in enumerate(techs):
        with tech_cols[i]:
            st.markdown(f"""
            <div class='info-card' style='text-align:center;'>
                <div style='color:#94a3b8; font-size:11px;'>{layer}</div>
                <div style='font-size:16px; font-weight:700; color:#3b82f6;'>{tech}</div>
                <div style='font-size:11px; color:#64748b; margin-top:4px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
