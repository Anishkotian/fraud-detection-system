"""
utils.py — Visualization and Helper Utilities
==============================================
All chart-generation functions used by the Streamlit dashboard.
Keeping visualizations here keeps app.py clean.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import io
import base64

# ── Color Palette ──────────────────────────────────────────────────────────────
COLORS = {
    "fraud": "#ef4444",
    "legit": "#22c55e",
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "accent": "#f59e0b",
    "bg": "#0f172a",
    "card": "#1e293b",
    "text": "#e2e8f0",
    "grid": "#334155"
}

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
        yaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    )
)


def fraud_pie_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart: Fraud vs Legitimate transaction ratio."""
    counts = df["is_fraud"].value_counts()
    labels = ["Legitimate", "Fraud"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=[COLORS["legit"], COLORS["fraud"]],
                    line=dict(color=COLORS["bg"], width=3)),
        textfont=dict(size=14, color="white"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Pct: %{percent}<extra></extra>"
    ))
    fig.add_annotation(text=f"<b>{values[1]}</b><br>Frauds",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=18, color=COLORS["fraud"]))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title=dict(text="Transaction Class Distribution", font=dict(size=16)),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(t=50, b=20, l=20, r=20),
        height=320
    )
    return fig


def amount_distribution(df: pd.DataFrame) -> go.Figure:
    """Overlapping histogram: transaction amount for fraud vs legit."""
    legit = df[df["is_fraud"] == 0]["transaction_amount"]
    fraud = df[df["is_fraud"] == 1]["transaction_amount"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=legit, name="Legitimate", opacity=0.7,
        marker_color=COLORS["legit"], nbinsx=50,
        hovertemplate="Amount: %{x}<br>Count: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Histogram(
        x=fraud, name="Fraud", opacity=0.7,
        marker_color=COLORS["fraud"], nbinsx=50,
        hovertemplate="Amount: %{x}<br>Count: %{y}<extra></extra>"
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        barmode="overlay",
        title="Transaction Amount Distribution by Class",
        xaxis_title="Amount ($)",
        yaxis_title="Count",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=60, b=40, l=40, r=20),
        height=350
    )
    return fig


def fraud_by_category(df: pd.DataFrame, col: str = "merchant_category") -> go.Figure:
    """Bar chart: fraud rate per category column."""
    if col not in df.columns:
        return go.Figure()
    grouped = df.groupby(col)["is_fraud"].agg(["sum", "count"]).reset_index()
    grouped["fraud_rate"] = (grouped["sum"] / grouped["count"] * 100).round(2)
    grouped = grouped.sort_values("fraud_rate", ascending=True)

    fig = go.Figure(go.Bar(
        x=grouped["fraud_rate"],
        y=grouped[col],
        orientation="h",
        marker=dict(
            color=grouped["fraud_rate"],
            colorscale=[[0, COLORS["legit"]], [0.5, COLORS["accent"]], [1, COLORS["fraud"]]],
            showscale=True,
            colorbar=dict(title="Fraud %")
        ),
        text=[f"{v:.1f}%" for v in grouped["fraud_rate"]],
        textposition="outside",
        hovertemplate=f"<b>%{{y}}</b><br>Fraud Rate: %{{x:.1f}}%<extra></extra>"
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title=f"Fraud Rate by {col.replace('_', ' ').title()}",
        xaxis_title="Fraud Rate (%)",
        margin=dict(t=50, b=40, l=120, r=80),
        height=max(300, len(grouped) * 40)
    )
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of numeric feature correlations."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="<b>%{x} × %{y}</b><br>Correlation: %{z:.2f}<extra></extra>"
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title="Feature Correlation Matrix",
        margin=dict(t=50, b=60, l=80, r=20),
        height=450
    )
    return fig


def fraud_by_hour(df: pd.DataFrame) -> go.Figure:
    """Line/area chart: fraud transactions by hour of day."""
    if "transaction_time" not in df.columns:
        return go.Figure()
    hourly = df.groupby(["transaction_time", "is_fraud"]).size().reset_index(name="count")
    fraud_hourly = hourly[hourly["is_fraud"] == 1]
    legit_hourly = hourly[hourly["is_fraud"] == 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=legit_hourly["transaction_time"], y=legit_hourly["count"],
        name="Legitimate", fill="tozeroy", mode="lines",
        line=dict(color=COLORS["legit"], width=2),
        fillcolor="rgba(34,197,94,0.15)"
    ))
    fig.add_trace(go.Scatter(
        x=fraud_hourly["transaction_time"], y=fraud_hourly["count"],
        name="Fraud", fill="tozeroy", mode="lines",
        line=dict(color=COLORS["fraud"], width=2),
        fillcolor="rgba(239,68,68,0.3)"
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title="Transaction Frequency by Hour of Day",
        xaxis=dict(title="Hour", tickmode="linear", dtick=2, gridcolor=COLORS["grid"]),
        yaxis=dict(title="Count", gridcolor=COLORS["grid"]),
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=60, b=40, l=50, r=20),
        height=320
    )
    return fig


def roc_curve_plot(results: dict) -> go.Figure:
    """Multi-model ROC AUC curves on one chart."""
    model_colors = {
        "Logistic Regression": "#6366f1",
        "Random Forest": "#22c55e",
        "Isolation Forest": "#f59e0b"
    }
    fig = go.Figure()
    # Diagonal baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Random Baseline", showlegend=True
    ))
    for model_name, metrics in results.items():
        if model_name == "feature_importances":
            continue
        if "fpr" in metrics and "tpr" in metrics:
            auc = metrics.get("roc_auc", 0)
            fig.add_trace(go.Scatter(
                x=metrics["fpr"], y=metrics["tpr"],
                mode="lines", name=f"{model_name} (AUC={auc:.3f})",
                line=dict(color=model_colors.get(model_name, "#fff"), width=2.5)
            ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title="ROC Curves — All Models",
        xaxis=dict(title="False Positive Rate", gridcolor=COLORS["grid"]),
        yaxis=dict(title="True Positive Rate", gridcolor=COLORS["grid"]),
        legend=dict(y=0.05, x=0.6, bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(t=60, b=50, l=50, r=20),
        height=420
    )
    return fig


def confusion_matrix_heatmap(cm_data: list, model_name: str) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    cm = np.array(cm_data)
    labels = ["Legitimate", "Fraud"]
    text = [[f"TN={cm[0,0]}", f"FP={cm[0,1]}"],
            [f"FN={cm[1,0]}", f"TP={cm[1,1]}"]]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted Legit", "Predicted Fraud"],
        y=["Actual Legit", "Actual Fraud"],
        colorscale=[[0, "#1e293b"], [1, COLORS["primary"]]],
        text=text,
        texttemplate="<b>%{text}</b><br>%{z}",
        hovertemplate="%{text}<br>Count: %{z}<extra></extra>",
        showscale=False
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title=f"Confusion Matrix — {model_name}",
        xaxis=dict(side="bottom"),
        margin=dict(t=60, b=50, l=80, r=20),
        height=320
    )
    return fig


def feature_importance_chart(importances: dict) -> go.Figure:
    """Horizontal bar chart of Random Forest feature importances."""
    features = importances["features"][:10]  # top 10
    values = importances["importances"][:10]

    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=features[::-1],
        orientation="h",
        marker=dict(
            color=values[::-1],
            colorscale=[[0, COLORS["secondary"]], [1, COLORS["primary"]]],
        ),
        text=[f"{v:.3f}" for v in values[::-1]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title="Top Feature Importances (Random Forest)",
        xaxis_title="Importance Score",
        margin=dict(t=60, b=40, l=160, r=80),
        height=380
    )
    return fig


def model_comparison_chart(results: dict) -> go.Figure:
    """Grouped bar chart comparing all model metrics."""
    model_names = [k for k in results if k != "feature_importances"]
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    colors = [COLORS["primary"], COLORS["legit"], COLORS["accent"], COLORS["secondary"], COLORS["fraud"]]

    fig = go.Figure()
    for metric, label, color in zip(metrics, metric_labels, colors):
        values = [results[m].get(metric, 0) for m in model_names]
        fig.add_trace(go.Bar(
            name=label,
            x=model_names,
            y=values,
            marker_color=color,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
        ))
  fig.update_layout(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["card"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    barmode="group",
    title="Model Performance Comparison",
    yaxis=dict(title="Score", range=[0, 1.15], gridcolor=COLORS["grid"]),
    xaxis=dict(gridcolor=COLORS["grid"]),
    legend=dict(orientation="h", y=1.15),
    margin=dict(t=80, b=40, l=50, r=20),
    height=420
)
    return fig


def probability_gauge(probability: float, model_name: str) -> go.Figure:
    """Gauge chart for fraud probability score."""
    color = COLORS["fraud"] if probability > 0.6 else (COLORS["accent"] if probability > 0.3 else COLORS["legit"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number=dict(suffix="%", font=dict(size=28, color=color)),
        title=dict(text=f"Fraud Probability<br><span style='font-size:12px'>{model_name}</span>",
                   font=dict(size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="white"),
            bar=dict(color=color, thickness=0.3),
            bgcolor=COLORS["card"],
            borderwidth=2,
            bordercolor=COLORS["grid"],
            steps=[
                dict(range=[0, 30], color="rgba(34,197,94,0.15)"),
                dict(range=[30, 60], color="rgba(245,158,11,0.15)"),
                dict(range=[60, 100], color="rgba(239,68,68,0.15)")
            ],
            threshold=dict(line=dict(color="white", width=3), thickness=0.75, value=50)
        )
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"]),
        margin=dict(t=60, b=20, l=30, r=30),
        height=260
    )
    return fig


def scatter_amount_vs_risk(df_results: pd.DataFrame) -> go.Figure:
    """Scatter: transaction amount vs fraud probability (batch predictions)."""
    if "fraud_probability" not in df_results.columns:
        return go.Figure()
    color_map = {
        "🔴 High Risk": COLORS["fraud"],
        "🟡 Medium Risk": COLORS["accent"],
        "🟢 Low Risk": COLORS["legit"]
    }
    fig = go.Figure()
    for risk, color in color_map.items():
        subset = df_results[df_results["risk_level"] == risk]
        if len(subset) > 0:
            fig.add_trace(go.Scatter(
                x=subset["transaction_amount"],
                y=subset["fraud_probability"],
                mode="markers",
                name=risk,
                marker=dict(color=color, size=6, opacity=0.7),
                hovertemplate="Amount: $%{x}<br>Fraud Prob: %{y:.3f}<extra></extra>"
            ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title="Transaction Amount vs Fraud Probability",
        xaxis_title="Transaction Amount ($)",
        yaxis_title="Fraud Probability",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=60, b=40, l=50, r=20),
        height=380
    )
    return fig
