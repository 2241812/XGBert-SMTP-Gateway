"""
smtpBERT Streamlit Dashboard
Visual upgrades: ECharts heatmaps, Plotly bar/box plots, AgGrid tables,
JetBrains Mono typography, glassmorphism UI.
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from st_aggrid import AgGrid  # noqa: F401 - kept for future use

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gateway.detector import PhishingDetector
from src.models.cross_validation import CV_RESULTS_PATH, run_kfold_cross_validation
from src.models import config as model_config

DISTILBERT_METRICS_PATH = os.path.join(model_config.MODEL_OUTPUT_DIR_DISTILBERT, "training_metrics.json")
HOLDOUT_RESULTS_PATH = os.path.join(model_config.LOG_DIR, "model_comparison", "model_comparison.json")


# -------------------------------------------------------------------------
# Theme & CSS
# -------------------------------------------------------------------------
st.set_page_config(page_title="smtpBERT Dashboard", page_icon="🛡️", layout="wide")

st.html("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

  :root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.75);
    --border-color: rgba(6, 182, 212, 0.18);
    --text-primary: #e2e8f0;
    --text-muted: #94a3b8;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-yellow: #f59e0b;
    --accent-red: #ef4444;
    --accent-purple: #a78bfa;
  }

  * { font-family: 'JetBrains Mono', monospace !important; }

  .stApp { background-color: var(--bg-primary) !important; color: var(--text-primary); }

  [data-testid="stHeader"] { background-color: var(--bg-primary) !important; }

  .stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--bg-secondary); border-radius: 8px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { border-radius: 6px; color: var(--text-muted); font-size: 0.82rem; font-weight: 500; padding: 8px 16px; }
  .stTabs [aria-selected="true"] { background: rgba(6, 182, 212, 0.15) !important; color: var(--accent-cyan) !important; }

  section[data-testid="stMainBlockContainer"] { padding-top: 1rem; }

  .glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
  }

  h1, h2, h3, h4 { color: var(--text-primary) !important; font-weight: 600; }
  h1 { font-size: 1.6rem !important; }
  h2 { font-size: 1.2rem !important; color: var(--accent-cyan) !important; }
  h3 { font-size: 1rem !important; color: var(--text-primary) !important; }

  .metric-label { color: var(--text-muted) !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-value { color: var(--accent-green) !important; font-size: 1.8rem; font-weight: 700; }

  .stMetric label { color: var(--text-muted) !important; }
  .stMetric [data-testid="stMetricValue"] { color: var(--accent-green) !important; font-family: 'JetBrains Mono', monospace !important; }

  .stButton > button { border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }
  .stButton > button[data-baseweb="button"] { border: 1px solid var(--border-color); }

  .stDataFrame { border-radius: 8px; overflow: hidden; }
  .dataframe { border: none !important; }

  .st-expander { border: 1px solid var(--border-color) !important; border-radius: 8px !important; background: var(--bg-card) !important; }
  .st-expander > details > summary { color: var(--text-primary) !important; font-weight: 500; }

  .stSpinner > div { border-color: var(--accent-cyan) !important; }

  .stSuccess, .stWarning, .stError, .stInfo { border-radius: 8px; }

  div[data-testid="stHorizontalBlock"] { gap: 1.5rem; }

  .tab-content { padding: 1rem 0; }

  .ag-theme-streamlit { --ag-background-color: var(--bg-secondary); --ag-header-background-color: rgba(6,182,212,0.08); --ag-foreground-color: var(--text-primary); --ag-border-color: var(--border-color); --ag-row-hover-color: rgba(6,182,212,0.06); font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
</style>
""")


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
@st.cache_resource
def get_detector() -> PhishingDetector:
    return PhishingDetector()


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _specificity_from_confusion(confusion: list) -> float:
    if not confusion:
        return 0.0
    matrix = [[float(v) for v in row] for row in confusion]
    total = sum(sum(row) for row in matrix)
    specificities = []
    for i in range(4):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(4)) - tp
        fn = sum(matrix[i]) - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        specificities.append(tn / denom if denom > 0 else 0.0)
    return sum(specificities) / 4


# -------------------------------------------------------------------------
# Plotly theme helpers
# -------------------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_white"
ACCENT_COLORS_LIGHT = ["#0891b2", "#059669", "#d97706", "#7c3aed"]
BG_LIGHT = "#ffffff"
TEXT_DARK = "#1e293b"
TEXT_MUTED = "#64748b"
GRID_LIGHT = "rgba(0,0,0,0.06)"


def _make_bar_chart(metrics_df: pd.DataFrame, metric_col: str, title: str) -> go.Figure:
    fig = px.bar(
        metrics_df,
        x="Model",
        y=metric_col,
        color="Model",
        color_discrete_sequence=ACCENT_COLORS_LIGHT,
        text_auto=".4f",
        title=title,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        font=dict(family="JetBrains Mono", size=12),
        paper_bgcolor=BG_LIGHT,
        plot_bgcolor=BG_LIGHT,
        showlegend=False,
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_range=[0, 1.05],
        xaxis=dict(tickfont=dict(color=TEXT_DARK), title=""),
        yaxis=dict(tickfont=dict(color=TEXT_DARK), gridcolor=GRID_LIGHT, title=""),
        title=dict(font=dict(color=TEXT_DARK, size=14, family="JetBrains Mono"), x=0.5),
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color=TEXT_DARK),
        marker_line_width=1.5,
        marker_line_color=TEXT_DARK,
    )
    return fig


def _make_box_plot(cv_results: Dict, metric: str, title: str) -> go.Figure:
    model_order = ["DistilBERT", "XGBoost", "LogisticRegression", "RandomForest"]
    rows = []
    for model_name in model_order:
        if model_name not in cv_results.get("summary", {}):
            continue
        model_data = cv_results["summary"][model_name]
        values_key = f"{metric}_values"
        for v in model_data.get(values_key, []):
            rows.append({"Model": model_name, "Value": v})

    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()

    order = [m for m in model_order if m in df["Model"].unique()]
    fig = px.box(
        df,
        x="Model",
        y="Value",
        color="Model",
        color_discrete_sequence=ACCENT_COLORS_LIGHT,
        points="all",
        title=title,
        template=PLOTLY_TEMPLATE,
        category_orders={"Model": order},
    )
    fig.update_layout(
        font=dict(family="JetBrains Mono", size=12),
        paper_bgcolor=BG_LIGHT,
        plot_bgcolor=BG_LIGHT,
        showlegend=False,
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_range=[0, 1.05],
        xaxis=dict(tickfont=dict(color=TEXT_DARK), title=""),
        yaxis=dict(tickfont=dict(color=TEXT_DARK), gridcolor=GRID_LIGHT, title=metric),
        title=dict(font=dict(color=TEXT_DARK, size=14, family="JetBrains Mono"), x=0.5),
    )
    fig.update_traces(jitter=0.3, pointpos=-1.8, marker=dict(size=5, opacity=0.7, line=dict(width=1, color=TEXT_DARK)))
    return fig


# -------------------------------------------------------------------------
# ECharts heatmap for confusion matrix
# -------------------------------------------------------------------------
def _render_heatmap(confusion: list, title: str) -> None:
    labels = ["Benign", "Phishing", "Malware", "Defacement"]

    totals = [sum(row) for row in confusion]
    matrix_pct = [
        [v / totals[r] * 100 if totals[r] > 0 else 0 for v in confusion[r]]
        for r in range(4)
    ]

    fig = go.Figure(go.Heatmap(
        x=labels,
        y=labels,
        z=[matrix_pct[i] for i in range(4)],
        text=[[f"{confusion[i][j]}\n({matrix_pct[i][j]:.1f}%)" for j in range(4)] for i in range(4)],
        texttemplate="%{text}",
        textfont=dict(
            color="#1e293b",
            family="JetBrains Mono",
            size=12,
            weight="bold",
        ),
        colorscale=[
            [0.0, "#f0f9ff"],
            [0.25, "#bae6fd"],
            [0.5, "#38bdf8"],
            [0.75, "#0284c7"],
            [1.0, "#0c4a6e"],
        ],
        zmin=0,
        zmax=100,
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{customdata}<br>Pct: %{z:.1f}%<extra></extra>",
        customdata=[[confusion[i][j] for j in range(4)] for i in range(4)],
        showscale=True,
        colorbar=dict(
            title="%",
            tickfont=dict(color=TEXT_DARK, family="JetBrains Mono", size=10),
            tickformat=".0f",
        ),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_DARK, size=13, family="JetBrains Mono"), x=0.5),
        font=dict(family="JetBrains Mono", size=11),
        paper_bgcolor=BG_LIGHT,
        plot_bgcolor=BG_LIGHT,
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(tickfont=dict(color=TEXT_DARK), title="", side="bottom", gridcolor=GRID_LIGHT),
        yaxis=dict(tickfont=dict(color=TEXT_DARK), title="True Label", autorange="reversed", gridcolor=GRID_LIGHT),
    )
    st.plotly_chart(fig, use_container_width=True)


ROC_OVR_PATH = os.path.join(model_config.LOG_DIR, "model_comparison", "roc_curves.json")


def _render_roc_ovr() -> None:
    if not os.path.exists(ROC_OVR_PATH):
        st.info("Run `python -m src.models.compute_inference_metrics` to generate ROC curves.")
        return
    roc_data = _load_json(ROC_OVR_PATH)
    classes = ["Benign", "Phishing", "Malware", "Defacement"]
    model_colors = {"DistilBERT": "#0891b2", "XGBoost": "#059669", "LogisticRegression": "#d97706", "RandomForest": "#7c3aed"}
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"<b>{c}</b>" for c in classes],
        horizontal_spacing=0.18,
        vertical_spacing=0.18,
    )
    for idx, cls in enumerate(classes):
        row, col = idx // 2 + 1, idx % 2 + 1
        aucs_for_legend: List[Tuple[str, float]] = []
        for model_name, model_key in [
            ("DistilBERT", "distilbert"),
            ("XGBoost", "xgboost"),
            ("LogisticRegression", "logistic_regression"),
            ("RandomForest", "random_forest"),
        ]:
            if model_key not in roc_data:
                continue
            cls_data = roc_data[model_key].get(cls.lower(), {})
            fpr = cls_data.get("fpr", [])
            tpr = cls_data.get("tpr", [])
            auc = cls_data.get("auc", 0)
            if not fpr or not tpr:
                continue
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=model_name,
                line=dict(color=model_colors[model_name], width=2.5),
                legendgroup=model_name,
                showlegend=(idx == 0),
            ), row=row, col=col)
            aucs_for_legend.append((model_name, auc))

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color="#94a3b8", width=1, dash="dash"),
            showlegend=False,
        ), row=row, col=col)

        fig.update_xaxes(
            range=[0, 1.02], tickfont=dict(color=TEXT_DARK, size=10),
            title=dict(text="FPR", font=dict(color=TEXT_DARK, size=11)),
            tick0=0, dtick=0.2,
            gridcolor=GRID_LIGHT, gridwidth=1,
            mirror=True, zeroline=False,
            row=row, col=col,
        )
        fig.update_yaxes(
            range=[0, 1.02], tickfont=dict(color=TEXT_DARK, size=10),
            title=dict(text="TPR", font=dict(color=TEXT_DARK, size=11)),
            tick0=0, dtick=0.2,
            gridcolor=GRID_LIGHT, gridwidth=1,
            mirror=True, zeroline=False,
            row=row, col=col,
        )

        auc_text = "<br>".join(
            f"<span style='color:{model_colors[m]}'>{m}: {a:.3f}</span>"
            for m, a in aucs_for_legend
        )
        fig.add_annotation(
            x=0.97, y=0.97,
            xref="x domain", yref="y domain",
            text=f"<b>AUC</b><br>{auc_text}",
            showarrow=False,
            font=dict(family="JetBrains Mono", size=10, color=TEXT_DARK),
            align="left",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=GRID_LIGHT,
            borderwidth=1,
            borderpad=4,
            xanchor="right", yanchor="top",
        )

    fig.update_layout(
        title=dict(
            text="One-vs-Rest ROC Curves per Class",
            font=dict(color=TEXT_DARK, size=14, family="JetBrains Mono"),
            x=0.5,
        ),
        font=dict(family="JetBrains Mono", size=11, color=TEXT_DARK),
        paper_bgcolor=BG_LIGHT,
        plot_bgcolor=BG_LIGHT,
        height=680,
        showlegend=True,
        legend=dict(
            font=dict(color=TEXT_DARK, size=10),
            orientation="h",
            yanchor="bottom",
            y=-0.02,
            xanchor="center",
            x=0.5,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------------
# Holdout comparison
# -------------------------------------------------------------------------
def _render_holdout_comparison() -> None:
    st.markdown("### 80/20 Holdout Comparison")

    holdout_data = _load_json(HOLDOUT_RESULTS_PATH)
    distilbert_data = _load_json(DISTILBERT_METRICS_PATH)

    rows = []
    confusion_lookup: Dict[str, list] = {}

    for item in holdout_data.get("results", []):
        model_name = item.get("model", "Unknown")
        confusion = item.get("confusion_matrix")
        specificity = _specificity_from_confusion(confusion) if confusion else 0
        rows.append({
            "Model": model_name,
            "Accuracy": item.get("accuracy", 0),
            "Precision": item.get("precision", 0),
            "Recall": item.get("recall", 0),
            "F1-Score": item.get("f1", 0),
            "Specificity": specificity,
            "ROC-AUC": item.get("roc_auc", 0),
        })
        if confusion:
            confusion_lookup[model_name] = confusion

    if distilbert_data:
        confusion = distilbert_data.get("eval_confusion_matrix")
        specificity = _specificity_from_confusion(confusion) if confusion else 0
        rows = [r for r in rows if r["Model"] != "DistilBERT"]
        rows.append({
            "Model": "DistilBERT",
            "Accuracy": distilbert_data.get("eval_accuracy", 0),
            "Precision": distilbert_data.get("eval_precision", 0),
            "Recall": distilbert_data.get("eval_recall", 0),
            "F1-Score": distilbert_data.get("eval_f1", 0),
            "Specificity": specificity,
            "ROC-AUC": distilbert_data.get("eval_roc_auc", 0),
        })
        if confusion:
            confusion_lookup["DistilBERT"] = confusion

    if not rows:
        st.warning("No holdout results found.")
        return

    df = pd.DataFrame(rows)
    order = ["DistilBERT", "XGBoost", "LogisticRegression", "RandomForest"]
    df["_sort"] = df["Model"].map({m: i for i, m in enumerate(order)})
    df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)

    st.markdown("#### Model Metrics (Bar Charts)")
    metrics_to_plot = ["F1-Score", "Precision", "Recall", "ROC-AUC", "Accuracy"]
    cols = st.columns(len(metrics_to_plot))
    for i, metric in enumerate(metrics_to_plot):
        with cols[i]:
            fig = _make_bar_chart(df[["Model", metric]], metric, metric)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Confusion Matrices (Interactive Heatmaps)")
    st.caption("Hover over cells for counts. Bright diagonal = correct predictions.")
    heat_cols = st.columns(len(confusion_lookup))
    for idx, (model_name, confusion) in enumerate(confusion_lookup.items()):
        with heat_cols[idx % 4 if len(confusion_lookup) <= 4 else 2]:
            _render_heatmap(confusion, model_name)


# -------------------------------------------------------------------------
# 10-fold CV results
# -------------------------------------------------------------------------
def _render_cv_results(cv_results: Dict[str, Any]) -> None:
    summary = cv_results.get("summary", {})
    if not summary:
        st.info("No 10-fold CV summary available.")
        return

    st.markdown("#### Metric Distribution Across 10 Folds (Box Plots)")

    model_display = {
        "DistilBERT": "DistilBERT",
        "XGBoost": "XGBoost",
        "LogisticRegression": "Logistic Regression",
        "RandomForest": "Random Forest",
    }

    metric_map = {
        "F1-Score": "f1",
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "ROC-AUC": "roc_auc",
    }

    for metric_name, metric_key in metric_map.items():
        fig = go.Figure()
        all_values = []
        for model_name in ["DistilBERT", "XGBoost", "LogisticRegression", "RandomForest"]:
            if model_name in summary:
                all_values.extend(summary[model_name].get(f"{metric_key}_values", []))

        for model_name in ["DistilBERT", "XGBoost", "LogisticRegression", "RandomForest"]:
            if model_name not in summary:
                continue
            model_data = summary[model_name]
            values = model_data.get(f"{metric_key}_values", [])
            if not values:
                continue
            color = ACCENT_COLORS_LIGHT[list(model_display.keys()).index(model_name)]
            fig.add_trace(go.Box(
                y=values,
                name=model_display.get(model_name, model_name),
                marker_color=color,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                text=[f"{v:.4f}" for v in values],
                hoverinfo="y+text",
            ))

        if all_values:
            data_min, data_max = min(all_values), max(all_values)
            pad = (data_max - data_min) * 0.3 or 0.01
            y_min = max(0, data_min - pad)
            y_max = min(1.05, data_max + pad)
        else:
            y_min, y_max = 0, 1.05

        fig.update_layout(
            title=dict(text=metric_name, font=dict(color=TEXT_DARK, size=14, family="JetBrains Mono"), x=0.5),
            font=dict(family="JetBrains Mono", size=12),
            paper_bgcolor=BG_LIGHT,
            plot_bgcolor=BG_LIGHT,
            height=320,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
            yaxis_range=[y_min, y_max],
            xaxis=dict(tickfont=dict(color=TEXT_DARK), title=""),
            yaxis=dict(tickfont=dict(color=TEXT_DARK), gridcolor=GRID_LIGHT, title=metric_name),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Summary statistics table
    st.markdown("#### Summary Statistics (Mean ± Std across 10 folds)")
    table_rows = []
    for model_name, display_name in model_display.items():
        if model_name not in summary:
            continue
        m = summary[model_name]
        table_rows.append({
            "Model": display_name,
            "Accuracy": f"{m.get('mean_accuracy', 0):.4f} ± {m.get('std_accuracy', 0):.4f}",
            "Precision": f"{m.get('mean_precision', 0):.4f} ± {m.get('std_precision', 0):.4f}",
            "Recall": f"{m.get('mean_recall', 0):.4f} ± {m.get('std_recall', 0):.4f}",
            "F1-Score": f"{m.get('mean_f1', 0):.4f} ± {m.get('std_f1', 0):.4f}",
            "ROC-AUC": f"{m.get('mean_roc_auc', 0):.4f} ± {m.get('std_roc_auc', 0):.4f}",
        })

    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Aggregated confusion matrix heatmap (averaged across folds)
    st.markdown("#### Averaged Confusion Matrix")
    model_keys_map = {
        "DistilBERT": "distilbert",
        "XGBoost": "xgboost",
        "LogisticRegression": "logistic_regression",
        "RandomForest": "random_forest",
    }

    agg_heat_cols = st.columns(2)
    for idx, model_name in enumerate(["DistilBERT", "XGBoost", "LogisticRegression", "RandomForest"]):
        key = model_keys_map[model_name]
        matrices = []
        for fold in cv_results.get("folds", []):
            if key in fold and isinstance(fold[key], dict) and "confusion_matrix" in fold[key]:
                matrices.append(fold[key]["confusion_matrix"])

        if not matrices:
            continue

        avg_matrix = []
        for i in range(4):
            row = []
            for j in range(4):
                row.append(sum(m[i][j] for m in matrices) / len(matrices))
            avg_matrix.append(row)

        with agg_heat_cols[idx % 2]:
            _render_heatmap(avg_matrix, f"{model_name} (avg of {len(matrices)} folds)")


# -------------------------------------------------------------------------
# URL Tester
# -------------------------------------------------------------------------
def _render_tester() -> None:
    st.markdown("### URL Tester")

    model_options = {
        "DistilBERT (Fine-tuned)": "distilbert",
        "XGBoost Baseline": "xgboost",
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
    }

    selected_model = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0,
        help="Choose which model to use for URL classification"
    )

    model_key = model_options[selected_model]

    detector = get_detector()

    col1, col2 = st.columns([3, 1])
    with col1:
        url_input = st.text_input(
            "Enter URL to analyze",
            placeholder="https://example.com/login",
            label_visibility="collapsed",
        )
    with col2:
        analyze = st.button("Analyze", type="primary")

    if analyze and url_input.strip():
        result = detector.predict(url_input.strip(), model_name=selected_model)
        label = result["label"]
        blocked = result["blocked"]
        confidence = result["probability"]
        class_probs = result["class_probabilities"]

        st.info(f"**Model Used:** {selected_model}", icon="🤖")

        label_colors = {
            "Benign": "#10b981",
            "Phishing": "#f59e0b",
            "Malware": "#ef4444",
            "Defacement": "#a78bfa",
        }
        color = label_colors.get(label, "#e2e8f0")

        result_cols = st.columns(4)
        with result_cols[0]:
            st.markdown(f"<div class='metric-label'>PREDICTION</div><div class='metric-value' style='color:{color}'>{label}</div>", unsafe_allow_html=True)
        with result_cols[1]:
            st.metric("Confidence", f"{confidence:.1%}")
        with result_cols[2]:
            st.metric("Malicious Prob", f"{sum(class_probs[1:]):.1%}")
        with result_cols[3]:
            st.metric("Blocked", "YES" if blocked else "NO", delta_color="normal")

        st.markdown("#### Class Probabilities")
        prob_df = pd.DataFrame({
            "Class": ["Benign", "Phishing", "Malware", "Defacement"],
            "Probability": class_probs,
        })
        class_colors = ["#0891b2", "#d97706", "#ef4444", "#7c3aed"]
        fig = px.bar(
            prob_df, x="Class", y="Probability",
            color="Class",
            color_discrete_sequence=class_colors,
            text_auto=".2f",
            template=PLOTLY_TEMPLATE,
            title="Per-Class Probability Distribution",
        )
        fig.update_layout(
            font=dict(family="JetBrains Mono", size=12),
            paper_bgcolor=BG_LIGHT,
            plot_bgcolor=BG_LIGHT,
            height=260,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis_range=[0, 1.05],
            xaxis=dict(tickfont=dict(color=TEXT_DARK)),
            yaxis=dict(tickfont=dict(color=TEXT_DARK), gridcolor=GRID_LIGHT),
            title=dict(font=dict(color=TEXT_DARK, size=13), x=0.5),
        )
        fig.update_traces(textposition="outside", textfont=dict(color=TEXT_DARK), marker_line_width=1.5)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="glass-card">
          <span class="metric-label">DECISION REASON</span><br/>
          <code style="color:#64748b; font-size:0.8rem">{result.get('decision_reason', '')}</code>
        </div>
        """, unsafe_allow_html=True)

    elif analyze:
        st.warning("Please enter a URL.")


# -------------------------------------------------------------------------
# Evaluation Dashboard
# -------------------------------------------------------------------------
def _render_evaluation() -> None:
    st.markdown("### Evaluation Dashboard")

    refresh_col1, refresh_col2 = st.columns([1, 3])
    with refresh_col1:
        if st.button("🔄 Refresh Metrics"):
            with st.spinner("Refreshing..."):
                from src.local_pipeline import gather_dashboard_metrics
                gather_dashboard_metrics(retrain_distilbert=False, run_baseline_cv=False)
            st.success("Metrics refreshed.", icon="✅")
    st.markdown("---")

    st.markdown("#### Inference Performance")
    st.caption("Precomputed on 5,000-sample fixed test set. Latency: avg ms/URL over 5 passes. Memory: GPU VRAM (DistilBERT) or model file size (baselines).")
    INFERENCE_METRICS_PATH = os.path.join(model_config.LOG_DIR, "model_comparison", "inference_metrics.json")
    MODEL_SIZES_PATH = os.path.join(model_config.LOG_DIR, "model_comparison", "model_sizes.json")
    ROC_OVR_AVAILABLE = os.path.exists(ROC_OVR_PATH)

    # If no data, let user skip this section
    if not os.path.exists(INFERENCE_METRICS_PATH):
        st.warning("⚠ Inference metrics not found. Run option **9 (Gather All Metrics)** first to generate latency, memory, and ROC data.", icon="⚡")
        skip_inference = st.button("Skip this section", key="skip_inference")
        if not skip_inference:
            st.stop()
        st.markdown("---")
    else:
        perf_data = _load_json(INFERENCE_METRICS_PATH)
        if perf_data:
            models = list(perf_data.keys())
            latencies = [perf_data[m].get("ms_per_url", 0) for m in models]
            mems = [perf_data[m].get("memory_mb", 0) for m in models]
            col1, col2 = st.columns(2)
            with col1:
                lat_df = pd.DataFrame({"Model": models, "ms/URL": latencies})
                lat_fig = px.bar(lat_df, x="Model", y="ms/URL", color="Model", color_discrete_sequence=ACCENT_COLORS_LIGHT, text_auto=".2f", title="Inference Latency (ms per URL)")
                lat_fig.update_layout(paper_bgcolor=BG_LIGHT, plot_bgcolor=BG_LIGHT, font=dict(family="JetBrains Mono", size=11), title=dict(font=dict(color=TEXT_DARK, size=12), x=0.5), showlegend=False, height=260, margin=dict(l=15, r=15, t=40, b=15), xaxis=dict(tickfont=dict(color=TEXT_DARK)), yaxis=dict(tickfont=dict(color=TEXT_DARK), gridcolor=GRID_LIGHT))
                lat_fig.update_traces(textposition="outside", textfont=dict(color=TEXT_DARK), marker_line_width=1.5)
                st.plotly_chart(lat_fig, use_container_width=True)
            with col2:
                mem_df = pd.DataFrame({"Model": models, "Memory (MB)": mems})
                mem_fig = px.bar(mem_df, x="Model", y="Memory (MB)", color="Model", color_discrete_sequence=ACCENT_COLORS_LIGHT, text_auto=".1f", title="Memory Usage (MB)")
                mem_fig.update_layout(paper_bgcolor=BG_LIGHT, plot_bgcolor=BG_LIGHT, font=dict(family="JetBrains Mono", size=11), title=dict(font=dict(color=TEXT_DARK, size=12), x=0.5), showlegend=False, height=260, margin=dict(l=15, r=15, t=40, b=15), xaxis=dict(tickfont=dict(color=TEXT_DARK)), yaxis=dict(tickfont=dict(color=TEXT_DARK), gridcolor=GRID_LIGHT))
                mem_fig.update_traces(textposition="outside", textfont=dict(color=TEXT_DARK), marker_line_width=1.5)
                st.plotly_chart(mem_fig, use_container_width=True)
        if os.path.exists(MODEL_SIZES_PATH):
            sizes_data = _load_json(MODEL_SIZES_PATH)
            if sizes_data:
                size_rows = [{"Model": m, "Size / Params": sizes_data[m].get("size_str", "N/A"), "Accuracy": sizes_data[m].get("accuracy", "N/A"), "ROC-AUC": sizes_data[m].get("roc_auc", "N/A")} for m in sizes_data]
                st.markdown("#### Model Size Comparison")
                st.dataframe(pd.DataFrame(size_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### One-vs-Rest ROC Curves (Per-Class)")
    st.caption("Computed on fixed 5,000-sample test set. Higher AUC = better class discrimination.")
    if not ROC_OVR_AVAILABLE:
        st.warning("⚠ ROC curves not found. Run option **9 (Gather All Metrics)** first to generate OvR ROC data.", icon="📈")
        skip_roc = st.button("Skip this section", key="skip_roc")
        if not skip_roc:
            st.stop()
    else:
        _render_roc_ovr()

    st.markdown("---")
    st.markdown("#### Holdout Comparison")
    st.caption("DistilBERT (full-data) vs XGBoost vs LogisticRegression vs RandomForest — 80/20 holdout.")
    _render_holdout_comparison()

    st.markdown("---")
    st.markdown("#### Ten-Fold Cross-Validation")
    st.caption("10-fold CV: 25k samples per fold (20k train / 5k val). All models trained per-fold.")

    cv_mode = st.radio(
        "CV Results",
        ["Use saved results (no retrain)", "Run baseline ten-fold CV now"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if cv_mode == "Run baseline ten-fold CV now":
        with st.spinner("Running 10-fold CV (this takes ~80 min on GPU)..."):
            results = run_kfold_cross_validation(model_type="all", n_splits=10)
        _render_cv_results(results)
    elif os.path.exists(CV_RESULTS_PATH):
        cv_data = _load_json(CV_RESULTS_PATH)
        _render_cv_results(cv_data)
    else:
        st.warning("⚠ No 10-fold CV results found. Run option **4 (Run 10-Fold CV)** or **9 (Gather All Metrics)** first.", icon="📊")
        skip_cv = st.button("Skip this section", key="skip_cv")
        if not skip_cv:
            st.stop()


# -------------------------------------------------------------------------
# Methods
# -------------------------------------------------------------------------
def _render_methods() -> None:
    st.markdown("""
    **Research Design**  
    This study uses a quantitative, experimental design to compare a BERT-based model and an XGBoost classifier for malicious URL detection.  
    Both models are trained and evaluated under comparable settings for valid performance and latency comparison.

    **Data and Code Availability**  
    All source code, preprocessing scripts, and model settings are version-controlled in the project repository.  
    Datasets (including the large URL corpus) are documented and linked in manuscript appendices.

    **Data Preprocessing and Feature Engineering**  
    - **BERT pipeline:** raw URLs are tokenized, padded/truncated, and converted into model inputs.  
    - **XGBoost pipeline:** handcrafted lexical/structural URL features plus TF-IDF features are extracted.

    **Experimental Setup and Training**  
    Data is split with stratified sampling (80/20 holdout), and ten-fold cross-validation is used for robustness.  
    Class imbalance is handled with cost-sensitive learning (class weighting / sample weighting).

    **Mapping of Research Questions to Methodologies**  
    - **RQ1:** Side-by-side confusion matrix, accuracy, precision, recall, and F1-score comparison.  
    - **RQ2:** Class-level recall and macro-F1 to assess minority-class sensitivity.  
    - **RQ3:** Inference latency and resource usage benchmarking for gateway deployment.  
    """)


SHAP_BASE_DIR = os.path.join(model_config.LOG_DIR, "figures")


def _render_xai() -> None:
    st.markdown("### Explainable AI — SHAP Visualizations")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) assigns each feature a **SHAP value** per prediction:
    positive means the feature pushed toward a class, negative means it pushed away.
    Each dot = one URL sample. Position on X-axis = impact magnitude. Color = feature value.
    """)
    st.markdown("---")

    xgb_shap_path = os.path.join(SHAP_BASE_DIR, "xgboost_shap_summary.png")
    xgb_shap_multi_path = os.path.join(SHAP_BASE_DIR, "xgboost_shap")

    distilbert_shap_path = os.path.join(SHAP_BASE_DIR, "distilbert_shap_summary.png")
    distilbert_shap_multi_path = os.path.join(SHAP_BASE_DIR, "distilbert_shap")

    model_choice = st.radio("Select Model", ["XGBoost SHAP", "DistilBERT SHAP"], horizontal=True)

    if model_choice == "XGBoost SHAP":
        st.markdown("#### XGBoost SHAP — All Classes")
        st.caption("Feature impact on predictions. Red = high feature value, Blue = low. Features sorted by mean |SHAP|.")
        if os.path.exists(xgb_shap_path):
            st.image(xgb_shap_path)
        elif os.path.exists(xgb_shap_multi_path):
            classes = ["benign", "phishing", "malware", "defacement"]
            cols = st.columns(2)
            for i, cls in enumerate(classes):
                img_path = os.path.join(xgb_shap_multi_path, f"{cls}.png")
                with cols[i % 2]:
                    st.markdown(f"**{cls.capitalize()}**")
                    if os.path.exists(img_path):
                        st.image(img_path)
                    else:
                        st.info(f"Run `python -m src.models.compute_xgboost_shap` to generate {cls} SHAP plot.")
        else:
            st.info("Run `python -m src.models.compute_xgboost_shap` to generate XGBoost SHAP plots.")

    else:
        st.markdown("#### DistilBERT SHAP — Token Importance")
        st.caption("Token-level impact on predictions via KernelSHAP. Run `python -m src.models.compute_distilbert_shap` first.")
        if os.path.exists(distilbert_shap_path):
            st.image(distilbert_shap_path)
        elif os.path.exists(distilbert_shap_multi_path):
            classes = ["benign", "phishing", "malware", "defacement"]
            cols = st.columns(2)
            for i, cls in enumerate(classes):
                img_path = os.path.join(distilbert_shap_multi_path, f"{cls}.png")
                with cols[i % 2]:
                    st.markdown(f"**{cls.capitalize()}**")
                    if os.path.exists(img_path):
                        st.image(img_path)
                    else:
                        st.info(f"Run `python -m src.models.compute_distilbert_shap` to generate {cls} SHAP plot.")
        else:
            st.info("Run `python -m src.models.compute_distilbert_shap` to generate DistilBERT SHAP plots (takes ~15-20 min).")

    st.markdown("---")
    st.markdown("""
    **Interpretation Guide**

    **XGBoost SHAP:** Each plot shows how feature values (red=high, blue=low) shift the prediction for each class.
    A feature with wide spread on the X-axis has strong influence. Features like URL length, special character count,
    and entropy are typically the most discriminative for malicious URL detection.

    **DistilBERT SHAP:** The model attends to subword tokens within URLs. Tokens highlighted as high-impact indicate
    the transformer learned patterns such as typosquatting (e.g., "paypa1.com"), suspicious TLDs, or encoded characters.
    """)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main() -> None:
    st.title("🛡️ smtpBERT Local Dashboard")
    st.caption("Local training / testing / evaluation dashboard. All data stays on your machine.")

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 URL Tester", "📊 Evaluation", "📋 Methods", "🧠 XAI / SHAP"])
    with tab1:
        _render_tester()
    with tab2:
        _render_evaluation()
    with tab3:
        _render_methods()
    with tab4:
        _render_xai()


if __name__ == "__main__":
    main()