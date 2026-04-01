"""
Plot CV (OOF AUC) vs LB score for all submitted experiments.
Interactive plotly scatter plot, color-coded by model type.

Usage: uv run python scripts/plot_cv_lb.py
"""

import plotly.graph_objects as go

# ============================================================================
# Data: (name, cv, lb, model_type)
# model_type: "single" or "ensemble"
# ============================================================================
DATA = [
    # Single models
    ("LogReg baseline", 0.9079, 0.90504, "single"),
    ("XGBoost depth1", 0.9135, 0.91039, "single"),
    ("XGBoost baseline", 0.9164, 0.91391, "single"),
    ("LightGBM baseline", 0.9163, 0.91378, "single"),
    ("Bartz baseline", 0.9158, 0.91405, "single"),
    ("XGBoost lossguide", 0.9159, 0.91311, "single"),
    ("XGBoost orig_ref", 0.91853, 0.91644, "single"),
    ("XGB Optuna 5-fold", 0.91893, 0.91656, "single"),
    ("LGBM orig_ref", 0.91844, 0.91631, "single"),
    ("LGBM Optuna", 0.91880, 0.91659, "single"),
    ("CatBoost orig_ref", 0.91853, 0.91640, "single"),
    ("RealMLP", 0.91895, 0.91655, "single"),
    ("MLP (PyTorch)", 0.91720, 0.91483, "single"),
    ("LogReg orig_ref", 0.91579, 0.91285, "single"),
    ("Ridge orig_ref", 0.91084, 0.90789, "single"),
    ("TabM 5-fold", 0.91854, 0.91657, "single"),
    ("Logit3 TE-Pair", 0.91595, 0.91348, "single"),
    ("XGB full data", None, 0.91662, "single"),
    ("Ridge->XGB 20f (NB1)", 0.91922, 0.91685, "single"),
    ("Ridge->XGB 5f (adapted)", 0.91888, 0.91665, "single"),
    ("Ridge->LGB 20f", 0.91914, 0.91680, "single"),
    # Ensemble models
    ("SLSQP ensemble", 0.91883, 0.91630, "ensemble"),
    ("HC v1", 0.91939, 0.91703, "ensemble"),
    ("HC v2", 0.91955, 0.91712, "ensemble"),
    ("HC v3", 0.91955, 0.91711, "ensemble"),
    ("HC v4", 0.91956, 0.91712, "ensemble"),
    ("HC v4+", 0.91957, 0.91712, "ensemble"),
    ("HC v4++", 0.91960, 0.91715, "ensemble"),
]

# ============================================================================
# Plot
# ============================================================================
fig = go.Figure()

colors = {"single": "#2563eb", "ensemble": "#dc2626"}
symbols = {"single": "circle", "ensemble": "diamond"}
labels = {"single": "Single Model", "ensemble": "Ensemble"}

for mtype in ["single", "ensemble"]:
    subset = [(name, cv, lb) for name, cv, lb, t in DATA if t == mtype and cv is not None]
    if not subset:
        continue
    names, cvs, lbs = zip(*subset)
    fig.add_trace(go.Scatter(
        x=list(cvs),
        y=list(lbs),
        mode="markers+text",
        name=labels[mtype],
        text=list(names),
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            color=colors[mtype],
            size=10,
            symbol=symbols[mtype],
            line=dict(width=1, color="white"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "CV: %{x:.5f}<br>"
            "LB: %{y:.5f}<br>"
            "<extra></extra>"
        ),
    ))

# y=x reference line
all_cvs = [cv for _, cv, _, _ in DATA if cv is not None]
all_lbs = [lb for _, _, lb, _ in DATA]
min_val = min(min(all_cvs), min(all_lbs)) - 0.001
max_val = max(max(all_cvs), max(all_lbs)) + 0.001
fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode="lines",
    line=dict(color="gray", width=1, dash="dash"),
    showlegend=False,
    hoverinfo="skip",
))

fig.update_layout(
    title="CV (OOF AUC) vs LB Score",
    xaxis_title="CV (OOF AUC-ROC)",
    yaxis_title="LB (Public Score)",
    xaxis=dict(tickformat=".4f"),
    yaxis=dict(tickformat=".4f"),
    width=1000,
    height=700,
    legend=dict(x=0.02, y=0.98),
    template="plotly_white",
)

output_path = "scripts/cv_lb_plot.html"
fig.write_html(output_path)
print(f"Saved to {output_path}")
fig.show()
