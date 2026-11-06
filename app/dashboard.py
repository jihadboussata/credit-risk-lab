"""Run ``python app/dashboard.py`` from the repository root.

The application reads committed evaluation outputs, never trains a model, and
uses validation predictions exclusively for its threshold exploration. Synthetic
profile scoring is enabled only when a trusted local model artifact is present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
NAVY, TEAL, GOLD, MUTED = "#142f44", "#008879", "#d4a64a", "#70808b"
MODEL_LABELS = {
    "dummy": "Prevalence baseline", "logistic_l1": "L1 logistic · lasso",
    "logistic_l2": "L2 logistic · ridge", "decision_tree": "Decision tree",
    "random_forest": "Random forest", "xgboost": "XGBoost", "svm_rbf": "RBF SVM",
}


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", " ").title())


def read_csv(path: Path) -> pd.DataFrame:
    """Read an optional evaluation export without inventing fallback results."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, float_precision="round_trip")


def load_results(root: Path = ROOT) -> dict:
    path = root / "reports" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {
        "summary": summary,
        "metrics": read_csv(root / "reports" / "metrics.csv"),
        "validation": read_csv(root / "reports" / "validation_predictions.csv"),
        "test": read_csv(root / "reports" / "test_predictions.csv"),
    }


def winner_name(summary: dict, metrics: pd.DataFrame) -> str | None:
    """Use the recorded validation decision; never select a model on test data."""
    for key in ("winner", "champion", "selected_model", "model_name"):
        value = summary.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for name_key in ("model_name", "model", "name"):
                if isinstance(value.get(name_key), str):
                    return value[name_key]
    return None


def recorded_threshold(summary: dict) -> float:
    for key in ("threshold", "selected_threshold", "decision_threshold"):
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for inner in ("value", "threshold", "selected"):
                if isinstance(value.get(inner), (int, float)):
                    return float(value[inner])
    return 0.5


def threshold_metrics(y_true, probability, threshold: float, fn_cost=5.0, fp_cost=1.0) -> dict:
    """Calculate observable classification workload under hypothetical costs.

    The cost is FN*fn_cost + FP*fp_cost in arbitrary units. It is not observed
    financial loss, estimated savings, or the effect of a review intervention.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError("Labels and probabilities must be nonempty matching vectors.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary labels and finite probabilities in [0, 1].")
    if not np.isfinite([threshold, fn_cost, fp_cost]).all() or not 0 <= threshold <= 1 or fn_cost < 0 or fp_cost < 0:
        raise ValueError("Threshold must be in [0, 1] and costs must be nonnegative.")
    pred = p >= threshold
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "review_rate": float(pred.mean()),
        "cost": float(fn * fn_cost + fp * fp_cost),
        "n": len(y),
    }


def styled_figure(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color=NAVY)),
        template="plotly_white", height=345,
        margin=dict(l=48, r=20, t=45, b=48),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color=NAVY, size=12),
        colorway=[TEAL, NAVY, GOLD, "#71aaa5", "#9c6d57", "#8399ac"],
        legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(gridcolor="#e9eeec", zeroline=False)
    fig.update_yaxes(gridcolor="#e9eeec", zeroline=False)
    return fig


def empty_figure(message: str) -> go.Figure:
    fig = styled_figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def curve_figure(predictions: pd.DataFrame, model_name: str | None, kind: str) -> go.Figure:
    if predictions.empty or model_name is None:
        return empty_figure("Run the training workflow to generate evaluation curves.")
    rows = predictions[predictions["model"] == model_name]
    if rows.empty:
        return empty_figure("No predictions available for the recorded champion.")
    y, p = rows["y_true"], rows["probability"]
    if kind == "roc":
        x, z, _ = roc_curve(y, p)
        fig = styled_figure("Ranking performance · held-out test")
        fig.add_trace(go.Scatter(x=x, y=z, mode="lines", name="Champion", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random ranking", line=dict(color=MUTED, dash="dot")))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    elif kind == "pr":
        precision, recall, _ = precision_recall_curve(y, p)
        fig = styled_figure("Precision–recall · held-out test")
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Champion", line=dict(width=3, shape="hv")))
        fig.add_hline(y=float(y.mean()), line_dash="dot", line_color=MUTED, annotation_text="Test prevalence")
        fig.update_xaxes(title="Recall", range=[0, 1])
        fig.update_yaxes(title="Precision", range=[0, 1.02])
    else:
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        fig = styled_figure("Probability calibration · held-out test")
        fig.add_trace(go.Scatter(x=predicted, y=observed, mode="lines+markers", name="Champion", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(color=MUTED, dash="dot")))
        fig.update_xaxes(title="Mean predicted probability", range=[0, 1])
        fig.update_yaxes(title="Observed default frequency", range=[0, 1])
    return fig


def comparison_figure(metrics: pd.DataFrame, summary: dict | None = None) -> go.Figure:
    if metrics.empty or "split" not in metrics:
        return empty_figure("No results yet. Run python -m credit_risk.run --root .")
    rows = metrics[(metrics["split"] == "validation") & ~metrics["model"].str.endswith("__policy")].copy()
    selection = (summary or {}).get("model_selection", {}).get("validation_average_precision", {})
    if selection:
        rows["average_precision"] = [selection.get(name, value) for name, value in zip(rows.model, rows.average_precision)]
    rows = rows.sort_values("average_precision")
    champion = (summary or {}).get("winner")
    fig = styled_figure("Initial model selection · validation AP")
    fig.add_trace(go.Bar(
        x=rows["average_precision"], y=rows["model"].map(model_label), orientation="h",
        marker_color=[TEAL if name == champion else "#aec9c5" for name in rows.model],
        text=[f"{v:.3f}" for v in rows["average_precision"]], textposition="outside",
        hovertemplate="%{y}<br>Average precision: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(margin=dict(l=138, r=45, t=45, b=40), showlegend=False)
    fig.update_xaxes(title="Average precision (higher is better)", range=[0, min(1, float(rows["average_precision"].max()) * 1.2)] if len(rows) else [0, 1])
    return fig


def importance_figure(root: Path = ROOT) -> go.Figure:
    paths = [root / "reports" / name for name in ("permutation_importance.csv", "feature_importance.csv")]
    table = next((read_csv(p) for p in paths if p.exists()), pd.DataFrame())
    if table.empty:
        return empty_figure("Feature-importance export is not available.")
    feature = next((c for c in ("feature", "feature_name") if c in table), table.columns[0])
    score = next((c for c in ("importance_mean", "importance", "mean_importance", "mean") if c in table), table.columns[1])
    rows = table.sort_values(score).tail(8)
    fig = styled_figure("Permutation importance · validation")
    fig.add_trace(go.Bar(x=rows[score], y=rows[feature], orientation="h", marker_color=TEAL))
    fig.update_layout(margin=dict(l=150, r=22, t=45, b=50))
    fig.update_xaxes(title="Drop in average precision after permutation")
    return fig


def synthetic_profile(limit: float, bill: float, payment: float, delay: int) -> pd.DataFrame:
    """Construct one explicitly synthetic six-month profile for a local demo.

    The user supplies four values; they are repeated across all six months.
    This is an illustrative scenario, not a replacement for a full history.
    """
    vals = np.asarray([limit, bill, payment, delay], dtype=float)
    if not np.isfinite(vals).all() or not 10000 <= limit <= 1000000:
        raise ValueError("Enter a credit limit between NT$10,000 and NT$1,000,000.")
    if not 0 <= bill <= 1500000 or not 0 <= payment <= 1000000:
        raise ValueError("Bill and payment amounts must be within the displayed ranges.")
    if int(delay) != delay or delay not in [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]:
        raise ValueError("Choose one of the documented repayment-status codes.")
    row = {"id": -1, "limit_bal": float(limit), "sex": 2, "education": 2, "marriage": 1, "age": 35}
    for column in ("pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"):
        row[column] = int(delay)
    for month in range(1, 7):
        row[f"bill_amt{month}"] = float(bill)
        row[f"pay_amt{month}"] = float(payment)
    return pd.DataFrame([row])


def kpi(value: str, label: str, note: str = ""):
    return html.Div([html.Span(label, className="kpi-label"), html.Strong(value, className="kpi-value"), html.Span(note, className="kpi-note")], className="kpi")


def section_heading(number: str, eyebrow: str, title: str, detail: str):
    return html.Div([
        html.Div(number, className="section-number"),
        html.Div([html.P(eyebrow, className="eyebrow"), html.H2(title), html.P(detail, className="section-detail")]),
    ], className="section-heading")


def create_app(root: Path = ROOT) -> Dash:
    data = load_results(root)
    summary, metrics = data["summary"], data["metrics"]
    champion = winner_name(summary, metrics)
    threshold = recorded_threshold(summary)
    costs = summary.get("costs", {})
    fn_assumption = float(costs.get("false_negative", 5.0))
    fp_assumption = float(costs.get("false_positive", 1.0))
    test_row = pd.DataFrame()
    if not metrics.empty and champion:
        test_row = metrics[(metrics["model"] == champion) & (metrics["split"] == "test")]

    def metric(column: str) -> str:
        return f"{float(test_row.iloc[0][column]):.3f}" if len(test_row) and column in test_row else "—"

    def interval_note(column: str, fallback: str) -> str:
        interval = summary.get("bootstrap95", {}).get(column, {})
        if "low" in interval and "high" in interval:
            return f"95% bootstrap interval: {interval['low']:.3f}–{interval['high']:.3f}"
        return fallback

    app = Dash(__name__, assets_folder=str(Path(__file__).parent / "assets"), title="Credit Risk Lab | Behavioral risk intelligence")
    app.layout = html.Div([
        html.Header([
            html.A([html.Span("CR", className="brand-symbol"), html.Span("CREDIT RISK", className="brand-text"), html.Span("LAB", className="brand-light")], href="#", className="brand"),
            html.Nav([html.A("Model evidence", href="#evidence"), html.A("Decision policy", href="#policy"), html.A("Try a scenario", href="#scenario")], **{"aria-label": "Main navigation"}),
            html.Span("RESEARCH EDITION / 01", className="edition"),
        ], className="topbar"),
        html.Main([
            html.Section([
                html.Div([
                    html.Div([html.Span(className="live-dot"), "CLASSICAL MACHINE LEARNING · UCI 350"], className="hero-tag"),
                    html.H1(["Understand the risk.", html.Br(), html.Span("Make the trade-off visible.")]),
                    html.P("A transparent study of next-month credit card default. Follow the evidence from model comparison to the cost of a decision.", className="hero-description"),
                    html.Div([html.A("Explore the evidence", href="#evidence", className="button button-primary"), html.Span("Validation-selected. Test-evaluated.", className="hero-caption")], className="hero-actions"),
                ], className="hero-copy"),
                html.Div([
                    html.P("THE RESEARCH QUESTION", className="eyebrow light"),
                    html.H3("Which repayment patterns precede a missed payment?"),
                    html.Div([html.Span("01", className="flow-number"), html.Div([html.Strong("Observe"), html.P("Six months of billing and repayment history")])], className="flow-item"),
                    html.Div([html.Span("02", className="flow-number"), html.Div([html.Strong("Estimate"), html.P("A probability, evaluated for calibration")])], className="flow-item"),
                    html.Div([html.Span("03", className="flow-number"), html.Div([html.Strong("Evaluate"), html.P("Missed defaults, false alarms, and review workload")])], className="flow-item"),
                    html.Div("Historical Taiwan cohort · educational research", className="hero-panel-foot"),
                ], className="hero-panel"),
            ], className="hero"),
            html.Div([
                kpi(model_label(champion) if champion else "Not trained", "RECORDED CHAMPION", "Chosen using validation average precision"),
                kpi(metric("average_precision"), "TEST AVERAGE PRECISION", interval_note("average_precision", "Ranking performance under class imbalance")),
                kpi(metric("roc_auc"), "TEST ROC-AUC", interval_note("roc_auc", "Discrimination on unseen records")),
                kpi(metric("brier"), "TEST BRIER SCORE", interval_note("brier", "Probability error · lower is better")),
            ], className="kpi-grid"),
            html.Section([
                section_heading("01", "MODEL EVIDENCE", "A fair comparison, an honest result.", "The champion is chosen on validation data. Test results describe the frozen decision; they do not choose it."),
                html.Div([
                    html.Div([html.Div("MODEL SELECTION · BEFORE OPTIONAL CHAMPION CALIBRATION", className="card-label"), dcc.Graph(figure=comparison_figure(metrics, summary), config={"displayModeBar": False})], className="card"),
                    html.Div([
                        html.Div([html.Div("CHAMPION DIAGNOSTICS", className="card-label"), dcc.Dropdown(id="curve-kind", options=[{"label": "Precision–recall", "value": "pr"}, {"label": "ROC", "value": "roc"}, {"label": "Calibration", "value": "calibration"}], value="pr", clearable=False, searchable=False, className="compact-select")], className="card-top"),
                        dcc.Graph(id="curve-chart", figure=curve_figure(data["test"], champion, "pr"), config={"displayModeBar": False}),
                    ], className="card"),
                ], className="two-column"),
            ], id="evidence", className="page-section"),
            html.Section([
                section_heading("02", "DECISION POLICY", "Every threshold has a consequence.", "Explore the validation set only. The published test metrics above retain the threshold selected in the research workflow."),
                html.Div([
                    html.Div([
                        html.Div("VALIDATION WHAT-IF", className="card-label"),
                        html.Div([html.H3("Review threshold"), html.Strong(id="threshold-value", className="threshold-display")], className="threshold-heading"),
                        dcc.Slider(id="policy-threshold", min=0, max=1, step=0.01, value=round(threshold, 2), marks={0: "0.00", 0.25: "0.25", 0.5: "0.50", 0.75: "0.75", 1: "1.00"}, tooltip={"placement": "bottom", "always_visible": False}),
                        html.P("A record is flagged when its predicted probability is at or above this threshold.", className="small-note"),
                        html.Div([
                            html.Div([html.Label("Missed-default cost", htmlFor="fn-cost"), dcc.Input(id="fn-cost", type="number", min=0, max=max(100, fn_assumption), step=1, value=fn_assumption)]),
                            html.Div([html.Label("False-alarm cost", htmlFor="fp-cost"), dcc.Input(id="fp-cost", type="number", min=0, max=max(100, fp_assumption), step=1, value=fp_assumption)]),
                        ], className="input-grid"),
                        html.P("Hypothetical cost units, not money or measured savings. Reviewing a flagged account is not assumed to prevent default.", className="disclaimer"),
                    ], className="card policy-controls"),
                    html.Div([html.Div("OBSERVED VALIDATION WORKLOAD", className="card-label"), html.Div(id="policy-outcome"), dcc.Graph(id="policy-confusion", config={"displayModeBar": False})], className="card"),
                ], className="two-column policy-grid"),
            ], id="policy", className="page-section"),
            html.Section([
                section_heading("03", "BEHAVIORAL DRIVERS", "What changes a model's ranking?", "Permutation importance measures predictive reliance. It does not establish causation or a policy recommendation."),
                html.Div([
                    html.Div([dcc.Graph(figure=importance_figure(root), config={"displayModeBar": False})], className="card"),
                    html.Div([
                        html.P("READ THE RESULT IN CONTEXT", className="eyebrow"),
                        html.H3("Better prediction begins with better questions."),
                        html.P("Does repayment history add value over basic account attributes? Do engineered features improve the same model under the same split? Are probabilities reliable enough to interpret?"),
                        html.Div([html.Strong("Data leakage is a design problem."), html.P("Preprocessing, feature fitting, model tuning, and calibration belong inside the training boundary. Validation informs selection. Test remains an assessment.")], className="inset-note"),
                    ], className="insight-panel"),
                ], className="two-column"),
            ], className="page-section"),
            html.Section([
                section_heading("04", "SYNTHETIC SCENARIO", "Explore a hypothetical account.", "These controls construct an invented account. No customer records are loaded into this form."),
                html.Div([
                    html.Div([
                        html.Div("ILLUSTRATIVE INPUTS · NT DOLLARS", className="card-label"),
                        html.Div([
                            html.Div([html.Label("Credit limit", htmlFor="demo-limit"), dcc.Input(id="demo-limit", type="number", value=150000, min=10000, max=1000000, step=10000)]),
                            html.Div([html.Label("Monthly bill", htmlFor="demo-bill"), dcc.Input(id="demo-bill", type="number", value=50000, min=0, max=1500000, step=1000)]),
                            html.Div([html.Label("Monthly payment", htmlFor="demo-payment"), dcc.Input(id="demo-payment", type="number", value=5000, min=0, max=1000000, step=500)]),
                            html.Div([html.Label("Repayment status", htmlFor="demo-delay"), dcc.Dropdown(id="demo-delay", value=0, clearable=False, searchable=False, options=[{"label": label, "value": code} for code, label in [(-2, "−2 · source category (undocumented)"), (-1, "−1 · paid duly"), (0, "0 · source category (undocumented)"), (1, "1 · one-month delay"), (2, "2 · two-month delay"), (3, "3 · three-month delay"), (4, "4 · four-month delay"), (5, "5 · five-month delay"), (6, "6 · six-month delay"), (7, "7 · seven-month delay"), (8, "8 · eight-month delay")]])]),
                        ], className="input-grid"),
                        html.P("The selected bill, payment, and status are repeated across all six months. Education is fixed at university and marital status at married. These simplifications can create unrealistic combinations; use this only to understand the interface.", className="small-note"),
                        html.Button("Score synthetic scenario →", id="score-button", n_clicks=0, className="button button-primary"),
                    ], className="card scenario-controls"),
                    html.Div(id="score-result", children=[html.P("SYNTHETIC ACCOUNT", className="eyebrow"), html.H3("A probability is an estimate."), html.P("Try a scenario to see the frozen model's output and recorded research threshold. No automated lending decision is made.")], className="scenario-result"),
                ], className="two-column"),
            ], id="scenario", className="page-section"),
            html.Div([
                html.Div([html.P("RESEARCH BOUNDARIES", className="eyebrow"), html.H3("Clear limits are part of the evidence.")]),
                html.Ul([html.Li("Historical data from Taiwan. A random holdout does not demonstrate future or cross-country performance."), html.Li("Existing credit card accounts only. This study does not evaluate new-loan approval or intervention effectiveness."), html.Li("Subgroup evaluation is descriptive and does not establish legal compliance or the absence of bias.")]),
            ], className="limitations"),
        ], className="page-container"),
        html.Footer([html.Span("CREDIT RISK LAB"), html.Span("IBM data science workflow · Python / SQL / scikit-learn / XGBoost / Dash"), html.A("UCI dataset · CC BY 4.0", href="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients", target="_blank", rel="noreferrer")], className="footer"),
    ])

    @app.callback(Output("curve-chart", "figure"), Input("curve-kind", "value"))
    def update_curve(kind):
        return curve_figure(data["test"], champion, kind)

    @app.callback(
        Output("threshold-value", "children"), Output("policy-outcome", "children"), Output("policy-confusion", "figure"),
        Input("policy-threshold", "value"), Input("fn-cost", "value"), Input("fp-cost", "value"),
    )
    def update_policy(value, fn_cost, fp_cost):
        rows = data["validation"]
        if rows.empty or not champion:
            return f"{value:.2f}", html.P("Train the workflow to explore validation predictions."), empty_figure("No validation predictions available.")
        rows = rows[rows["model"] == champion]
        if rows.empty:
            return f"{value:.2f}", html.P("No champion validation predictions available."), empty_figure("No validation predictions available.")
        if fn_cost is None or fp_cost is None or not 0 <= fn_cost <= max(100, fn_assumption) or not 0 <= fp_cost <= max(100, fp_assumption):
            return f"{value:.2f}", html.P("Enter nonnegative costs within the displayed input ranges."), empty_figure("Valid cost assumptions are required.")
        result = threshold_metrics(rows["y_true"], rows["probability"], value, fn_cost, fp_cost)
        cards = html.Div([kpi(f"{result['review_rate']:.1%}", "FLAGGED FOR REVIEW"), kpi(f"{result['recall']:.1%}", "DEFAULT RECALL"), kpi(f"{result['precision']:.1%}", "REVIEW PRECISION"), kpi(f"{result['cost']:,.0f}", "HYPOTHETICAL COST UNITS")], className="policy-kpis")
        fig = styled_figure()
        fig.add_trace(go.Heatmap(z=[[result["tn"], result["fp"]], [result["fn"], result["tp"]]], x=["Not flagged", "Flagged"], y=["No default", "Default"], texttemplate="%{z:,}", colorscale=[[0, "#edf4f1"], [1, TEAL]], showscale=False, hovertemplate="Actual: %{y}<br>Decision: %{x}<br>Records: %{z:,}<extra></extra>"))
        fig.update_layout(height=235, margin=dict(l=85, r=20, t=15, b=40))
        fig.update_yaxes(autorange="reversed")
        return f"{value:.2f}", cards, fig

    @app.callback(
        Output("score-result", "children"), Input("score-button", "n_clicks"),
        State("demo-limit", "value"), State("demo-bill", "value"), State("demo-payment", "value"), State("demo-delay", "value"),
        prevent_initial_call=True,
    )
    def score_synthetic(n_clicks, limit, bill, payment, delay):
        artifact_path = root / "models" / "champion.joblib"
        if not artifact_path.exists():
            return [html.P("MODEL ARTIFACT REQUIRED", className="eyebrow"), html.H3("Run the training workflow first."), html.P("Evaluation charts remain available from the committed reports. Synthetic scoring requires the locally generated models/champion.joblib artifact.")]
        try:
            if any(v is None for v in (limit, bill, payment, delay)):
                raise ValueError("Complete all four scenario inputs.")
            profile = synthetic_profile(limit, bill, payment, delay)
            import joblib
            artifact = joblib.load(artifact_path)  # Trusted artifact generated locally by this repository.
            probability = float(artifact["pipeline"].predict_proba(profile)[0, 1])
            chosen = float(artifact.get("threshold", threshold))
            flagged = probability >= chosen
            return [
                html.P("SYNTHETIC SCENARIO · MODEL ESTIMATE", className="eyebrow"),
                html.Div(f"{probability:.1%}", className="score-probability"),
                html.H3("Flagged for review" if flagged else "Below the review threshold"),
                html.P(f"Recorded research threshold: {chosen:.3f}. Model: {artifact.get('model_name', champion)}."),
                html.P("This probability describes a hypothetical profile under a historical model. It is not a credit decision, a guarantee, or a personalized financial recommendation.", className="small-note"),
            ]
        except ValueError as exc:
            return [html.P("CHECK THE INPUTS", className="eyebrow"), html.H3(str(exc))]
        except (KeyError, ImportError, OSError, AttributeError) as exc:
            return [html.P("MODEL COULD NOT BE LOADED", className="eyebrow"), html.H3("Check the local environment."), html.P(f"Recreate the model using the pinned project dependencies. Details: {type(exc).__name__}.")]

    return app


if __name__ == "__main__":
    create_app().run(debug=False, host="127.0.0.1", port=8050)
