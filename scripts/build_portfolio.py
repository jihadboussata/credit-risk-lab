"""Build the human-readable portfolio from actual frozen study outputs.

No models are fitted and no policies are selected here. This script only reads
saved results, updates documentation and embeds existing figures in offline HTML.
"""
from __future__ import annotations
import base64
import html
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"dummy": "Prevalence baseline", "logistic_l1": "L1 logistic (lasso)",
          "logistic_l2": "L2 logistic (ridge)", "decision_tree": "Decision tree",
          "random_forest": "Random forest", "xgboost": "XGBoost", "svm_rbf": "RBF SVM"}


def markdown_table(frame):
    header = "| " + " | ".join(frame.columns) + " |\n"
    rule = "| " + " | ".join(["---"] * len(frame.columns)) + " |\n"
    return header + rule + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in frame.to_numpy())


def build(root=ROOT):
    reports = root / "reports"
    summary = json.loads((reports / "summary.json").read_text(encoding="utf-8"))
    all_metrics = pd.read_csv(reports / "metrics.csv")
    winner, threshold, metrics = summary["winner"], summary["threshold"], summary["test_metrics"]
    label = LABELS.get(winner, winner)
    fn_cost = summary["costs"]["false_negative"]
    fp_cost = summary["costs"]["false_positive"]
    ratio_label = f"{fn_cost:g}:{fp_cost:g}"
    counts = summary["split_counts"]
    selection = summary["model_selection"]["validation_average_precision"]
    test = all_metrics[(all_metrics.split == "test") & ~all_metrics.model.str.endswith("__policy")].copy()
    test["selection_ap"] = test.model.map(selection)
    test = test.sort_values("selection_ap", ascending=False, na_position="last")
    rows = []
    for _, row in test.iterrows():
        rows.append({"Model": LABELS.get(row.model, row.model),
                     "Selection val AP": "—" if pd.isna(row.selection_ap) else f"{row.selection_ap:.4f}",
                     "Test AP": f"{row.average_precision:.4f}", "Test ROC-AUC": f"{row.roc_auc:.4f}",
                     "Test Brier ↓": f"{row.brier:.4f}"})
    comparison = pd.DataFrame(rows)
    intervals = summary["bootstrap95"]
    ci_ap, ci_auc = intervals["average_precision"], intervals["roc_auc"]
    headline = (
        f"**Validation-selected model: {label}.** On {counts['test']:,} held-out accounts: "
        f"average precision **{metrics['average_precision']:.3f}**, "
        f"ROC-AUC **{metrics['roc_auc']:.3f}**, Brier score **{metrics['brier']:.3f}**. "
        f"At the frozen threshold **{threshold:.2f}**, recall is **{metrics['recall']:.1%}** "
        f"and **{metrics['review_rate']:.1%}** of accounts are flagged. "
        f"The threshold uses an illustrative missed-default:false-alarm cost ratio of {ratio_label}.\n\n"
        f"95% customer-bootstrap intervals: AP {ci_ap['low']:.3f}–{ci_ap['high']:.3f}; "
        f"ROC-AUC {ci_auc['low']:.3f}–{ci_auc['high']:.3f}. "
        "These intervals condition on the fitted model; they do not measure future drift or model-selection uncertainty."
    )
    readme = root / "README.md"
    readme.write_text(re.sub(r"<!-- RESULTS_START -->.*?<!-- RESULTS_END -->",
                            "<!-- RESULTS_START -->\n" + headline + "\n<!-- RESULTS_END -->",
                            readme.read_text(encoding="utf-8"), flags=re.S), encoding="utf-8")
    ablation = pd.read_csv(reports / "feature_ablation.csv")
    delta = float(ablation.loc[ablation.features == "original_plus_engineered", "mean_cv_average_precision"].iloc[0]
                  - ablation.loc[ablation.features == "original_only", "mean_cv_average_precision"].iloc[0])
    calibration = summary["calibration"]
    cost = summary["costs"]
    all_clear = float(cost["false_negative"] * summary["split_prevalence"]["test"] * 1000)
    all_flagged = float(cost["false_positive"] * (1 - summary["split_prevalence"]["test"]) * 1000)
    outcome = (
        f"At the frozen policy, the test confusion counts are TN={metrics['tn']:,}, "
        f"FP={metrics['fp']:,}, FN={metrics['fn']:,}, TP={metrics['tp']:,}. "
        f"Precision is {metrics['precision']:.1%}; recall is {metrics['recall']:.1%}. "
        f"The illustrative error cost is {metrics['cost_units_per_1000']:.1f} units per 1,000 accounts, "
        f"compared with {all_clear:.1f} for flagging none and {all_flagged:.1f} for flagging all. "
        "This comparison is a consequence of assumed classification costs. It is not measured money saved "
        "or evidence that reviewing an account prevents default."
    )
    budget = summary["review_budget"]
    capacity_note = (
        f"The cost-selected policy is not capacity constrained: it flags {metrics['review_rate']:.1%} "
        f"of test accounts. Under a separately prespecified top-{budget['review_fraction']:.0%} "
        f"ranking budget ({budget['reviewed']:,} accounts), precision is "
        f"{budget['precision_at_budget']:.1%} and recall is {budget['recall_at_budget']:.1%}. "
        "This is a ranking benchmark with stable tie-breaking, not an estimated intervention benefit."
    )
    results = f"""# Results and interpretation

{headline}

## What was selected, and when?

The six learned models were tuned using 3-fold stratified cross-validation on
{counts['train']:,} training accounts. Their tuned candidates were compared on {counts['validation']:,} validation
accounts using average precision. The initial winning family was **{label}**.
The selected family's optional calibration was decided using validation Brier
score, with calibration fitted on training folds. Recorded calibration: **{calibration['method']}**.
The policy threshold was then selected on validation data under a {ratio_label} cost ratio.
Model, calibration and threshold were fixed before the final {counts['test']:,}-account assessment.

## Complete model comparison

{markdown_table(comparison)}

Rows are ordered by **initial validation selection AP**, not final test score.
The champion's test row uses its final calibration; other model rows use their
tuned fitted candidate. Thus the selection AP and final calibrated AP can differ.
All test comparisons are descriptive. The project does not reselect a winner from this table.
AP is average precision, not accuracy or trapezoidal precision–recall area.
The baseline predicts training prevalence for every customer; its AP is test prevalence.

## Decision consequences

{outcome}

{capacity_note}

![Validation policy trade-offs](../reports/figures/threshold_tradeoffs.png)

## Did engineered features help?

{markdown_table(ablation[['features', 'mean_cv_average_precision', 'std_cv_average_precision']].round(4).astype(str))}

Engineered minus original training-CV AP: **{delta:+.4f}**. This is an exploratory,
fixed-parameter ablation: hyperparameters were tuned with engineered features.
It is not an independently tuned comparison or an unbiased confirmatory estimate.
A small or negative improvement is a useful result; the feature set was not rewritten after inspecting test scores.

## Explanations and subgroup diagnostics

Permutation importance measures how validation AP changes when a raw input is
shuffled and all derived features are recomputed. Correlated inputs can share or
mask importance. It does not identify causes of default. `subgroup_metrics.csv`
reports sample sizes, class prevalence, recall and false-positive rates by sex
and age. Small groups are unstable; no fairness or regulatory certification is claimed.

## What the results do not establish

No future-period or external-bank validation is available in this dataset.
Bootstrap intervals are conditional on this fitted model and historical cohort.
Validation has supported several decisions, so its scores can be optimistic.
No customer intervention, recovery or realized economic benefit is observed.
Before an operational pilot: secure suitable contemporary data, conduct external
and temporal validation, review feature and fairness policies with domain experts,
and test intervention effects prospectively.

Study mode: **{summary['run_mode']}**. Seed: **{summary['seed']}**.
Full environment and archive fingerprint: `reports/summary.json` and `reports/provenance.json`.
"""
    (root / "docs/results.md").write_text(results, encoding="utf-8")
    model_card = f"""# Model card — Credit Risk Lab

| Field | Description |
|---|---|
| Artifact | `models/champion.joblib`, generated by the reproducible pipeline |
| Model | {label}; calibration: {calibration['method']} |
| Target | Source next-month default indicator, 1=yes, 0=no |
| Intended use | Educational research and account-review scenario exploration |
| Unsupported uses | Operational lending, approvals/denials, credit limits, automated treatment |
| Source | UCI 350, Taiwan 2005; CC BY 4.0 |
| Training / validation / test | {counts['train']:,} / {counts['validation']:,} / {counts['test']:,} distinct source account IDs; stratified seed{summary['seed']} |
| Selection | Training-only 3-fold tuning; validation AP; test assessment after freeze |
| Policy | Flag if probability ≥ {threshold:.2f}; illustrative FN cost{fn_cost:g}, FP cost{fp_cost:g} |
| Input | Canonical lowercase account and historical financial columns, NTD amounts |
| Excluded | Customer ID, outcome, sex, age; education/marriage remain possible proxies |
| Output | Research probability of the recorded target; not a guarantee or causal estimate |
| Monitoring status | Historical prototype; no production, live feedback or observed monitoring |

## Measured performance

{headline}

{outcome}

{capacity_note}

## Known limitations

Historical, single-country cohort; random holdout cannot establish contemporary,
out-of-time or external-bank generalization. Source repayment codes include
ambiguities preserved as categorical values. Negative bills are retained; undefined
ratios are imputed from training medians. Distinct IDs with repeated predictor
profiles are retained because similar values do not establish duplicate people.

Age and sex are excluded as direct model features. This does not remove proxies,
selection bias or historical inequity. Audit results are descriptive with varying
sample sizes. No conclusion of fairness follows from omitting attributes.

## Reproduction and artifact handling

Install the pinned dependencies and run `python -m credit_risk.run --root .`.
The saved artifact contains preprocessing, fitted estimator, threshold, source
fingerprint and environment metadata. Load only trusted local joblib files;
pickle-based artifacts can execute code. Version compatibility is not guaranteed
outside the documented environment. A fresh clone trains locally; binaries are
excluded from Git.

## Proposed feedback and monitoring

See `feedback.md` for a proposed review protocol. Track input validity, missingness,
distribution shift, flag rate, delayed-label AP and calibration, and subgroup
diagnostics. Establish alert/retraining criteria on suitable contemporary data.
These are plans, not claims of completed monitoring or stakeholder approval.
"""
    (root / "docs/model_card.md").write_text(model_card, encoding="utf-8")
    cards = [("HELD-OUT AVERAGE PRECISION", f"{metrics['average_precision']:.3f}", "Ranking under class imbalance"),
             ("HELD-OUT ROC–AUC", f"{metrics['roc_auc']:.3f}", "Discrimination across thresholds"),
             ("DEFAULT RECALL", f"{metrics['recall']:.1%}", f"At frozen threshold {threshold:.2f}"),
             ("ACCOUNTS FLAGGED", f"{metrics['review_rate']:.1%}", "Illustrative review workload")]
    card_html = ''.join(f'<article class="kpi"><small>{a}</small><strong>{b}</strong><span>{c}</span></article>' for a,b,c in cards)
    captions = {
        "training_eda": ("01 / Understand the cohort", "Training-only descriptive associations; credit amounts are in NTD."),
        "model_comparison": ("02 / Compare the evidence", "Final test comparison is descriptive. The champion was selected on validation data."),
        "roc_pr_curves": ("03 / Evaluate beyond accuracy", "ROC and precision–recall expose different aspects of ranking behavior."),
        "calibration": ("04 / Check probability quality", "Reliability estimates use quantile bins. Small bins and shifted cohorts can be misleading."),
        "threshold_tradeoffs": ("05 / Make the trade-off explicit", "Validation-only threshold exploration. Error costs are illustrative units."),
        "confusion_matrix": ("06 / Account for every decision", "Final holdout confusion counts at the frozen validation-selected threshold."),
        "feature_importance": ("07 / Explain model behavior", "Validation permutation importance describes association, not causation."),
        "subgroup_audit": ("08 / Audit who the model serves", "Sample sizes and error rates vary by subgroup. This is not a fairness certification."),
    }
    figure_html = ''
    for name,(title,caption) in captions.items():
        path = reports / "figures" / f"{name}.png"
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode()
            figure_html += f'<section class="panel"><p class="eyebrow">{html.escape(title)}</p><img alt="{html.escape(title)}" src="data:image/png;base64,{encoded}"><p class="caption">{html.escape(caption)}</p></section>'
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Credit Risk Lab | Research report</title>
<style> :root{{--navy:#102c40;--teal:#008879;--cream:#f4f3ed;--muted:#62737c}}*{{box-sizing:border-box}}body{{margin:0;background:var(--cream);color:var(--navy);font:16px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}}.wrap{{max-width:1200px;margin:auto;padding:0 32px}}header{{background:var(--navy);color:white;padding:26px 0 66px}}nav{{display:flex;justify-content:space-between;border-bottom:1px solid #ffffff25;padding-bottom:22px;font-size:13px;letter-spacing:2px}}.eyebrow{{font-size:12px;letter-spacing:2px;font-weight:700;color:var(--teal);text-transform:uppercase}}header .eyebrow{{color:#78d7c7;margin-top:48px}}h1{{font-size:clamp(42px,7vw,76px);line-height:1.08;letter-spacing:-3px;margin:18px 0}}.lead{{max-width:780px;color:#c9dae2;font-size:21px}}.tag{{border:1px solid #ffffff33;border-radius:30px;padding:6px 14px;display:inline-block;margin:12px 8px 0 0;font-size:12px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:30px 0}}.kpi{{background:white;border-top:4px solid var(--teal);padding:22px;border-radius:8px}}.kpi small{{font-size:10px;letter-spacing:1px;display:block}}.kpi strong{{font-size:42px;display:block;letter-spacing:-2px}}.kpi span{{font-size:12px;color:var(--muted)}}.panel{{background:white;padding:28px;margin:24px 0;border:1px solid #e2e7e2;border-radius:10px}}.panel img{{width:100%;height:auto;display:block;max-width:1000px;margin:auto}}.caption{{color:var(--muted);font-size:13px}}h2{{font-size:28px;letter-spacing:-.7px;line-height:1.25}}.callout{{border-left:4px solid var(--teal);background:#e8f2ee;padding:20px 24px;margin:20px 0}}table{{width:100%;border-collapse:collapse;font-size:14px}}th{{text-align:left;background:#edf3f0}}td,th{{padding:12px;border-bottom:1px solid #e1e7e3}}.scroll{{overflow:auto}}footer{{color:var(--muted);padding:40px 0;font-size:13px}}a{{color:var(--teal)}}@media(max-width:760px){{.wrap{{padding:0 18px}}.grid{{grid-template-columns:repeat(2,1fr)}}.panel{{padding:18px}}nav{{letter-spacing:.5px}}.kpi{{padding:14px}}.kpi strong{{font-size:34px}}}}@media print{{header{{background:white;color:#102c40}}.lead{{color:#62737c}}.panel{{break-inside:avoid}}}}</style></head><body>
<header><div class="wrap"><nav><b>CREDIT RISK LAB</b><span>REPRODUCIBLE RESEARCH / 01</span></nav><p class="eyebrow">Applied data science · Classical machine learning</p><h1>From payment history<br>to defensible decisions.</h1><p class="lead">A 30,000-account study connecting SQL, behavioral features, model evaluation and the practical cost of a missed default.</p><span class="tag">10-stage methodology</span><span class="tag">Six classifiers + baseline</span><span class="tag">Frozen holdout evaluation</span></div></header>
<main class="wrap"><div class="grid">{card_html}</div><section class="panel"><p class="eyebrow">The research decision</p><h2>Which existing accounts should a bank review?</h2><p>Six months of recorded bills, payments and repayment status are used to predict the source dataset's next-month default indicator. This is an educational prioritization study, not an approval system or a deployed banking product.</p><div class="callout"><b>Validation-selected champion: {html.escape(label)}.</b> Calibration: {html.escape(calibration['method'])}. Policy threshold: {threshold:.2f}, under an illustrative 5:1 missed-default:false-alarm cost ratio.</div><p>95% bootstrap intervals: average precision {ci_ap['low']:.3f}–{ci_ap['high']:.3f}; ROC–AUC {ci_auc['low']:.3f}–{ci_auc['high']:.3f}. Intervals condition on the fitted model and do not include selection uncertainty or future drift.</p></section>
<section class="panel"><p class="eyebrow">Selection discipline</p><h2>18,000 train. 6,000 validate. 6,000 assess.</h2><p>All model tuning uses three stratified folds within training. Validation selects the model family, optional probability calibration and policy. The test set is assessed after those decisions are frozen. Age, sex, customer ID and target are excluded from primary features.</p><div class="scroll">{comparison.to_html(index=False,border=0)}</div><p class="caption">Ordered by initial validation selection AP. Test rows describe the final candidates; the champion row includes its chosen calibration. Brier is better when lower; AP and ROC–AUC are better when higher.</p></section>
{figure_html}<section class="panel"><p class="eyebrow">Decision accounting</p><h2>The consequence of the chosen threshold</h2><p>{html.escape(outcome)}</p><p>Feature-engineering ablation: engineered minus original training-CV average precision <b>{delta:+.4f}</b>. This exploratory comparison holds hyperparameters fixed and does not independently tune each feature set.</p></section>
<section class="panel"><p class="eyebrow">Limits and next evidence</p><h2>A historical prototype with explicit boundaries.</h2><p>The source is Taiwan in 2005. Random holdout results do not establish performance in a future period or another bank. Missing direct protected attributes does not remove proxies. No intervention effects, measured financial savings, live monitoring or stakeholder approval are claimed.</p><p>Next evidence would require contemporary data, temporal and external validation, domain review and a prospective test of whether review actions help customers and reduce harm.</p><p><a href="../docs/methodology.md">Ten-stage methodology</a> · <a href="../docs/model_card.md">Model card</a> · <a href="../docs/results.md">Detailed results</a></p></section></main><footer class="wrap">Source: I-Cheng Yeh, Default of Credit Card Clients, UCI ML Repository, DOI 10.24432/C55S3H · CC BY 4.0. Code: MIT.<br>Study mode: {summary['run_mode']} · Seed {summary['seed']} · Actual experiment outputs, no illustrative performance data.</footer></body></html>'''
    page = page.replace("illustrative 5:1 missed-default", f"illustrative {ratio_label} missed-default")
    page = page.replace("<h2>The consequence of the chosen threshold</h2>",
                        "<h2>The consequence of the chosen threshold</h2><div class='callout'>" + html.escape(capacity_note) + "</div>")
    (reports / "portfolio_report.html").write_text(page, encoding="utf-8")
    print("Updated README results, results report, model card and standalone HTML.")


if __name__ == "__main__":
    build()
