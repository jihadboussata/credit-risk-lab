# A complete, decision-led data science workflow

This project implements the ten stages of IBM's Foundational Methodology for Data Science. The stages are iterative: an implausible result should trigger a return to data understanding, not a more elaborate model.

## 1. Business Understanding

**Decision:** prioritize an existing credit-card portfolio for account review. **Prediction:** the probability that an account has the dataset's next-month default outcome. **User:** an analyst reviewing a finite number of accounts.

The analysis does not approve applications, set credit limits, or recommend denying customers services. It evaluates predictive evidence for a portfolio-review demonstration. It does not measure whether contacting a customer prevents default.

The operational trade-off is explicit: missed defaults versus unnecessary reviews. A missed default has an assumed cost of five units and a false alarm one unit in the default scenario. These are illustrative relative costs, not estimated financial losses. A sensitivity analysis should precede any business use.

## 2. Analytic Approach

Use supervised binary classification to estimate risk, supported by descriptive SQL and statistical analysis. Compare a prevalence baseline with L1-regularized logistic regression, L2-regularized logistic regression, a decision tree, random forest, XGBoost, and an RBF support vector machine.

Here, “lasso” and “ridge” refer to L1 and L2 penalties **on logistic regression**. Continuous-outcome `Lasso` and `Ridge` regressors are not substitutes for probability-producing classifiers. Model complexity is justified by measured validation performance and the intended use.

Average precision is the primary selection metric because the positive outcome is less common and the intended workflow ranks accounts. ROC-AUC measures another aspect of ranking; Brier score and reliability diagrams assess the quality of probabilities. Threshold-specific precision, recall, F1, and the confusion matrix translate scores into an illustrative review policy.

## 3. Data Requirements

Require one account row, six historical monthly repayment observations, next-month outcome, and a stable identifier. Preserve the original repayment codes, monetary signs, and field ordering. Document which variables exist before the prediction time.

The target and identifier cannot enter predictors. Sex and age are retained for auditing but excluded from the primary model. Education and marital-status categories remain available predictors and can act as demographic proxies; excluding two attributes does not prove fairness.

The dataset provides an account cohort, not repeated calendar cohorts suitable for a genuine chronological backtest. The split therefore supports generalization within this historical cohort only.

## 4. Data Collection

Download from UCI's official archive using the package command. Record the source URL, audit time, archive hash, and package versions in generated provenance. Audit time records the current verification, including when a cached archive is reused; it is not necessarily the original download time. Store source material separately from analytical outputs. Use the source license and citation in [references.md](references.md).

The downloader and parser make collection repeatable. A checksum identifies the exact file used; it is not evidence that upstream records are error-free. The generated SQLite tables support relational analysis without replacing the original source.

## 5. Data Understanding

Validate expected row counts, column names, unique account IDs, numeric types, binary targets, repayment-code values, and missingness. Report duplicate predictor profiles separately from duplicate IDs. Identical profiles can belong to different accounts; their presence should be inspected and disclosed rather than silently removed.

Use SQL to examine cohort composition and six-month payment trajectories. Conduct outcome-aware exploratory analysis only on the training partition. Reserve validation for model and policy choices, and test for the final frozen assessment. The notebooks follow the saved split manifest to avoid accidentally analyzing test outcomes while designing features.

Check surprising values against the dictionary before changing them. Negative bills can be credit balances; positive values above a credit limit can be meaningful. Repayment codes -2 and 0 lack the same explicit definition as the published delay scale. This project preserves them as categories and documents that ambiguity.

## 6. Data Preparation

Use a deterministic stratified **60% training / 20% validation / 20% test** partition with seed 42. Save account IDs and assignments so later analyses reproduce the exact split. The training sample is the only input to learned preprocessing, hyperparameter search, and cross-validated calibration.

Place feature engineering and preprocessing inside each scikit-learn pipeline. Encode categorical fields, including repayment-status categories, and standardize continuous features for algorithms that depend on scale. Preserve defensible monetary information. Handle division by zero explicitly; do not replace infinity after it has contaminated a model.

Engineered features summarize observed behavior: delayed-month counts, payment and bill summaries, and balance/payment ratios. All inputs predate the target. Features computed from the outcome or future observations are prohibited. A raw-feature comparison tests whether engineered summaries add measurable value.

## 7. Modeling

Tune compact, declared hyperparameter grids with three-fold stratified cross-validation on training data. Use identical fold assignments and scoring definitions across candidates. Save cross-validation results, parameter choices, elapsed times, and random seeds.

Select the candidate using validation average precision. Then assess a training-only cross-validated sigmoid calibration of that candidate, retaining calibration only when the declared validation Brier criterion improves. Select the review threshold on validation data using the stated cost scenario. Freeze the chosen model, probability transformation, and threshold before accessing test outcomes.

The validation set serves several choices, so its estimates can be optimistic. The untouched test partition is essential. This is a compact portfolio study, not an unbiased comparison from a nested cross-validation research design.

## 8. Evaluation

Publish validation and test tables, distinguishing **selection evidence** from **final descriptive evidence**. Showing every candidate's test metrics does not authorize choosing a different champion after seeing those results. If a test result motivates changes, a new external holdout is needed for a fresh unbiased final claim.

Include average precision, ROC-AUC, Brier score, precision, recall, F1, class prevalence, the confusion matrix, and threshold sensitivity. Report bootstrap intervals for the frozen champion. Resampling accounts quantifies sampling variation within this cohort; it does not cover future-population shifts or model-selection uncertainty.

Reload saved prediction CSVs with pandas `float_precision="round_trip"` to reconstruct the written floating-point probabilities faithfully. Ranking metrics can change slightly if a default CSV parser merges nearly equal scores into ties. The notebook, application, and artifact checks therefore use round-trip parsing instead of relaxing numerical consistency checks.

Audit performance by available sex and age groups using sample sizes and outcome counts alongside point estimates. Small groups can have unstable metrics. Differences are investigation signals, not proof of causal discrimination or regulatory compliance. Inspect validation permutation importance as model behavior, not causal explanation.

## 9. Deployment

Package the fitted feature pipeline and champion together so the Dash application uses exactly the training transformations. The local dashboard provides model comparison, probability diagnostics, a **validation-only** threshold laboratory, and synthetic account scoring.

The inference example uses fictional inputs, labels the model's probability and review threshold clearly, and displays the historical-data limitations. No real customer records are needed to demonstrate the application. A locally runnable prototype is the deployment deliverable; production banking deployment is outside the evidence in this repository.

## 10. Feedback

Define a realistic review process in [feedback.md](feedback.md). Collect actual feedback on usability, explanations, review workload, and failure cases. Record who reviewed what, which version was shown, and the resulting decisions. No stakeholder approval or production impact is fabricated.

For a future live pilot, monitor input validity, missingness, score distributions, review volume, and—after outcomes mature—ranking and calibration. Require new-data validation before claiming success in a different period or institution. Feedback can improve the product, but a repeatedly consulted test set cannot remain an untouched holdout.

## Deliverable map

| Evidence | Where to inspect it |
|---|---|
| Collection, audit, SQL | `notebooks/01_data_and_sql.ipynb`, `sql/`, generated provenance |
| Training-only EDA and statistical reasoning | `notebooks/02_exploration_and_hypotheses.ipynb` |
| Search, comparisons, calibration and decisions | `notebooks/03_models_and_decisions.ipynb`, `reports/` |
| Reusable implementation | `src/credit_risk/` |
| Decision-support prototype | `app/` |
| Evidence boundaries and communication | `docs/model_card.md`, `docs/results.md`, `docs/feedback.md` |

**Methodological source:** John B. Rollins, *Foundational Methodology for Data Science*, IBM Analytics (2015), [IBM white paper hosted by TDWI](https://tdwi.org/~/media/64511a895d86457e964174edc5c4c7b1).
