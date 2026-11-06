# Feedback and monitoring plan

**Status:** this repository is a retrospective portfolio prototype. No production pilot, bank approval, or stakeholder feedback is claimed.

## A practical usability review

Ask a reviewer to complete the following tasks using a named dashboard version:

1. Identify the target population and prediction horizon.
2. Explain the champion's ranking performance and one limitation.
3. Compare a lower and higher threshold using the validation laboratory.
4. Explain the difference between a predicted probability and an intervention benefit.
5. Score a fictional account and find the assumptions behind the output.

Record task completion, confusing labels, incorrect interpretations, and suggested changes. A reviewer may be a fellow student for usability feedback; describe their role accurately. Domain validation needs an appropriately qualified reviewer and is a separate activity.

## Feedback log

Create one row per actual review. Leave the table empty until reviews happen.

| Date | Version | Reviewer role | Task or observation | Proposed change | Decision and rationale | Follow-up |
|---|---|---|---|---|---|---|

## Proposed acceptance criteria for a future pilot

These criteria must be agreed with real stakeholders and tested on external data before deployment:

- Inputs have validated meanings and a known pre-outcome timestamp.
- Review volume is feasible and error costs are based on the intended workflow.
- Probability calibration and subgroup errors are acceptable on a new cohort.
- A human can inspect and contest the output; no autonomous adverse decision is made by this prototype.
- Operational ownership, rollback, logging, and data-access responsibilities are defined.

The project intentionally does not assert that these criteria have been met.

## Monitoring if new data become available

| When | Signal | Investigation |
|---|---|---|
| At ingestion | Invalid columns, codes, ranges, missingness, duplicates | Reject invalid schema; inspect upstream data changes |
| At scoring | Score distribution and predicted positive rate | Compare with reference cohort and operational workload |
| After the outcome window | Average precision, Brier score, reliability bins | Check whether ranking and probabilities still transfer |
| After the outcome window | Recall/precision by available groups with counts | Investigate shifts and unequal error patterns |
| At a version change | Feature definitions, dependencies, fitted transformations | Run regression checks and verify stored metadata |

Set numeric alert thresholds using external pilot data and domain input; do not invent them from the existing test set. Drift can justify investigation without proving performance degradation. Performance degradation can also occur without an obvious univariate drift alarm.

## Iteration discipline

If feedback motivates a change, log the hypothesis, use training/validation evidence, and version the result. The existing test results may explain what happened in this study, but repeated consultation turns that partition into development data. Obtain new held-out data for a fresh final claim.
