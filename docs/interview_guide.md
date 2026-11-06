# Explain the project with confidence

Use this as a study guide, not a script to memorize. Before including the repository on an application, rerun the notebooks, reproduce one model comparison, and explain the limitations without reading the README.

## A 60-second walkthrough

1. **Problem:** “I built a decision-support prototype that ranks existing credit-card accounts for review using six months of recorded behavior.”
2. **Data:** “It uses a public historical cohort from Taiwan. That makes it reproducible, but it does not establish current banking performance.”
3. **Method:** “I separated training, validation, and test data; fit transformations inside cross-validation; compared regularized logistic models, trees, ensembles, and SVM; and selected the champion using validation average precision.”
4. **Decision:** “I selected an illustrative review threshold on validation data under explicit relative error costs. The dashboard makes the trade-off visible.”
5. **Evidence:** “The report contains measured ranking, calibration, threshold, uncertainty, and subgroup results, including the weaker cases.”

Replace this high-level walkthrough with your own explanation of the actual results in `docs/results.md`. Do not recite a model name or metric until you have checked the generated artifacts.

## Questions you should be able to answer

### Why not optimize accuracy?

If one class is common, a classifier that always predicts it can appear accurate while missing the outcome of interest. Explain what the prevalence baseline achieves, why average precision is useful for ranking, and why a high ROC-AUC alone says little about workload at a specific threshold.

**Practice:** change the dashboard threshold and explain why recall increases while precision or review volume may change. Compare the default threshold with 0.5 using the validation predictions.

### Why separate validation from test?

Training data fit model parameters. Cross-validation inside training helps select hyperparameters. Validation chooses among candidate models, calibration policies, and thresholds. The test partition assesses the frozen workflow. Repeatedly choosing features or thresholds after looking at test outcomes makes the test set part of development.

**Practice:** point to the exact code that creates partitions and the manifest that stores their IDs. Explain why learned scaling must be refitted within each training fold.

### What are ridge and lasso doing here?

The models are logistic classifiers with L2 and L1 penalties. Regularization controls coefficients to reduce sensitivity to noise or correlated predictors. L1 can set coefficients to zero; L2 shrinks them. Neither automatically identifies causal drivers. The `C` parameter is inverse regularization strength: smaller values mean stronger regularization.

**Practice:** inspect the parameter grid, identify the chosen penalty and `C`, and discuss why correlated bills and payments can affect coefficient interpretation.

### What does SVM add?

An RBF SVM models nonlinear decision boundaries using distances in feature space. Scaling is therefore consequential. Its raw decision function is not a probability. Explain the training-only calibration step and why calibration quality is evaluated separately from discrimination.

**Practice:** describe `C` and `gamma`, then identify where calibration occurs in the implementation. Do not call a margin a default probability.

### Random forest versus XGBoost?

A random forest averages randomized trees, while gradient boosting builds successive trees to improve an objective. Both can learn nonlinear interactions, but complexity still needs control. Explain the parameters actually searched and the runtime/performance trade-off observed in this run.

### What does calibration tell you?

A ranking model can order accounts well while assigning probabilities that are too high or too low. Reliability plots and Brier score examine a different property from ranking. Calibration chosen on one historical validation sample may not hold after the population changes.

**Practice:** compare a predicted risk band with its observed event rate and discuss uncertainty in sparsely populated bins.

### Do the engineered features help?

The ablation compares the same modeling approach with and without behavioral summaries. It is evidence about incremental predictive value under a defined protocol. It cannot show that changing a feature would change a customer's outcome.

**Practice:** identify one safe-ratio denominator, explain what happens when it is zero, and verify that no outcome enters the calculation.

### Is the model fair because age and sex are excluded?

No. Other variables can be proxies, sample composition can differ, and labels reflect existing systems. Subgroup error rates are descriptive checks with sample-size limitations. A credible deployment would require domain, legal, and stakeholder review in its intended jurisdiction.

### What does “five cost units” mean?

It is a scenario assumption for comparing false negatives and false positives. It is not NT$, a measured loss, or a proven saving. Explain how the threshold changes when the assumption changes. A review intervention has its own cost, benefit, and possible harms that this observational dataset does not measure.

### How reproducible is the result?

Identify the source hash, dependency versions, seed, partitions, parameter grid, chosen pipeline, and reproduction command. Explain that bootstrap intervals reflect sampling variability under the cohort assumptions, not every uncertainty in model development or future deployment.

### What would you do next?

Prioritize a new period or institution for external evaluation, stronger data definitions, deployment-specific costs, and real analyst feedback. More algorithms are useful only if they answer an unresolved question.

## Honest contribution and AI assistance

This repository was developed with AI assistance. Before submitting it, review the code and results, verify the commands yourself, and make changes you understand. Describe your actual contribution accurately. A defensible statement is “I developed and studied this AI-assisted portfolio project, reproduced the analysis, and can explain its design and limitations” only after you have done those things.

Do not imply that you collected bank customer data, collaborated with a bank, deployed a production model, prevented defaults, or measured financial savings. Do not invent stakeholder feedback, authorship history, awards, or employment experience.

## Personalize before applying

- Write a short paragraph in your own words about the question that interested you.
- Rerun the project and record the actual metrics from your own environment.
- Change one justified assumption or feature, compare it using validation data, and document the result.
- Capture your own dashboard screenshot and a brief walkthrough if useful.
- Link the repository from your CV using its final public URL.
