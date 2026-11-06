# Sources, attribution, and useful primary documentation

## Data

Yeh, I. (2009). **Default of Credit Card Clients** [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C55S3H](https://doi.org/10.24432/C55S3H).

- [Official dataset page and field descriptions](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- [Official downloadable archive](https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip)
- Data license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- The generated provenance records the archive used by this execution. The data license is separate from the repository's code license.
- Source data are normalized to lowercase field names, the outcome is renamed `target`, and deterministic derived features are added by the pipeline. These transformations do not imply endorsement by the dataset creator or UCI.

Original research: Yeh, I.-C., & Lien, C.-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications, 36(2), 2473–2480. [https://doi.org/10.1016/j.eswa.2007.12.020](https://doi.org/10.1016/j.eswa.2007.12.020).

The original paper is context, not the performance target of this project. Different partitions, feature decisions, and evaluation definitions prevent a direct claim of improvement over it.

## Methodology and learning context

- Rollins, J. B. (2015). *Foundational Methodology for Data Science*. IBM Analytics. [IBM white paper hosted by TDWI](https://tdwi.org/~/media/64511a895d86457e964174edc5c4c7b1).
- [IBM Data Science Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-science).

This is an independent portfolio implementation using the skills taught by the certificate. It is not an IBM product or an IBM-reviewed banking model.

## Implementation references

- [scikit-learn: preventing data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- [scikit-learn: cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn: logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [scikit-learn: precision–recall metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)
- [scikit-learn: permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [XGBoost Python documentation](https://xgboost.readthedocs.io/en/stable/python/)
- [Dash documentation](https://dash.plotly.com/)
- [SQLite documentation](https://www.sqlite.org/docs.html)

Consult the pinned dependency files for the exact library versions used to generate the included results. Online documentation may describe later releases.
