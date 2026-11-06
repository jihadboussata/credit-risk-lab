# Model artifacts

`python -m credit_risk.run --root .` writes `champion.joblib` here.
The downloadable project bundle may contain the verified trained artifact;
Git intentionally excludes executable pickle/joblib artifacts. A fresh clone
can explore the committed report and dashboard immediately; train locally to
enable synthetic-profile scoring.

Only load artifacts you trust. Joblib/pickle can execute Python on loading.
Use the pinned environment: serialized scikit-learn models are version-specific.
The saved pipeline includes feature engineering and preprocessing so inference
uses the same transformations as training.
