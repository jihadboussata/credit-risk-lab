# Publish the project to GitHub

Publish from the **project folder**, not the parent folder containing the ZIP or other chat outputs. GitHub should show `README.md`, `src/`, `notebooks/`, `app/`, `docs/`, and `reports/` at the repository root.

## 1. Review and personalize

Read the README and the generated results, rerun the tests, and inspect the dashboard. Use [interview_guide.md](interview_guide.md) to check your understanding. Add a truthful short introduction in your own words. Verify that no private files, credentials, virtual environments, or source account records are staged.

The repository's `.gitignore` separates reproducible source code and aggregate evidence from local source data and fitted binaries. Check `git status` before every commit. Data attribution remains required even when the raw archive is not published.

## 2. Create an empty repository

On GitHub, create a new repository named **credit-risk-lab** (or choose your own name). A public repository is easy for recruiters to review. Start with an empty repository: do not add a separate README, license, or `.gitignore` on GitHub if those files already exist locally.

Suggested description:

> End-to-end credit risk analysis with SQL, regularized logistic regression, SVM, random forest, XGBoost, calibration, and an interactive Dash decision-support prototype.

Suggested topics: `data-science`, `credit-risk`, `python`, `sql`, `scikit-learn`, `xgboost`, `classification`, `dash`, `portfolio-project`.

## 3. Commit locally and connect the remote

Run the following commands in a terminal opened in the project folder. If the delivered folder already has a Git repository, skip `git init`. Git may ask you to configure your own commit name and email; use your real preferred Git identity.

```shell
git init -b main
git add .
git status
git commit -m "Build reproducible credit risk portfolio study"
```

Copy the repository URL from GitHub and use it as the argument to `git remote add origin`. Then push:

```shell
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

`YOUR_REPOSITORY_URL` is an instruction placeholder in this document, not a command to paste unchanged. Use GitHub's normal authentication flow; never paste access tokens into source files or notebooks. If a remote already exists, inspect it with `git remote -v` instead of adding another one blindly.

## 4. Verify the published view

- Open the README and follow its internal links.
- Confirm notebook outputs render, charts load, and results match the generated report.
- Check the Actions tab for the automated checks.
- Confirm large raw datasets, local environments, and unnecessary model binaries are absent.
- Add the repository description and topics, then pin it to your profile if appropriate.
- Run the project from a fresh clone when possible.

The supplied code and results are ready for review; the publishing instructions do not claim that a remote repository has already been created.

## 5. Use it in an application

Describe the work using measured evidence from your own run. For example, explain that you compared classification approaches, built SQL analyses, separated model selection from final evaluation, and created a Dash prototype. Add a metric only after checking `docs/results.md` and its evaluation context.

Avoid “production-ready bank model,” “prevented defaults,” or a currency saving. Those claims require evidence beyond this dataset. A clear, reproducible repository and a confident explanation are more useful than overstated impact.
