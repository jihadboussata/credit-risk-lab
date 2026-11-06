"""Refresh static figures from the existing study, without fitting any models."""
import json
from pathlib import Path
import pandas as pd
from credit_risk.plots import make_figures

root = Path(__file__).resolve().parents[1]
reports = root / "reports"
summary = json.loads((reports / "summary.json").read_text(encoding="utf-8"))
raw = pd.read_csv(root / "data/processed/credit.csv")
manifest = pd.read_csv(reports / "split_manifest.csv")
train = raw.merge(manifest, on="id", validate="one_to_one").query("split == 'train'")
make_figures(root, train,
             pd.read_csv(reports / "metrics.csv", float_precision="round_trip"),
             pd.read_csv(reports / "test_predictions.csv", float_precision="round_trip"),
             summary["winner"], summary["threshold"],
             pd.read_csv(reports / "threshold_analysis.csv", float_precision="round_trip"),
             pd.read_csv(reports / "permutation_importance.csv"),
             pd.read_csv(reports / "subgroup_metrics.csv"))
print("Refreshed eight figures using saved study outputs; no model refit.")
