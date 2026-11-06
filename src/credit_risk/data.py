"""Download, audit and normalize UCI dataset 350; materialize a relational view."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .features import STATUS, BILLS, PAYMENTS

SOURCE_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
DATASET_URL = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def download(root: Path, skip_download=False):
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    destination = raw / "uci_credit_card.zip"
    if not destination.exists():
        if skip_download:
            raise FileNotFoundError(f"{destination} missing; run without --skip-download.")
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "CreditRiskLab/1.0 educational research"})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
        # Validate before writing to avoid caching an HTML error page.
        import io
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if not any(name.lower().endswith(".xls") for name in archive.namelist()):
                raise ValueError("Downloaded archive contains no XLS dataset.")
        destination.write_bytes(payload)
    provenance = {
        "source": DATASET_URL, "download_url": SOURCE_URL,
        "local_archive": str(destination.relative_to(root)),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "bytes": destination.stat().st_size,
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY 4.0 (source dataset); see DATA_LICENSE.md for attribution",
    }
    return destination, provenance


def load_dataset(archive_path: Path):
    with zipfile.ZipFile(archive_path) as archive:
        name = next(name for name in archive.namelist() if name.lower().endswith(".xls"))
        with archive.open(name) as handle:
            frame = pd.read_excel(handle, header=1, engine="xlrd")
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    frame = frame.rename(columns={"default payment next month": "target"})
    required = ["id", "limit_bal", "sex", "education", "marriage", "age", *STATUS, *BILLS, *PAYMENTS, "target"]
    if set(frame.columns) != set(required):
        raise ValueError(f"Unexpected source schema: {list(frame.columns)}")
    frame = frame[required].apply(pd.to_numeric, errors="raise")
    if len(frame) != 30000 or not frame["id"].is_unique:
        raise ValueError("Expected exactly 30,000 unique customer records.")
    if frame.isna().any().any() or not set(frame.target.unique()).issubset({0, 1}):
        raise ValueError("Source contains unexpected missing values or invalid labels.")
    audit = {
        "rows": int(len(frame)), "columns_including_id_and_target": int(frame.shape[1]),
        "unique_ids": int(frame.id.nunique()), "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_ids": int(frame.id.duplicated().sum()),
        "duplicate_predictor_rows": int(frame.drop(columns=["id", "target"]).duplicated().sum()),
        "positive_class_count": int(frame.target.sum()), "prevalence": float(frame.target.mean()),
        "negative_bill_cells": int(frame[BILLS].lt(0).sum().sum()),
        "nonstandard_education_count": int((~frame.education.isin([1, 2, 3, 4])).sum()),
        "education_codes": sorted(int(v) for v in frame.education.unique()),
        "marriage_codes": sorted(int(v) for v in frame.marriage.unique()),
        "repayment_status_codes": sorted(int(v) for v in pd.unique(frame[STATUS].values.ravel())),
        "excluded_primary_features": ["id", "sex", "age", "target"],
        "cohort": "Taiwan credit-card accounts, April–September 2005 predictors; next-month default target",
        "audit_notes": [
            "Negative bills and repayment codes retained as source values.",
            "Undocumented education categories consolidated into category 0 in the feature transformer.",
            "Distinct customer IDs retained; identical feature rows alone do not establish duplicate people.",
            "No timestamps permitting out-of-time validation; random holdout measures same-cohort generalization only.",
        ],
    }
    return frame, audit


def build_sqlite(frame: pd.DataFrame, path: Path, sql_dir: Path, reports_dir: Path):
    """Store customer attributes and one row per historical month without target leakage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    customer_columns = ["id", "limit_bal", "sex", "education", "marriage", "age", "target"]
    histories = []
    for index, month in enumerate(["2005-09", "2005-08", "2005-07", "2005-06", "2005-05", "2005-04"]):
        block = frame[["id", STATUS[index], BILLS[index], PAYMENTS[index]]].copy()
        block.columns = ["customer_id", "repayment_status", "bill_amount", "payment_amount"]
        block["month"] = month
        histories.append(block)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        frame[customer_columns].to_sql("customers", connection, if_exists="replace", index=False)
        pd.concat(histories).to_sql("repayment_history", connection, if_exists="replace", index=False)
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS customer_pk ON customers(id)")
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS history_pk ON repayment_history(customer_id, month)")


def export_sql(root: Path, split_manifest: pd.DataFrame):
    """Persist split membership, then export training-only analytical SQL results."""
    with sqlite3.connect(root / "data/processed/credit.sqlite") as connection:
        split_manifest.to_sql("split_manifest", connection, if_exists="replace", index=False)
        for query_path in sorted((root / "sql").glob("*.sql")):
            query = query_path.read_text(encoding="utf-8")
            pd.read_sql_query(query, connection).to_csv(root / "reports/sql" / f"{query_path.stem}.csv", index=False)


def prepare_data(root: Path, skip_download=False):
    archive, provenance = download(root, skip_download=skip_download)
    frame, audit = load_dataset(archive)
    destination = root / "data/processed"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "credit.csv", index=False)
    build_sqlite(frame, destination / "credit.sqlite", root / "sql", root / "reports/sql")
    write_json(root / "reports/provenance.json", provenance)
    write_json(root / "reports/data_audit.json", audit)
    return frame, audit, provenance
