"""SQL joins must not multiply customers or leak holdout outcomes into EDA."""
import sqlite3
from pathlib import Path
import pandas as pd
from credit_risk.data import build_sqlite, export_sql


def test_relational_history_and_training_only_sql(tmp_path, raw_accounts):
    root = tmp_path
    (root / "sql").mkdir()
    source_sql = Path(__file__).resolve().parents[1] / "sql"
    for query in source_sql.glob("*.sql"):
        (root / "sql" / query.name).write_text(query.read_text(), encoding="utf-8")
    build_sqlite(raw_accounts, root / "data/processed/credit.sqlite", root / "sql", root / "reports/sql")
    manifest = pd.DataFrame({"id": raw_accounts.id,
                             "split": ["train"] * 32 + ["test"] * 16})
    export_sql(root, manifest)
    with sqlite3.connect(root / "data/processed/credit.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM repayment_history").fetchone()[0] == 48 * 6
        assert connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 48
    tables = [pd.read_csv(path) for path in (root / "reports/sql").glob("*.csv")]
    cohort = next(table for table in tables if "next_month_defaults" in table)
    assert cohort.iloc[0]["customers"] == 32
    assert cohort.iloc[0]["next_month_defaults"] == raw_accounts.iloc[:32].target.sum()
