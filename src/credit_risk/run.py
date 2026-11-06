"""Command-line entry point for the complete, reproducible experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import prepare_data
from .train import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root (default: current directory)")
    parser.add_argument("--quick", action="store_true", help="One configuration per model; development smoke run only")
    parser.add_argument("--skip-download", action="store_true", help="Require an existing data/raw/uci_credit_card.zip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fn-cost", type=float, default=5.0, help="Assumed cost of a missed default; illustrative units")
    parser.add_argument("--fp-cost", type=float, default=1.0, help="Assumed cost of an unnecessary review; illustrative units")
    arguments = parser.parse_args()
    if arguments.fn_cost <= 0 or arguments.fp_cost <= 0:
        parser.error("Cost assumptions must be positive.")
    root = arguments.root.resolve()
    if not (root / "sql").is_dir():
        parser.error("--root must point to the repository containing sql/.")
    frame, audit, provenance = prepare_data(root, arguments.skip_download)
    run_experiment(root, frame, audit, provenance, arguments.quick, arguments.seed, arguments.fn_cost, arguments.fp_cost)


if __name__ == "__main__":
    main()
