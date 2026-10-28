#!/usr/bin/env python3
"""Find the epoch with the best validation accuracy in results_per_epoch_multi.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = _SCRIPT_DIR / "results_per_epoch_nchunks_11.csv"

REQUIRED_COLUMNS = (
    "epoch",
    "val_acc",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_pos_acc",
    "val_neg_acc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print metrics for the epoch with the best validation accuracy."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Per-epoch metrics CSV (default: {DEFAULT_CSV})",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(
                f"{csv_path} is missing required column(s): " + ", ".join(missing)
            )

        for line_number, row in enumerate(reader, start=2):
            parsed: dict[str, float] = {}
            for column in REQUIRED_COLUMNS:
                raw_value = (row.get(column) or "").strip()
                if not raw_value:
                    raise ValueError(
                        f"Missing value in column '{column}' at "
                        f"{csv_path} line {line_number}."
                    )
                parsed[column] = float(raw_value)
            rows.append(parsed)

    if not rows:
        raise ValueError(f"CSV file contains no data rows: {csv_path}")
    return rows


def format_epoch(epoch: float) -> str:
    return str(int(epoch)) if float(epoch).is_integer() else f"{epoch:g}"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    best = max(rows, key=lambda row: (row["val_acc"], -row["epoch"]))

    print(f"Best validation epoch: {format_epoch(best['epoch'])}")
    print(f"val_acc: {best['val_acc']}")
    print(f"val_precision: {best['val_precision']}")
    print(f"val_recall: {best['val_recall']}")
    print(f"val_f1: {best['val_f1']}")
    print(f"val_pos_acc: {best['val_pos_acc']}")
    print(f"val_neg_acc: {best['val_neg_acc']}")


if __name__ == "__main__":
    main()
