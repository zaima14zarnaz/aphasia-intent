#!/usr/bin/env python3
"""Summarize validation metrics from a per-epoch training-results CSV file."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_CSV_PATH = Path(
    "/home/z/zaimazarnaz/research1/Aphasia/src/training_outputs/"
    "results_per_epoch.csv"
)

REQUIRED_COLUMNS = (
    "epoch",
    "train_acc",
    "train_precision",
    "train_recall",
    "train_f1",
    "val_acc",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_pos_acc",
    "val_neg_acc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the best validation metrics and POS/NEG accuracy gap."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"CSV file to analyze (default: {DEFAULT_CSV_PATH})",
    )
    return parser.parse_args()


def read_results(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    results: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(
                "CSV is missing required column(s): " + ", ".join(missing)
            )

        for line_number, row in enumerate(reader, start=2):
            parsed_row: dict[str, float] = {}
            for column in REQUIRED_COLUMNS:
                raw_value = row.get(column, "").strip()
                if not raw_value:
                    raise ValueError(
                        f"Missing value in column '{column}' at CSV line {line_number}."
                    )
                try:
                    value = float(raw_value)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid numeric value {raw_value!r} in column "
                        f"'{column}' at CSV line {line_number}."
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite value {raw_value!r} in column '{column}' "
                        f"at CSV line {line_number}."
                    )
                parsed_row[column] = value
            results.append(parsed_row)

    if not results:
        raise ValueError(f"CSV file contains no data rows: {csv_path}")

    return results


def format_epoch(epoch: float) -> str:
    return str(int(epoch)) if epoch.is_integer() else f"{epoch:g}"


def print_summary(results: list[dict[str, float]]) -> None:
    best_val_acc = max(results, key=lambda row: row["val_acc"])
    best_val_f1 = max(results, key=lambda row: row["val_f1"])
    best_val_pos_acc = max(results, key=lambda row: row["val_pos_acc"])

    average_class_gap = sum(
        (row["val_pos_acc"] - row["val_neg_acc"]) for row in results
    ) / len(results)

    print(
        "Highest validation accuracy: "
        f"{best_val_acc['val_acc']:.6f} "
        f"(epoch {format_epoch(best_val_acc['epoch'])})"
    )
    print(
        "  val precision at best val accuracy: "
        f"{best_val_acc['val_precision']:.6f}"
    )
    print(
        "  val recall at best val accuracy: "
        f"{best_val_acc['val_recall']:.6f}"
    )
    print(
        "  val F1 at best val accuracy: "
        f"{best_val_acc['val_f1']:.6f}"
    )
    print(
        "Highest validation F1: "
        f"{best_val_f1['val_f1']:.6f} "
        f"(epoch {format_epoch(best_val_f1['epoch'])})"
    )
    print(
        "Highest positive validation accuracy: "
        f"{best_val_pos_acc['val_pos_acc']:.6f} "
        f"(epoch {format_epoch(best_val_pos_acc['epoch'])})"
    )
    print(
        "Average absolute difference between positive and negative "
        f"validation accuracies: {average_class_gap:.6f}"
    )


def main() -> None:
    args = parse_args()
    results = read_results(args.csv_path)
    print_summary(results)


if __name__ == "__main__":
    main()
