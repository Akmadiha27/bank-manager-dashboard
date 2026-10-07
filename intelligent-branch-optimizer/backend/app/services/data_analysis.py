"""Exploratory analysis helpers for the synthetic branch dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import get_settings

LOAD_LEVEL_LABELS = ("low", "medium", "high")


def resolve_dataset_path(csv_path: str | Path | None = None) -> Path:
    """Resolve CSV path; default comes from application settings."""
    if csv_path is None:
        csv_path = get_settings().data_csv_path
    path = Path(csv_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path


def load_dataset(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the branch dataset from disk."""
    path = resolve_dataset_path(csv_path)
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def get_row_column_counts(df: pd.DataFrame) -> dict[str, int]:
    return {"rows": int(len(df)), "columns": int(len(df.columns))}


def get_column_names(df: pd.DataFrame) -> list[str]:
    return list(df.columns)


def get_data_types(df: pd.DataFrame) -> dict[str, str]:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def get_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    missing = df.isna().sum()
    return {
        "per_column": {col: int(missing[col]) for col in df.columns},
        "total_missing_cells": int(missing.sum()),
        "rows_with_any_missing": int(df.isna().any(axis=1).sum()),
    }


def get_duplicate_rows(df: pd.DataFrame) -> dict[str, int]:
    dup_mask = df.duplicated(keep=False)
    return {
        "duplicate_row_count": int(df.duplicated().sum()),
        "rows_in_duplicate_groups": int(dup_mask.sum()),
    }


def get_unique_branches(df: pd.DataFrame) -> dict[str, Any]:
    if "branch_id" not in df.columns:
        return {"unique_branch_count": 0, "branch_ids": [], "branch_names": []}
    ids = df["branch_id"].dropna().unique().tolist()
    names = (
        df.drop_duplicates("branch_id")["branch_name"].tolist()
        if "branch_name" in df.columns
        else []
    )
    return {
        "unique_branch_count": int(df["branch_id"].nunique()),
        "branch_ids": sorted(ids),
        "branch_names": names,
    }


def get_date_range(df: pd.DataFrame) -> dict[str, str | None]:
    if "date" not in df.columns or df["date"].isna().all():
        return {"min_date": None, "max_date": None, "unique_days": 0}
    dates = pd.to_datetime(df["date"])
    return {
        "min_date": dates.min().strftime("%Y-%m-%d"),
        "max_date": dates.max().strftime("%Y-%m-%d"),
        "unique_days": int(dates.dt.normalize().nunique()),
    }


def get_numerical_statistics(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return {}
    desc = numeric.describe().round(4)
    return {
        col: {stat: float(desc.loc[stat, col]) for stat in desc.index}
        for col in desc.columns
    }


def _distribution_summary(series: pd.Series, bins: int = 5) -> dict[str, Any]:
    clean = series.dropna()
    if clean.empty:
        return {"count": 0, "histogram": {}}
    counts, edges = np.histogram(clean, bins=bins)
    histogram = {
        f"{edges[i]:.2g}-{edges[i + 1]:.2g}": int(counts[i])
        for i in range(len(counts))
    }
    return {
        "count": int(len(clean)),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=0)) if len(clean) > 1 else 0.0,
        "percentiles": {
            "25": float(clean.quantile(0.25)),
            "50": float(clean.quantile(0.50)),
            "75": float(clean.quantile(0.75)),
        },
        "histogram": histogram,
    }


def get_customer_arrival_distribution(df: pd.DataFrame) -> dict[str, Any]:
    """Distribution of queue pressure (`customers_in_queue`) and mean arrivals by hour."""
    result: dict[str, Any] = {}
    if "customers_in_queue" in df.columns:
        result["customers_in_queue"] = _distribution_summary(
            df["customers_in_queue"]
        )
    if "hour" in df.columns and "customers_in_queue" in df.columns:
        by_hour = (
            df.groupby("hour")["customers_in_queue"]
            .agg(["mean", "count"])
            .round(4)
            .reset_index()
        )
        result["mean_customers_in_queue_by_hour"] = {
            int(row["hour"]): {
                "mean": float(row["mean"]),
                "observations": int(row["count"]),
            }
            for _, row in by_hour.iterrows()
        }
    return result


def get_waiting_time_distribution(df: pd.DataFrame) -> dict[str, Any]:
    if "avg_wait_minutes" not in df.columns:
        return {}
    return {"avg_wait_minutes": _distribution_summary(df["avg_wait_minutes"])}


def compute_utilization(df: pd.DataFrame) -> pd.Series:
    """
    Utilization proxy: transactions per active teller (higher => busier staff).
    """
    if "transactions_count" not in df.columns or "tellers_active" not in df.columns:
        return pd.Series(dtype=float)
    tellers = df["tellers_active"].replace(0, np.nan)
    return df["transactions_count"] / tellers


def assign_load_level(df: pd.DataFrame) -> pd.Series:
    """
    Load level from `customers_in_queue` using tertile bins (low / medium / high).
    """
    if "customers_in_queue" not in df.columns:
        return pd.Series([pd.NA] * len(df), dtype="object")
    q1, q2 = df["customers_in_queue"].quantile([1 / 3, 2 / 3])
    if q1 == q2:
        return pd.Series(["medium"] * len(df), index=df.index, dtype="object")

    def _label(value: float) -> str:
        if value <= q1:
            return "low"
        if value <= q2:
            return "medium"
        return "high"

    return df["customers_in_queue"].apply(_label)


def get_load_level_distribution(df: pd.DataFrame) -> dict[str, Any]:
    levels = assign_load_level(df)
    counts = levels.value_counts().reindex(LOAD_LEVEL_LABELS, fill_value=0)
    total = int(len(levels))
    return {
        "definition": "Tertiles of customers_in_queue -> low / medium / high",
        "counts": {label: int(counts[label]) for label in LOAD_LEVEL_LABELS},
        "percentages": {
            label: round(100 * counts[label] / total, 2) if total else 0.0
            for label in LOAD_LEVEL_LABELS
        },
    }


def _pearson_pair(x: pd.Series, y: pd.Series) -> dict[str, float | int | None]:
    paired = pd.concat([x, y], axis=1).dropna()
    n = int(len(paired))
    if n < 2:
        return {"pearson_r": None, "observations": n}
    r = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
    return {"pearson_r": round(r, 4), "observations": n}


def get_staff_availability_vs_waiting_time(df: pd.DataFrame) -> dict[str, Any]:
    if "avg_wait_minutes" not in df.columns:
        return {}
    out: dict[str, Any] = {}
    for col in ("staff_on_duty", "tellers_active", "service_desks_open"):
        if col in df.columns:
            out[col] = _pearson_pair(df[col], df["avg_wait_minutes"])
    return out


def get_customer_arrivals_vs_waiting_time(df: pd.DataFrame) -> dict[str, Any]:
    if "customers_in_queue" not in df.columns or "avg_wait_minutes" not in df.columns:
        return {}
    return {
        "customers_in_queue_vs_avg_wait_minutes": _pearson_pair(
            df["customers_in_queue"], df["avg_wait_minutes"]
        )
    }


def get_utilization_vs_load_level(df: pd.DataFrame) -> dict[str, Any]:
    utilization = compute_utilization(df)
    levels = assign_load_level(df)
    if utilization.empty:
        return {}

    frame = pd.DataFrame({"utilization": utilization, "load_level": levels}).dropna()
    by_level = (
        frame.groupby("load_level")["utilization"]
        .agg(["mean", "median", "count"])
        .reindex(LOAD_LEVEL_LABELS)
        .round(4)
    )
    return {
        "utilization_definition": "transactions_count / tellers_active",
        "by_load_level": {
            str(idx): {
                "mean_utilization": float(row["mean"]) if pd.notna(row["mean"]) else None,
                "median_utilization": float(row["median"])
                if pd.notna(row["median"])
                else None,
                "observations": int(row["count"]) if pd.notna(row["count"]) else 0,
            }
            for idx, row in by_level.iterrows()
        },
        "overall_correlation_utilization_vs_customers_in_queue": _pearson_pair(
            utilization, df["customers_in_queue"]
        ),
    }


def run_dataset_analysis(csv_path: str | Path | None = None) -> dict[str, Any]:
    """Run all checks and return a structured report dictionary."""
    df = load_dataset(csv_path)
    return {
        "dataset_path": str(resolve_dataset_path(csv_path)),
        "shape": get_row_column_counts(df),
        "column_names": get_column_names(df),
        "data_types": get_data_types(df),
        "missing_values": get_missing_values(df),
        "duplicates": get_duplicate_rows(df),
        "branches": get_unique_branches(df),
        "date_range": get_date_range(df),
        "numerical_statistics": get_numerical_statistics(df),
        "customer_arrival_distribution": get_customer_arrival_distribution(df),
        "waiting_time_distribution": get_waiting_time_distribution(df),
        "load_level_distribution": get_load_level_distribution(df),
        "staff_availability_vs_waiting_time": get_staff_availability_vs_waiting_time(df),
        "customer_arrivals_vs_waiting_time": get_customer_arrivals_vs_waiting_time(df),
        "utilization_vs_load_level": get_utilization_vs_load_level(df),
    }


def format_analysis_report(analysis: dict[str, Any]) -> str:
    """Human-readable summary for logs or CLI output."""
    lines = [
        "=== Branch dataset analysis ===",
        f"Path: {analysis['dataset_path']}",
        f"Shape: {analysis['shape']['rows']} rows x {analysis['shape']['columns']} columns",
        f"Columns: {', '.join(analysis['column_names'])}",
        "",
        "Missing values:",
        f"  total missing cells: {analysis['missing_values']['total_missing_cells']}",
        f"  rows with any missing: {analysis['missing_values']['rows_with_any_missing']}",
        "",
        "Duplicates:",
        f"  exact duplicate rows: {analysis['duplicates']['duplicate_row_count']}",
        "",
        f"Branches: {analysis['branches']['unique_branch_count']} unique "
        f"({', '.join(analysis['branches']['branch_ids'])})",
        f"Date range: {analysis['date_range']['min_date']} to "
        f"{analysis['date_range']['max_date']} "
        f"({analysis['date_range']['unique_days']} day(s))",
        "",
        "Customer arrivals (customers_in_queue):",
    ]
    arrivals = analysis.get("customer_arrival_distribution", {}).get(
        "customers_in_queue", {}
    )
    if arrivals:
        lines.append(
            f"  mean={arrivals['mean']}, median={arrivals['median']}, "
            f"min={arrivals['min']}, max={arrivals['max']}"
        )
    wait = analysis.get("waiting_time_distribution", {}).get("avg_wait_minutes", {})
    if wait:
        lines.extend(
            [
                "",
                "Waiting time (avg_wait_minutes):",
                f"  mean={wait['mean']}, median={wait['median']}, "
                f"min={wait['min']}, max={wait['max']}",
            ]
        )
    load = analysis.get("load_level_distribution", {})
    if load:
        lines.extend(
            [
                "",
                f"Load levels ({load.get('definition', '')}):",
                f"  {load.get('counts', {})}",
            ]
        )
    staff_wait = analysis.get("staff_availability_vs_waiting_time", {})
    if staff_wait:
        lines.append("")
        lines.append("Staff vs waiting time (Pearson r):")
        for key, vals in staff_wait.items():
            lines.append(f"  {key}: r={vals.get('pearson_r')} (n={vals['observations']})")
    arr_wait = analysis.get("customer_arrivals_vs_waiting_time", {})
    if arr_wait:
        pair = arr_wait.get("customers_in_queue_vs_avg_wait_minutes", {})
        lines.append("")
        lines.append(
            f"Arrivals vs wait: r={pair.get('pearson_r')} (n={pair.get('observations')})"
        )
    util_load = analysis.get("utilization_vs_load_level", {})
    if util_load:
        lines.append("")
        lines.append(f"Utilization ({util_load.get('utilization_definition', '')}) by load:")
        for level, stats in util_load.get("by_load_level", {}).items():
            lines.append(
                f"  {level}: mean={stats.get('mean_utilization')}, "
                f"n={stats.get('observations')}"
            )
    return "\n".join(lines)


def print_analysis_report(csv_path: str | Path | None = None) -> dict[str, Any]:
    analysis = run_dataset_analysis(csv_path)
    print(format_analysis_report(analysis))
    return analysis


if __name__ == "__main__":
    print_analysis_report()
