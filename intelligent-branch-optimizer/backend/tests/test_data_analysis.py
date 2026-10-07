"""Tests for branch dataset exploratory analysis."""

from app.services.data_analysis import (
    assign_load_level,
    compute_utilization,
    format_analysis_report,
    get_column_names,
    get_customer_arrival_distribution,
    get_customer_arrivals_vs_waiting_time,
    get_data_types,
    get_date_range,
    get_duplicate_rows,
    get_load_level_distribution,
    get_missing_values,
    get_numerical_statistics,
    get_row_column_counts,
    get_staff_availability_vs_waiting_time,
    get_unique_branches,
    get_utilization_vs_load_level,
    get_waiting_time_distribution,
    load_dataset,
    resolve_dataset_path,
    run_dataset_analysis,
)

EXPECTED_COLUMNS = [
    "branch_id",
    "branch_name",
    "region",
    "date",
    "hour",
    "customers_in_queue",
    "avg_wait_minutes",
    "staff_on_duty",
    "tellers_active",
    "service_desks_open",
    "transactions_count",
    "avg_service_time_min",
    "customer_satisfaction_score",
    "atm_transactions",
    "digital_signups",
    "complaints_count",
]


def test_dataset_path_exists() -> None:
    path = resolve_dataset_path()
    assert path.is_file()


def test_load_dataset_shape() -> None:
    df = load_dataset()
    shape = get_row_column_counts(df)
    assert shape["rows"] >= 1
    assert shape["columns"] == len(EXPECTED_COLUMNS)


def test_column_names_match_schema() -> None:
    df = load_dataset()
    assert get_column_names(df) == EXPECTED_COLUMNS


def test_no_missing_or_duplicate_rows() -> None:
    df = load_dataset()
    missing = get_missing_values(df)
    assert missing["total_missing_cells"] == 0
    dupes = get_duplicate_rows(df)
    assert dupes["duplicate_row_count"] == 0


def test_unique_branches_and_date_range() -> None:
    df = load_dataset()
    branches = get_unique_branches(df)
    assert branches["unique_branch_count"] == 5
    assert len(branches["branch_ids"]) == 5
    date_range = get_date_range(df)
    assert date_range["min_date"] == "2025-01-06"
    assert date_range["max_date"] == "2025-01-06"


def test_numerical_statistics_nonempty() -> None:
    df = load_dataset()
    stats = get_numerical_statistics(df)
    assert "avg_wait_minutes" in stats
    assert "mean" in stats["avg_wait_minutes"]


def test_distributions_and_relationships() -> None:
    df = load_dataset()
    arrivals = get_customer_arrival_distribution(df)
    assert "customers_in_queue" in arrivals
    wait = get_waiting_time_distribution(df)
    assert "avg_wait_minutes" in wait
    load = get_load_level_distribution(df)
    assert sum(load["counts"].values()) == len(df)

    staff = get_staff_availability_vs_waiting_time(df)
    assert "staff_on_duty" in staff
    assert staff["staff_on_duty"]["observations"] == len(df)

    arr_wait = get_customer_arrivals_vs_waiting_time(df)
    assert arr_wait["customers_in_queue_vs_avg_wait_minutes"]["pearson_r"] is not None

    util = get_utilization_vs_load_level(df)
    assert "by_load_level" in util


def test_utilization_and_load_level_helpers() -> None:
    df = load_dataset()
    util = compute_utilization(df)
    assert len(util) == len(df)
    assert (util > 0).all()
    levels = assign_load_level(df)
    assert set(levels.dropna().unique()).issubset({"low", "medium", "high"})


def test_run_dataset_analysis_and_report() -> None:
    analysis = run_dataset_analysis()
    assert analysis["shape"]["columns"] == 16
    report = format_analysis_report(analysis)
    assert "Branch dataset analysis" in report
    assert "Pearson" in report or "r=" in report


def test_data_types_include_date() -> None:
    df = load_dataset()
    dtypes = get_data_types(df)
    assert "datetime64" in dtypes["date"] or "date" in dtypes["date"].lower()
