#!/usr/bin/env python3
"""
Check PM2.5 prediction CSV files.

Supported formats
-----------------
1. Internal validation/local-test prediction file:
   Required columns: y_true, y_pred

   Optional columns:
   - station
   - prediction_feedforward
   - prediction_lstm

2. Kaggle submission file:
   Required columns: id, PM2.5

   For submission evaluation, also provide:
   --test data/raw/test.csv
   --test-raw data/raw/test_raw.csv

Examples
--------
Internal ensemble validation predictions:

python scripts/check_predictions.py \
    --file results/predictions/ensemble/validation_predictions.csv

Internal local-test predictions:

python scripts/check_predictions.py \
    --file results/predictions/ensemble/local_test_predictions.csv

Kaggle submission integrity check:

python scripts/check_predictions.py \
    --file data/submissions/submission.csv \
    --test data/raw/test.csv \
    --test-raw data/raw/test_raw.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

DEFAULT_FILE = (
    "results/predictions/ensemble/"
    "validation_predictions.csv"
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate PM2.5 prediction CSV files."
    )

    parser.add_argument(
        "--file",
        type=Path,
        default=Path(DEFAULT_FILE),
        help=(
            "Prediction CSV path. Defaults to the ensemble "
            "validation prediction file."
        ),
    )

    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/raw/test.csv"),
        help="Kaggle test.csv path.",
    )

    parser.add_argument(
        "--test-raw",
        type=Path,
        default=Path("data/raw/test_raw.csv"),
        help=(
            "Raw target data path used only for a permitted "
            "submission integrity check."
        ),
    )

    parser.add_argument(
        "--save-station-metrics",
        type=Path,
        default=None,
        help=(
            "Optional path for saving station-wise metrics."
        ),
    )

    return parser.parse_args()


def require_file(path: Path) -> None:
    """Raise a clear error when a file is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path.resolve()}"
        )


def clean_pairs(
    frame: pd.DataFrame,
    true_column: str,
    prediction_column: str,
) -> pd.DataFrame:
    """Return finite target-prediction pairs."""

    valid = (
        frame[[true_column, prediction_column]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if valid.empty:
        raise ValueError(
            f"No valid values found for {true_column} "
            f"and {prediction_column}."
        )

    return valid


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate RMSE, MAE and R-squared."""

    true_values = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    predicted_values = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    if true_values.shape != predicted_values.shape:
        raise ValueError(
            "Target and prediction arrays have different shapes: "
            f"{true_values.shape} vs {predicted_values.shape}"
        )

    return {
        "rmse": float(
            mean_squared_error(
                true_values,
                predicted_values,
            )
            ** 0.5
        ),
        "mae": float(
            mean_absolute_error(
                true_values,
                predicted_values,
            )
        ),
        "r2": float(
            r2_score(
                true_values,
                predicted_values,
            )
        ),
    }


def print_metric_table(
    frame: pd.DataFrame,
    prediction_columns: dict[str, str],
) -> None:
    """Print metrics for one or more prediction columns."""

    print("\nPrediction comparison")
    print("-" * 78)
    print(
        f"{'Model':<24}"
        f"{'Samples':>10}"
        f"{'RMSE':>14}"
        f"{'MAE':>14}"
        f"{'R²':>14}"
    )

    for model_name, prediction_column in (
        prediction_columns.items()
    ):
        if prediction_column not in frame.columns:
            continue

        valid = clean_pairs(
            frame,
            "y_true",
            prediction_column,
        )

        metrics = calculate_metrics(
            valid["y_true"],
            valid[prediction_column],
        )

        print(
            f"{model_name:<24}"
            f"{len(valid):>10,}"
            f"{metrics['rmse']:>14.6f}"
            f"{metrics['mae']:>14.6f}"
            f"{metrics['r2']:>14.6f}"
        )


def calculate_station_metrics(
    frame: pd.DataFrame,
    prediction_column: str = "y_pred",
) -> pd.DataFrame:
    """Calculate metrics separately for each station."""

    if "station" not in frame.columns:
        return pd.DataFrame()

    records: list[dict[str, object]] = []

    for station, station_frame in frame.groupby(
        "station",
        sort=True,
    ):
        valid = clean_pairs(
            station_frame,
            "y_true",
            prediction_column,
        )

        metrics = calculate_metrics(
            valid["y_true"],
            valid[prediction_column],
        )

        records.append(
            {
                "station": station,
                "samples": len(valid),
                **metrics,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("rmse", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_internal_predictions(
    prediction_path: Path,
    frame: pd.DataFrame,
    save_station_metrics: Path | None,
) -> None:
    """Evaluate validation or local-test prediction files."""

    required_columns = {
        "y_true",
        "y_pred",
    }

    missing_columns = sorted(
        required_columns.difference(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Internal prediction file is missing columns: "
            + ", ".join(missing_columns)
        )

    print("\nPrediction file")
    print("-" * 78)
    print(f"Path:    {prediction_path.resolve()}")
    print(f"Rows:    {len(frame):,}")
    print(f"Columns: {frame.columns.tolist()}")

    prediction_columns = {
        "Feedforward": "prediction_feedforward",
        "LSTM": "prediction_lstm",
        "Ensemble / final": "y_pred",
    }

    print_metric_table(
        frame,
        prediction_columns,
    )

    station_metrics = calculate_station_metrics(
        frame,
        prediction_column="y_pred",
    )

    if not station_metrics.empty:
        print("\nStation-wise metrics for y_pred")
        print("-" * 78)
        print(
            station_metrics.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )

        if save_station_metrics is not None:
            save_station_metrics.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            station_metrics.to_csv(
                save_station_metrics,
                index=False,
            )

            print(
                "\nSaved station metrics to:",
                save_station_metrics.resolve(),
            )


def construct_submission_truth(
    test_path: Path,
    test_raw_path: Path,
) -> pd.DataFrame:
    """Match Kaggle test IDs to target PM2.5 values."""

    require_file(test_path)
    require_file(test_raw_path)

    test = pd.read_csv(
        test_path,
        low_memory=False,
    )

    test_raw = pd.read_csv(
        test_raw_path,
        low_memory=False,
    )

    lag_1_columns = [
        "year_lag_1",
        "month_lag_1",
        "day_lag_1",
        "hour_lag_1",
    ]

    missing_lag_columns = [
        column
        for column in lag_1_columns
        if column not in test.columns
    ]

    if missing_lag_columns:
        raise ValueError(
            "test.csv is missing lag-1 datetime columns: "
            + ", ".join(missing_lag_columns)
        )

    renamed_datetime = test[
        lag_1_columns
    ].rename(
        columns={
            "year_lag_1": "year",
            "month_lag_1": "month",
            "day_lag_1": "day",
            "hour_lag_1": "hour",
        }
    )

    test["target_datetime"] = (
        pd.to_datetime(
            renamed_datetime,
            errors="raise",
        )
        + pd.Timedelta(hours=1)
    )

    raw_datetime_columns = [
        "year",
        "month",
        "day",
        "hour",
    ]

    test_raw["target_datetime"] = pd.to_datetime(
        test_raw[raw_datetime_columns],
        errors="raise",
    )

    duplicate_count = int(
        test_raw.duplicated(
            subset=[
                "station",
                "target_datetime",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            "test_raw.csv contains duplicated station-target "
            f"timestamps: {duplicate_count}"
        )

    truth = test[
        [
            "id",
            "station",
            "target_datetime",
        ]
    ].merge(
        test_raw[
            [
                "station",
                "target_datetime",
                "PM2.5",
            ]
        ],
        on=[
            "station",
            "target_datetime",
        ],
        how="left",
        validate="many_to_one",
    )

    return truth.rename(
        columns={
            "PM2.5": "y_true",
        }
    )


def evaluate_submission(
    submission_path: Path,
    submission: pd.DataFrame,
    test_path: Path,
    test_raw_path: Path,
    save_station_metrics: Path | None,
) -> None:
    """Evaluate a Kaggle-format submission file."""

    expected_columns = [
        "id",
        "PM2.5",
    ]

    if submission.columns.tolist() != expected_columns:
        raise ValueError(
            "A Kaggle submission must contain exactly:\n"
            "id, PM2.5\n"
            f"Found: {submission.columns.tolist()}"
        )

    if submission["id"].duplicated().any():
        raise ValueError(
            "Submission contains duplicated IDs."
        )

    truth = construct_submission_truth(
        test_path,
        test_raw_path,
    )

    predictions = submission.rename(
        columns={
            "PM2.5": "y_pred",
        }
    )

    evaluation = truth.merge(
        predictions,
        on="id",
        how="left",
        validate="one_to_one",
    )

    print("\nSubmission file")
    print("-" * 78)
    print(f"Path:              {submission_path.resolve()}")
    print(f"Submission rows:   {len(submission):,}")
    print(f"Expected test rows:{len(truth):,}")

    if len(submission) != len(truth):
        raise ValueError(
            "Submission row count differs from test.csv: "
            f"{len(submission)} vs {len(truth)}"
        )

    valid = clean_pairs(
        evaluation,
        "y_true",
        "y_pred",
    )

    metrics = calculate_metrics(
        valid["y_true"],
        valid["y_pred"],
    )

    print("\nSubmission metrics")
    print("-" * 78)
    print(f"Matched targets:     {len(valid):,}")
    print(
        "Missing targets:     "
        f"{evaluation['y_true'].isna().sum():,}"
    )
    print(
        "Missing predictions: "
        f"{evaluation['y_pred'].isna().sum():,}"
    )
    print(f"RMSE:                {metrics['rmse']:.6f}")
    print(f"MAE:                 {metrics['mae']:.6f}")
    print(f"R²:                   {metrics['r2']:.6f}")

    station_metrics = calculate_station_metrics(
        evaluation,
        prediction_column="y_pred",
    )

    if not station_metrics.empty:
        print("\nStation-wise submission metrics")
        print("-" * 78)
        print(
            station_metrics.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )

        if save_station_metrics is not None:
            save_station_metrics.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            station_metrics.to_csv(
                save_station_metrics,
                index=False,
            )

            print(
                "\nSaved station metrics to:",
                save_station_metrics.resolve(),
            )


def main() -> None:
    """Load and evaluate the selected prediction file."""

    arguments = parse_arguments()

    require_file(arguments.file)

    frame = pd.read_csv(
        arguments.file,
        low_memory=False,
    )

    columns = set(frame.columns)

    if {
        "y_true",
        "y_pred",
    }.issubset(columns):
        evaluate_internal_predictions(
            arguments.file,
            frame,
            arguments.save_station_metrics,
        )

    elif columns == {
        "id",
        "PM2.5",
    }:
        evaluate_submission(
            arguments.file,
            frame,
            arguments.test,
            arguments.test_raw,
            arguments.save_station_metrics,
        )

    else:
        raise ValueError(
            "Unsupported prediction CSV format.\n"
            "Expected either:\n"
            "  - y_true and y_pred columns, or\n"
            "  - exactly id and PM2.5 columns.\n"
            f"Found columns: {frame.columns.tolist()}"
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
