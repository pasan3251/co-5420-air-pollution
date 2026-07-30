#!/usr/bin/env python3
"""
Generate a Kaggle PM2.5 submission from the project's saved models.

The script supports:
- feedforward model only;
- LSTM model only;
- validation-selected feedforward + LSTM ensemble.

Run from the repository root:

python scripts/generate_submission.py

Default output:

data/submissions/submission_ensemble.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import (
    WIND_DIRECTION_TO_DEGREES,
)

DEFAULT_TEST_PATH = Path("data/raw/test.csv")
DEFAULT_SAMPLE_SUBMISSION_PATH = Path(
    "data/raw/sample_submission.csv"
)
DEFAULT_PREPROCESSOR_PATH = Path(
    "data/processed/air_pollution_preprocessor.joblib"
)
DEFAULT_FEEDFORWARD_MODEL_PATH = Path(
    "models/feedforward_nn.keras"
)
DEFAULT_LSTM_MODEL_PATH = Path(
    "models/lstm_forecaster.keras"
)
DEFAULT_WEIGHT_SEARCH_PATH = Path(
    "results/metrics/ensemble_weight_search.csv"
)
DEFAULT_FEATURE_NAMES_PATH = Path(
    "results/metrics/baseline_feature_names.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "data/submissions/submission_ensemble.csv"
)

WINDOW_SIZE = 24
DELTA_HOURS = (1, 3, 6, 12)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a Kaggle PM2.5 submission from "
            "saved feedforward/LSTM models."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "ensemble",
            "feedforward",
            "lstm",
        ],
        default="ensemble",
        help="Prediction model to use.",
    )

    parser.add_argument(
        "--test",
        type=Path,
        default=DEFAULT_TEST_PATH,
    )

    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=DEFAULT_SAMPLE_SUBMISSION_PATH,
    )

    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=DEFAULT_PREPROCESSOR_PATH,
    )

    parser.add_argument(
        "--feedforward-model",
        type=Path,
        default=DEFAULT_FEEDFORWARD_MODEL_PATH,
    )

    parser.add_argument(
        "--lstm-model",
        type=Path,
        default=DEFAULT_LSTM_MODEL_PATH,
    )

    parser.add_argument(
        "--weight-search",
        type=Path,
        default=DEFAULT_WEIGHT_SEARCH_PATH,
    )

    parser.add_argument(
        "--baseline-feature-names",
        type=Path,
        default=DEFAULT_FEATURE_NAMES_PATH,
    )

    parser.add_argument(
        "--feedforward-weight",
        type=float,
        default=None,
        help=(
            "Optional manual ensemble weight. When omitted, "
            "the validation-selected weight is loaded from "
            "ensemble_weight_search.csv."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--save-components",
        action="store_true",
        help=(
            "Also save separate feedforward and LSTM "
            "submission files."
        ),
    )

    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    """Raise a clear error if a required file is missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n"
            f"{path.resolve()}"
        )


def load_selected_weight(
    weight_search_path: Path,
    manual_weight: float | None,
) -> float:
    """Load the feedforward ensemble weight."""

    if manual_weight is not None:
        if not 0.0 <= manual_weight <= 1.0:
            raise ValueError(
                "--feedforward-weight must be between 0 and 1."
            )

        return float(manual_weight)

    require_file(
        weight_search_path,
        "Ensemble weight-search file",
    )

    search = pd.read_csv(weight_search_path)

    required_columns = {
        "feedforward_weight",
        "lstm_weight",
        "rmse",
        "mae",
    }

    missing_columns = sorted(
        required_columns.difference(search.columns)
    )

    if missing_columns:
        raise ValueError(
            "Weight-search file is missing columns: "
            + ", ".join(missing_columns)
        )

    best_row = (
        search.sort_values(
            [
                "rmse",
                "mae",
                "feedforward_weight",
            ]
        )
        .iloc[0]
    )

    return float(best_row["feedforward_weight"])


def validate_test_columns(
    test: pd.DataFrame,
    numeric_columns: list[str],
) -> None:
    """Validate the expected lagged test structure."""

    required_base_columns = {
        "id",
        "station",
    }

    missing_base = sorted(
        required_base_columns.difference(test.columns)
    )

    if missing_base:
        raise ValueError(
            "test.csv is missing columns: "
            + ", ".join(missing_base)
        )

    required_lag_features = [
        "year",
        "month",
        "day",
        "hour",
        *numeric_columns,
        "wd",
    ]

    missing_lag_columns: list[str] = []

    for lag in range(WINDOW_SIZE, 0, -1):
        for feature in required_lag_features:
            column = f"{feature}_lag_{lag}"

            if column not in test.columns:
                missing_lag_columns.append(column)

    if missing_lag_columns:
        preview = ", ".join(missing_lag_columns[:10])

        raise ValueError(
            "test.csv is missing required lag columns. "
            f"First missing columns: {preview}"
        )

    if test["id"].duplicated().any():
        raise ValueError(
            "test.csv contains duplicated IDs."
        )


def reshape_test_windows(
    test: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    """
    Convert the wide lagged test data to a sample-major long dataframe.

    Rows are ordered:

    sample 0: lag_24, lag_23, ..., lag_1
    sample 1: lag_24, lag_23, ..., lag_1
    ...
    """

    long_parts: list[pd.DataFrame] = []

    for step, lag in enumerate(
        range(WINDOW_SIZE, 0, -1)
    ):
        source_columns = [
            f"year_lag_{lag}",
            f"month_lag_{lag}",
            f"day_lag_{lag}",
            f"hour_lag_{lag}",
            *[
                f"{column}_lag_{lag}"
                for column in numeric_columns
            ],
            f"wd_lag_{lag}",
        ]

        part = test[source_columns].copy()

        rename_map = {
            f"year_lag_{lag}": "year",
            f"month_lag_{lag}": "month",
            f"day_lag_{lag}": "day",
            f"hour_lag_{lag}": "hour",
            f"wd_lag_{lag}": "wd",
        }

        rename_map.update(
            {
                f"{column}_lag_{lag}": column
                for column in numeric_columns
            }
        )

        part = part.rename(
            columns=rename_map
        )

        part.insert(
            0,
            "sample_index",
            np.arange(
                len(test),
                dtype=np.int64,
            ),
        )

        part.insert(
            1,
            "step",
            step,
        )

        part["station"] = test[
            "station"
        ].to_numpy()

        long_parts.append(part)

    long_frame = pd.concat(
        long_parts,
        ignore_index=True,
    )

    long_frame = (
        long_frame
        .sort_values(
            [
                "sample_index",
                "step",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return long_frame


def preprocess_test_windows(
    test: pd.DataFrame,
    preprocessor: object,
) -> tuple[np.ndarray, list[str]]:
    """Apply the saved training preprocessor to the 24-hour test windows."""

    numeric_columns = list(
        preprocessor.numeric_columns
    )

    feature_columns = list(
        preprocessor.get_feature_names_out()
    )

    validate_test_columns(
        test,
        numeric_columns,
    )

    long_frame = reshape_test_windows(
        test,
        numeric_columns,
    )

    datetime_columns = [
        "year",
        "month",
        "day",
        "hour",
    ]

    long_frame["datetime"] = pd.to_datetime(
        long_frame[datetime_columns],
        errors="raise",
    )

    # Preserve original missingness before imputation.
    for column in numeric_columns:
        long_frame[f"{column}_missing"] = (
            long_frame[column]
            .isna()
            .astype("float32")
        )

    # Forward fill only within each independent 24-hour test window.
    long_frame[numeric_columns] = (
        long_frame
        .groupby(
            "sample_index",
            sort=False,
        )[numeric_columns]
        .ffill()
    )

    # Fill leading missing values using training-only medians.
    long_frame[numeric_columns] = (
        long_frame[numeric_columns]
        .fillna(preprocessor.medians_)
    )

    remaining_numeric_nans = int(
        long_frame[numeric_columns]
        .isna()
        .sum()
        .sum()
    )

    if remaining_numeric_nans:
        raise ValueError(
            "Numeric test features still contain "
            f"{remaining_numeric_nans} missing values."
        )

    # Apply the scaler fitted on the original training split.
    scaler = preprocessor.scaler_

    if scaler is None:
        raise RuntimeError(
            "The saved preprocessor has no fitted scaler."
        )

    long_frame[numeric_columns] = (
        scaler.transform(
            long_frame[numeric_columns]
        )
        .astype("float32")
    )

    # Cyclical temporal features.
    datetime_values = long_frame["datetime"]

    hour = datetime_values.dt.hour.astype(float)
    day_of_week = (
        datetime_values.dt.dayofweek.astype(float)
    )
    day_of_year = (
        datetime_values.dt.dayofyear.astype(float)
        - 1.0
    )

    long_frame["hour_sin"] = np.sin(
        2.0 * np.pi * hour / 24.0
    ).astype("float32")

    long_frame["hour_cos"] = np.cos(
        2.0 * np.pi * hour / 24.0
    ).astype("float32")

    long_frame["day_of_week_sin"] = np.sin(
        2.0 * np.pi * day_of_week / 7.0
    ).astype("float32")

    long_frame["day_of_week_cos"] = np.cos(
        2.0 * np.pi * day_of_week / 7.0
    ).astype("float32")

    long_frame["day_of_year_sin"] = np.sin(
        2.0 * np.pi * day_of_year / 365.25
    ).astype("float32")

    long_frame["day_of_year_cos"] = np.cos(
        2.0 * np.pi * day_of_year / 365.25
    ).astype("float32")

    # Circular wind direction.
    wind_values = (
        long_frame["wd"]
        .astype("string")
        .str.upper()
    )

    degrees = wind_values.map(
        WIND_DIRECTION_TO_DEGREES
    )

    unknown_direction = degrees.isna()

    radians = np.deg2rad(
        degrees.fillna(0.0).astype(float)
    )

    long_frame["wd_sin"] = np.sin(
        radians
    ).astype("float32")

    long_frame["wd_cos"] = np.cos(
        radians
    ).astype("float32")

    long_frame["wd_missing"] = (
        unknown_direction.astype("float32")
    )

    # Stable station one-hot columns.
    station_categories = list(
        preprocessor.station_categories_
    )

    unknown_stations = sorted(
        set(
            test["station"]
            .astype(str)
            .unique()
        ).difference(station_categories)
    )

    if unknown_stations:
        raise ValueError(
            "Unknown stations in test.csv: "
            + ", ".join(unknown_stations)
        )

    for station in station_categories:
        long_frame[f"station_{station}"] = (
            long_frame["station"]
            .astype(str)
            .eq(station)
            .astype("float32")
        )

    missing_features = sorted(
        set(feature_columns).difference(
            long_frame.columns
        )
    )

    if missing_features:
        raise ValueError(
            "Unable to construct model features: "
            + ", ".join(missing_features)
        )

    feature_values = long_frame[
        feature_columns
    ].to_numpy(dtype=np.float32)

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Processed test features contain missing or "
            "infinite values."
        )

    sequence_features = feature_values.reshape(
        len(test),
        WINDOW_SIZE,
        len(feature_columns),
    )

    return sequence_features, feature_columns


def build_feedforward_features(
    sequence_features: np.ndarray,
    feature_columns: list[str],
    numeric_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Create the same 115 tabular features used during training."""

    latest_features = sequence_features[
        :,
        -1,
        :,
    ]

    latest_feature_names = [
        f"latest_{column}"
        for column in feature_columns
    ]

    pm25_index = feature_columns.index(
        "PM2.5"
    )

    pm25_history = sequence_features[
        :,
        :,
        pm25_index,
    ]

    pm25_lag_names = [
        f"PM2.5_lag_{lag}"
        for lag in range(
            WINDOW_SIZE,
            0,
            -1,
        )
    ]

    numeric_indices = [
        feature_columns.index(column)
        for column in numeric_columns
    ]

    numeric_history = sequence_features[
        :,
        :,
        numeric_indices,
    ]

    summary_arrays = [
        numeric_history.mean(
            axis=1,
            dtype=np.float64,
        ).astype(np.float32),
        numeric_history.std(
            axis=1,
            dtype=np.float64,
        ).astype(np.float32),
        numeric_history.min(axis=1),
        numeric_history.max(axis=1),
    ]

    summary_feature_names: list[str] = []

    for statistic in [
        "mean",
        "std",
        "min",
        "max",
    ]:
        summary_feature_names.extend(
            [
                f"{column}_{statistic}_{WINDOW_SIZE}"
                for column in numeric_columns
            ]
        )

    latest_pm25 = pm25_history[:, -1]

    delta_arrays: list[np.ndarray] = []
    delta_feature_names: list[str] = []

    for hours in DELTA_HOURS:
        earlier_pm25 = pm25_history[
            :,
            -(hours + 1),
        ]

        delta_arrays.append(
            (
                latest_pm25
                - earlier_pm25
            )[:, None]
        )

        delta_feature_names.append(
            f"PM2.5_change_{hours}h"
        )

    tabular_features = np.concatenate(
        [
            latest_features,
            pm25_history,
            *summary_arrays,
            *delta_arrays,
        ],
        axis=1,
    ).astype(np.float32)

    feature_names = (
        latest_feature_names
        + pm25_lag_names
        + summary_feature_names
        + delta_feature_names
    )

    if tabular_features.shape[1] != len(
        feature_names
    ):
        raise RuntimeError(
            "Tabular feature count does not match names."
        )

    if not np.isfinite(
        tabular_features
    ).all():
        raise ValueError(
            "Feedforward test features contain invalid values."
        )

    return tabular_features, feature_names


def verify_feedforward_feature_names(
    actual_feature_names: list[str],
    metadata_path: Path,
) -> None:
    """Verify exact compatibility with the training feature order."""

    if not metadata_path.exists():
        print(
            "WARNING: Baseline feature-name metadata was not "
            "found; skipping exact name verification."
        )
        return

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    expected_names = metadata.get(
        "feature_names"
    )

    if expected_names != actual_feature_names:
        raise ValueError(
            "Generated feedforward feature order does not "
            "match the training feature metadata."
        )


def predict_feedforward(
    model_path: Path,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Load and run the feedforward model."""

    require_file(
        model_path,
        "Feedforward model",
    )

    model = tf.keras.models.load_model(
        model_path
    )

    predictions = model.predict(
        features,
        batch_size=batch_size,
        verbose=0,
    ).reshape(-1)

    return np.maximum(
        predictions,
        0.0,
    ).astype(np.float32)


def predict_lstm(
    model_path: Path,
    sequence_features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Load and run the LSTM model."""

    require_file(
        model_path,
        "LSTM model",
    )

    model = tf.keras.models.load_model(
        model_path
    )

    predictions = model.predict(
        sequence_features,
        batch_size=batch_size,
        verbose=0,
    ).reshape(-1)

    return np.maximum(
        predictions,
        0.0,
    ).astype(np.float32)


def validate_submission(
    submission: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Validate the exact Kaggle submission contract."""

    if submission.columns.tolist() != [
        "id",
        "PM2.5",
    ]:
        raise ValueError(
            "Submission columns must be exactly: id, PM2.5"
        )

    if len(submission) != len(test):
        raise ValueError(
            "Submission row count does not match test.csv."
        )

    if not submission["id"].equals(
        test["id"]
    ):
        raise ValueError(
            "Submission ID order does not match test.csv."
        )

    if submission["id"].duplicated().any():
        raise ValueError(
            "Submission contains duplicated IDs."
        )

    values = submission[
        "PM2.5"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            "Submission predictions contain invalid values."
        )

    if (values < 0.0).any():
        raise ValueError(
            "Submission contains negative PM2.5 values."
        )


def save_component_submission(
    sample_submission: pd.DataFrame,
    predictions: np.ndarray,
    path: Path,
) -> None:
    """Save one component-model submission."""

    output = sample_submission.copy()
    output["PM2.5"] = predictions

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        path,
        index=False,
    )


def main() -> None:
    """Generate and validate the requested submission."""

    arguments = parse_arguments()

    if arguments.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    require_file(
        arguments.test,
        "Kaggle test.csv",
    )

    require_file(
        arguments.sample_submission,
        "Sample submission",
    )

    require_file(
        arguments.preprocessor,
        "Saved preprocessor",
    )

    print(
        "TensorFlow version:",
        tf.__version__,
    )

    print(
        "GPU devices:",
        tf.config.list_physical_devices(
            "GPU"
        ),
    )

    print("\nLoading competition files...")

    test = pd.read_csv(
        arguments.test,
        low_memory=False,
    )

    sample_submission = pd.read_csv(
        arguments.sample_submission,
        low_memory=False,
    )

    if sample_submission.columns.tolist() != [
        "id",
        "PM2.5",
    ]:
        raise ValueError(
            "sample_submission.csv must contain exactly "
            "id and PM2.5."
        )

    if not sample_submission["id"].equals(
        test["id"]
    ):
        raise ValueError(
            "Sample-submission ID order differs from test.csv."
        )

    preprocessor = joblib.load(
        arguments.preprocessor
    )

    print(
        f"Test rows: {len(test):,}"
    )

    print("Constructing 24-hour model inputs...")

    (
        sequence_features,
        feature_columns,
    ) = preprocess_test_windows(
        test,
        preprocessor,
    )

    print(
        "Sequence input shape:",
        sequence_features.shape,
    )

    feedforward_predictions: np.ndarray | None = None
    lstm_predictions: np.ndarray | None = None

    if arguments.mode in {
        "ensemble",
        "feedforward",
    }:
        print(
            "Constructing feedforward tabular features..."
        )

        (
            tabular_features,
            tabular_feature_names,
        ) = build_feedforward_features(
            sequence_features,
            feature_columns,
            list(preprocessor.numeric_columns),
        )

        verify_feedforward_feature_names(
            tabular_feature_names,
            arguments.baseline_feature_names,
        )

        print(
            "Feedforward input shape:",
            tabular_features.shape,
        )

        print("Predicting with feedforward model...")

        feedforward_predictions = (
            predict_feedforward(
                arguments.feedforward_model,
                tabular_features,
                arguments.batch_size,
            )
        )

    if arguments.mode in {
        "ensemble",
        "lstm",
    }:
        print("Predicting with LSTM model...")

        lstm_predictions = predict_lstm(
            arguments.lstm_model,
            sequence_features,
            arguments.batch_size,
        )

    if arguments.mode == "feedforward":
        assert feedforward_predictions is not None

        final_predictions = feedforward_predictions
        output_name = "feedforward"

    elif arguments.mode == "lstm":
        assert lstm_predictions is not None

        final_predictions = lstm_predictions
        output_name = "lstm"

    else:
        assert feedforward_predictions is not None
        assert lstm_predictions is not None

        feedforward_weight = (
            load_selected_weight(
                arguments.weight_search,
                arguments.feedforward_weight,
            )
        )

        lstm_weight = (
            1.0 - feedforward_weight
        )

        print("\nEnsemble weights")
        print("-" * 60)
        print(
            "Feedforward:",
            f"{feedforward_weight:.4f}",
        )
        print(
            "LSTM:       ",
            f"{lstm_weight:.4f}",
        )

        final_predictions = (
            feedforward_weight
            * feedforward_predictions
            + lstm_weight
            * lstm_predictions
        )

        final_predictions = np.maximum(
            final_predictions,
            0.0,
        ).astype(np.float32)

        output_name = "ensemble"

    submission = sample_submission.copy()

    submission["PM2.5"] = (
        final_predictions
    )

    validate_submission(
        submission,
        test,
    )

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        arguments.output,
        index=False,
    )

    if arguments.save_components:
        if feedforward_predictions is not None:
            save_component_submission(
                sample_submission,
                feedforward_predictions,
                arguments.output.with_name(
                    "submission_feedforward.csv"
                ),
            )

        if lstm_predictions is not None:
            save_component_submission(
                sample_submission,
                lstm_predictions,
                arguments.output.with_name(
                    "submission_lstm.csv"
                ),
            )

    print("\nSubmission summary")
    print("-" * 60)
    print("Mode:       ", output_name)
    print("Output:     ", arguments.output.resolve())
    print("Rows:       ", len(submission))
    print(
        "Minimum:    ",
        f"{submission['PM2.5'].min():.6f}",
    )
    print(
        "Maximum:    ",
        f"{submission['PM2.5'].max():.6f}",
    )
    print(
        "Mean:       ",
        f"{submission['PM2.5'].mean():.6f}",
    )
    print(
        "Missing:    ",
        int(submission["PM2.5"].isna().sum()),
    )

    print("\nFirst five rows")
    print("-" * 60)
    print(
        submission.head().to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("SUBMISSION GENERATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"\nERROR: {error}",
            file=sys.stderr,
        )

        raise SystemExit(1) from error
