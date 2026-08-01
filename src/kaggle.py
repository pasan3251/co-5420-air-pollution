"""Kaggle test-window transformation and submission utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    ID_COLUMN,
    POLLUTANT_COLUMNS,
    STATION_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMNS,
    WEATHER_COLUMNS,
    WINDOW_SIZE,
)
from src.preprocessing import (
    WIND_DIRECTION_TO_DEGREES,
)

RAW_SEQUENCE_COLUMNS = [
    *TIME_COLUMNS,
    *POLLUTANT_COLUMNS,
    *WEATHER_COLUMNS,
]


@dataclass(frozen=True)
class KaggleTestData:
    """Processed Kaggle test representations."""

    metadata: pd.DataFrame
    flat_features: pd.DataFrame
    sequence_index: pd.DataFrame
    sequence_tensor: np.ndarray


def expected_kaggle_test_columns(
    *,
    window_size: int = WINDOW_SIZE,
) -> list[str]:
    """Return the required flattened Kaggle test columns."""

    columns = [
        ID_COLUMN,
        STATION_COLUMN,
    ]

    for lag in range(
        window_size,
        0,
        -1,
    ):
        columns.extend(
            [
                f"{column}_lag_{lag}"
                for column in RAW_SEQUENCE_COLUMNS
            ]
        )

    return columns


def validate_kaggle_test_frame(
    frame: pd.DataFrame,
    *,
    window_size: int = WINDOW_SIZE,
) -> None:
    """Validate the official flattened Kaggle test frame."""

    if frame.empty:
        raise ValueError(
            "The Kaggle test frame must not be empty."
        )

    required_columns = expected_kaggle_test_columns(
        window_size=window_size
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            "Kaggle test data is missing columns: "
            + ", ".join(missing_columns[:10])
        )

    if frame[ID_COLUMN].isna().any():
        raise ValueError(
            "Kaggle test IDs must not be missing."
        )

    if frame[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Kaggle test IDs must be unique."
        )

    if frame[STATION_COLUMN].isna().any():
        raise ValueError(
            "Kaggle test stations must not be missing."
        )


def reshape_kaggle_test_frame(
    frame: pd.DataFrame,
    *,
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """Convert flattened test rows into sample-major hourly rows."""

    validate_kaggle_test_frame(
        frame,
        window_size=window_size,
    )

    window_ids = np.arange(
        len(frame),
        dtype=np.int64,
    )

    parts = []

    for time_step, lag in enumerate(
        range(
            window_size,
            0,
            -1,
        )
    ):
        lag_columns = [
            f"{column}_lag_{lag}"
            for column in RAW_SEQUENCE_COLUMNS
        ]

        part = frame[
            [
                ID_COLUMN,
                STATION_COLUMN,
                *lag_columns,
            ]
        ].copy()

        part = part.rename(
            columns={
                f"{column}_lag_{lag}": column
                for column in RAW_SEQUENCE_COLUMNS
            }
        )

        part["window_id"] = window_ids
        part["time_step"] = time_step

        parts.append(part)

    long_frame = pd.concat(
        parts,
        ignore_index=True,
    )

    long_frame = (
        long_frame
        .sort_values(
            [
                "window_id",
                "time_step",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    long_frame["datetime"] = pd.to_datetime(
        long_frame[TIME_COLUMNS],
        errors="raise",
    )

    time_differences = (
        long_frame
        .groupby(
            "window_id",
            sort=False,
        )["datetime"]
        .diff()
    )

    invalid_intervals = (
        time_differences.notna()
        & time_differences.ne(
            pd.Timedelta(hours=1)
        )
    )

    invalid_count = int(
        invalid_intervals.sum()
    )

    if invalid_count != 0:
        raise ValueError(
            "Kaggle test data contains "
            f"{invalid_count} non-hourly intervals."
        )

    station_counts = (
        long_frame
        .groupby(
            "window_id",
            sort=False,
        )[STATION_COLUMN]
        .nunique()
    )

    if not station_counts.eq(1).all():
        raise ValueError(
            "At least one Kaggle window contains "
            "multiple stations."
        )

    return long_frame


def transform_kaggle_test_frame(
    frame: pd.DataFrame,
    preprocessor: Any,
    *,
    window_size: int = WINDOW_SIZE,
) -> KaggleTestData:
    """
    Transform independent Kaggle windows with a fitted preprocessor.

    Numerical forward filling occurs separately inside each 24-hour
    window. Values are never carried between Kaggle test samples.
    """

    feature_columns = (
        preprocessor.get_feature_names_out()
    )

    long_frame = reshape_kaggle_test_frame(
        frame,
        window_size=window_size,
    )

    long_frame[STATION_COLUMN] = (
        long_frame[STATION_COLUMN].astype(str)
    )

    unknown_stations = sorted(
        set(
            long_frame[
                STATION_COLUMN
            ].unique()
        ).difference(
            preprocessor.station_categories_
        )
    )

    if unknown_stations:
        raise ValueError(
            "Kaggle test data contains unknown stations: "
            + ", ".join(unknown_stations)
        )

    numeric_columns = list(
        preprocessor.numeric_columns
    )

    for column in numeric_columns:
        long_frame[f"{column}_missing"] = (
            long_frame[column]
            .isna()
            .astype("float32")
        )

    # Each competition row is an independent historical window.
    long_frame[numeric_columns] = (
        long_frame
        .groupby(
            "window_id",
            sort=False,
        )[numeric_columns]
        .ffill()
    )

    long_frame[numeric_columns] = (
        long_frame[numeric_columns]
        .fillna(preprocessor.medians_)
    )

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

    wind_values = (
        long_frame["wd"]
        .astype("string")
        .str.upper()
    )

    wind_degrees = wind_values.map(
        WIND_DIRECTION_TO_DEGREES
    )

    unknown_wind = wind_degrees.isna()

    wind_radians = np.deg2rad(
        wind_degrees.fillna(0.0).astype(float)
    )

    long_frame["wd_sin"] = np.sin(
        wind_radians
    ).astype("float32")

    long_frame["wd_cos"] = np.cos(
        wind_radians
    ).astype("float32")

    long_frame["wd_missing"] = (
        unknown_wind.astype("float32")
    )

    for station in (
        preprocessor.station_categories_
    ):
        long_frame[f"station_{station}"] = (
            long_frame[STATION_COLUMN]
            .eq(station)
            .astype("float32")
        )

    scaler = preprocessor.scaler_

    if scaler is None:
        raise RuntimeError(
            "The fitted preprocessor has no scaler."
        )

    scaled_values = scaler.transform(
        long_frame[numeric_columns]
    )

    long_frame[numeric_columns] = (
        scaled_values.astype("float32")
    )

    missing_feature_count = int(
        long_frame[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    if missing_feature_count != 0:
        raise ValueError(
            "Processed Kaggle features contain "
            f"{missing_feature_count} missing values."
        )

    flat_features = (
        long_frame[feature_columns]
        .copy()
        .reset_index(drop=True)
    )

    feature_values = flat_features.to_numpy(
        dtype=np.float32
    )

    expected_rows = (
        len(frame) * window_size
    )

    if len(feature_values) != expected_rows:
        raise RuntimeError(
            "Unexpected number of transformed "
            "Kaggle hourly rows."
        )

    sequence_tensor = feature_values.reshape(
        len(frame),
        window_size,
        len(feature_columns),
    )

    sequence_tensor = np.ascontiguousarray(
        sequence_tensor,
        dtype=np.float32,
    )

    first_rows = (
        long_frame.loc[
            long_frame["time_step"] == 0
        ]
        .sort_values("window_id")
        .reset_index(drop=True)
    )

    last_rows = (
        long_frame.loc[
            long_frame["time_step"]
            == window_size - 1
        ]
        .sort_values("window_id")
        .reset_index(drop=True)
    )

    metadata = (
        frame[
            [
                ID_COLUMN,
                STATION_COLUMN,
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    metadata["window_start_datetime"] = (
        first_rows["datetime"].to_numpy()
    )

    metadata["window_end_datetime"] = (
        last_rows["datetime"].to_numpy()
    )

    metadata["target_datetime"] = (
        pd.to_datetime(
            metadata["window_end_datetime"]
        )
        + pd.Timedelta(hours=1)
    )

    start_rows = (
        np.arange(
            len(frame),
            dtype=np.int64,
        )
        * window_size
    )

    sequence_index = pd.DataFrame(
        {
            "sequence_id": np.arange(
                len(frame),
                dtype=np.int64,
            ),
            "window_start_row": start_rows,
            "window_end_row": (
                start_rows
                + window_size
                - 1
            ),
        }
    )

    return KaggleTestData(
        metadata=metadata,
        flat_features=flat_features,
        sequence_index=sequence_index,
        sequence_tensor=sequence_tensor,
    )


def validate_submission_frame(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> dict[str, float | int]:
    """Validate a generated competition submission."""

    expected_columns = [
        ID_COLUMN,
        TARGET_COLUMN,
    ]

    if submission.columns.tolist() != expected_columns:
        raise ValueError(
            "Submission columns must be exactly: "
            + ", ".join(expected_columns)
        )

    if sample_submission.columns.tolist() != (
        expected_columns
    ):
        raise ValueError(
            "Unexpected sample-submission columns."
        )

    if len(submission) != len(sample_submission):
        raise ValueError(
            "Submission row count does not match "
            "the sample submission."
        )

    if not submission[ID_COLUMN].equals(
        sample_submission[ID_COLUMN]
    ):
        raise ValueError(
            "Submission ID order does not match "
            "the sample submission."
        )

    if submission[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Submission IDs must be unique."
        )

    predictions = submission[
        TARGET_COLUMN
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Submission predictions contain "
            "missing or infinite values."
        )

    if (predictions < 0.0).any():
        raise ValueError(
            "Submission predictions contain "
            "negative PM2.5 values."
        )

    return {
        "rows": len(submission),
        "prediction_min": float(
            predictions.min()
        ),
        "prediction_max": float(
            predictions.max()
        ),
        "prediction_mean": float(
            predictions.mean()
        ),
        "prediction_standard_deviation": float(
            predictions.std()
        ),
    }


def build_kaggle_submission(
    sample_submission: pd.DataFrame,
    test_metadata: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Build a valid submission in the official ID order."""

    values = np.asarray(
        predictions,
        dtype=np.float64,
    ).reshape(-1)

    if len(values) != len(test_metadata):
        raise ValueError(
            "Prediction count does not match "
            "Kaggle test metadata."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Predictions contain missing or "
            "infinite values."
        )

    if not test_metadata[
        ID_COLUMN
    ].reset_index(drop=True).equals(
        sample_submission[
            ID_COLUMN
        ].reset_index(drop=True)
    ):
        raise ValueError(
            "Test metadata ID order does not match "
            "the sample submission."
        )

    values = np.maximum(
        values,
        0.0,
    )

    submission = sample_submission.copy()

    submission[TARGET_COLUMN] = (
        values.astype(np.float64)
    )

    validate_submission_frame(
        submission,
        sample_submission,
    )

    return submission