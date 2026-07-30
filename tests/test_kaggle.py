"""Tests for Kaggle window transformation and submission creation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import (
    NUMERIC_INPUT_COLUMNS,
    POLLUTANT_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMNS,
    WEATHER_COLUMNS,
)
from src.kaggle import (
    build_kaggle_submission,
    reshape_kaggle_test_frame,
    transform_kaggle_test_frame,
    validate_kaggle_test_frame,
    validate_submission_frame,
)
from src.preprocessing import (
    AirPollutionPreprocessor,
)

RAW_COLUMNS = [
    *TIME_COLUMNS,
    *POLLUTANT_COLUMNS,
    *WEATHER_COLUMNS,
]


def make_training_frame() -> pd.DataFrame:
    """Create full-column training observations."""

    station_frames = []

    for station_number, station in enumerate(
        [
            "StationA",
            "StationB",
        ]
    ):
        datetimes = pd.date_range(
            "2015-01-01",
            periods=48,
            freq="h",
        )

        frame = pd.DataFrame(
            {
                "year": datetimes.year,
                "month": datetimes.month,
                "day": datetimes.day,
                "hour": datetimes.hour,
                "station": station,
                "wd": ["N"] * len(datetimes),
            }
        )

        for column_number, column in enumerate(
            NUMERIC_INPUT_COLUMNS
        ):
            frame[column] = (
                10.0
                + station_number * 20.0
                + column_number
                + np.arange(
                    len(datetimes),
                    dtype=float,
                )
            )

        station_frames.append(frame)

    return pd.concat(
        station_frames,
        ignore_index=True,
    )


def make_test_frame() -> pd.DataFrame:
    """Create two flattened 24-hour Kaggle windows."""

    data: dict[str, list[object]] = {
        "id": ["test_00001", "test_00002"],
        "station": ["StationA", "StationB"],
    }

    # Populate all lag columns in data[...]

    return pd.DataFrame(data)


def make_preprocessor() -> AirPollutionPreprocessor:
    """Fit a preprocessing fixture."""

    return AirPollutionPreprocessor().fit(
        make_training_frame()
    )


def test_reshape_preserves_oldest_to_latest_order() -> None:
    """Lag 24 must be first and lag 1 must be last."""

    long_frame = reshape_kaggle_test_frame(
        make_test_frame()
    )

    first_window = long_frame.loc[
        long_frame["window_id"] == 0
    ]

    assert len(first_window) == 24

    assert first_window[
        "time_step"
    ].tolist() == list(range(24))

    assert first_window.iloc[
        0
    ]["datetime"] == pd.Timestamp(
        "2016-01-01 00:00:00"
    )

    assert first_window.iloc[
        -1
    ]["datetime"] == pd.Timestamp(
        "2016-01-01 23:00:00"
    )


def test_transformed_test_tensor_shape() -> None:
    """Test windows must become 24 × feature-count tensors."""

    preprocessor = make_preprocessor()

    transformed = transform_kaggle_test_frame(
        make_test_frame(),
        preprocessor,
    )

    feature_count = len(
        preprocessor.get_feature_names_out()
    )

    assert transformed.sequence_tensor.shape == (
        2,
        24,
        feature_count,
    )

    assert transformed.flat_features.shape == (
        48,
        feature_count,
    )

    assert np.isfinite(
        transformed.sequence_tensor
    ).all()


def test_leading_missing_value_uses_training_median() -> None:
    """A leading missing value must not use another window."""

    test_frame = make_test_frame()

    test_frame.loc[
        1,
        "PM2.5_lag_24",
    ] = np.nan

    preprocessor = make_preprocessor()

    transformed = transform_kaggle_test_frame(
        test_frame,
        preprocessor,
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    pm25_feature_index = feature_names.index(
        "PM2.5"
    )

    numeric_columns = list(
        preprocessor.numeric_columns
    )

    scaler_index = numeric_columns.index(
        "PM2.5"
    )

    scaled_value = transformed.sequence_tensor[
        1,
        0,
        pm25_feature_index,
    ]

    restored_value = (
        scaled_value
        * preprocessor.scaler_.scale_[
            scaler_index
        ]
        + preprocessor.scaler_.mean_[
            scaler_index
        ]
    )

    assert np.isclose(
        restored_value,
        preprocessor.medians_["PM2.5"],
    )


def test_non_hourly_window_is_rejected() -> None:
    """A duplicated timestamp must fail validation."""

    test_frame = make_test_frame()

    test_frame.loc[
        0,
        "hour_lag_23",
    ] = test_frame.loc[
        0,
        "hour_lag_24",
    ]

    with pytest.raises(
        ValueError,
        match="non-hourly intervals",
    ):
        reshape_kaggle_test_frame(
            test_frame
        )


def test_missing_lag_column_is_rejected() -> None:
    """Every official lag column is required."""

    test_frame = make_test_frame().drop(
        columns=["PM2.5_lag_1"]
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        validate_kaggle_test_frame(
            test_frame
        )


def test_submission_preserves_ids_and_clips_negative() -> None:
    """Submission IDs must remain ordered and predictions nonnegative."""

    sample = pd.DataFrame(
        {
            "id": [
                "test_00001",
                "test_00002",
            ],
            "PM2.5": [
                0.0,
                0.0,
            ],
        }
    )

    metadata = pd.DataFrame(
        {
            "id": [
                "test_00001",
                "test_00002",
            ]
        }
    )

    submission = build_kaggle_submission(
        sample,
        metadata,
        np.array(
            [-5.0, 30.0]
        ),
    )

    assert submission["id"].equals(
        sample["id"]
    )

    assert submission[TARGET_COLUMN].tolist() == [
        0.0,
        30.0,
    ]

    summary = validate_submission_frame(
        submission,
        sample,
    )

    assert summary["rows"] == 2