"""Tests for leakage-safe air-pollution preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    NUMERIC_INPUT_COLUMNS,
    TARGET_OBSERVED_COLUMN,
)
from src.preprocessing import (
    AirPollutionPreprocessor,
    add_datetime_column,
    assign_time_split,
)


def make_test_frame() -> pd.DataFrame:
    """Create a small hourly dataset for testing."""

    rows = 8

    frame = pd.DataFrame(
        {
            "year": [2015] * rows,
            "month": [1] * rows,
            "day": [1] * rows,
            "hour": list(range(rows)),
            "station": ["TestStation"] * rows,
            "wd": [
                "N",
                "NE",
                None,
                "E",
                "SE",
                "S",
                "SW",
                "W",
            ],
        }
    )

    for index, column in enumerate(NUMERIC_INPUT_COLUMNS):
        frame[column] = (
            np.arange(
                1,
                rows + 1,
                dtype=float,
            )
            + index
        )

    frame.loc[1, "PM2.5"] = np.nan
    frame.loc[0, "CO"] = np.nan

    return frame


def test_datetime_construction() -> None:
    """Datetime must be constructed from date columns."""

    frame = make_test_frame()

    result = add_datetime_column(frame)

    assert "datetime" in result.columns
    assert result["datetime"].notna().all()
    assert result.loc[0, "datetime"] == pd.Timestamp("2015-01-01 00:00:00")


def test_target_missingness_is_preserved() -> None:
    """Imputed PM2.5 input must not replace the true target."""

    frame = add_datetime_column(make_test_frame())

    preprocessor = AirPollutionPreprocessor()

    transformed = preprocessor.fit_transform(frame)

    assert pd.isna(
        transformed.loc[
            1,
            TARGET_OBSERVED_COLUMN,
        ]
    )

    assert not pd.isna(
        transformed.loc[
            1,
            "PM2.5",
        ]
    )


def test_forward_fill_does_not_use_future_value() -> None:
    """A missing value must receive the previous value."""

    frame = add_datetime_column(make_test_frame())

    preprocessor = AirPollutionPreprocessor()

    transformed = preprocessor.fit_transform(frame)

    first_value = transformed.loc[0, "PM2.5"]
    missing_row_value = transformed.loc[1, "PM2.5"]

    assert np.isclose(
        first_value,
        missing_row_value,
    )


def test_processed_features_have_no_missing_values() -> None:
    """All model input features must be complete."""

    frame = add_datetime_column(make_test_frame())

    preprocessor = AirPollutionPreprocessor()

    transformed = preprocessor.fit_transform(frame)

    feature_columns = preprocessor.get_feature_names_out()

    assert transformed[feature_columns].isna().sum().sum() == 0


def test_unknown_wind_direction_is_marked() -> None:
    """Missing wind direction must set its indicator."""

    frame = add_datetime_column(make_test_frame())

    preprocessor = AirPollutionPreprocessor()

    transformed = preprocessor.fit_transform(frame)

    assert transformed.loc[2, "wd_missing"] == 1.0


def test_station_one_hot_feature_is_created() -> None:
    """Training stations must receive stable indicators."""

    frame = add_datetime_column(make_test_frame())

    preprocessor = AirPollutionPreprocessor()

    transformed = preprocessor.fit_transform(frame)

    station_column = "station_TestStation"

    assert station_column in transformed.columns
    assert transformed[station_column].eq(1.0).all()


def test_chronological_split_assignment() -> None:
    """Dates must be assigned to the correct split."""

    datetimes = pd.Series(
        pd.to_datetime(
            [
                "2015-08-31 23:00:00",
                "2015-09-01 00:00:00",
                "2015-11-30 23:00:00",
                "2015-12-01 00:00:00",
            ]
        )
    )

    result = assign_time_split(datetimes)

    assert list(result.astype(str)) == [
        "train",
        "validation",
        "validation",
        "local_test",
    ]
