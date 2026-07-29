"""Tests for temporal ablation utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ablation import (
    resize_sequence_windows,
    select_temporal_features,
)


def make_feature_columns() -> list[str]:
    """Create a representative processed feature list."""

    return [
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "CO",
        "O3",
        "TEMP",
        "PRES",
        "DEWP",
        "RAIN",
        "WSPM",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "wd_sin",
        "wd_cos",
        "wd_missing",
        "PM2.5_missing",
        "PM10_missing",
        "SO2_missing",
        "NO2_missing",
        "CO_missing",
        "O3_missing",
        "TEMP_missing",
        "PRES_missing",
        "DEWP_missing",
        "RAIN_missing",
        "WSPM_missing",
        "station_A",
        "station_B",
    ]


def make_sequence_index() -> pd.DataFrame:
    """Create one valid 24-hour sequence record."""

    target_datetime = pd.Timestamp(
        "2015-01-02 00:00:00"
    )

    return pd.DataFrame(
        {
            "sequence_id": [0],
            "window_start_row": [0],
            "window_end_row": [23],
            "target_row": [24],
            "window_start_datetime": [
                pd.Timestamp(
                    "2015-01-01 00:00:00"
                )
            ],
            "window_end_datetime": [
                pd.Timestamp(
                    "2015-01-01 23:00:00"
                )
            ],
            "target_datetime": [
                target_datetime
            ],
            "target": [50.0],
        }
    )


def test_all_feature_set_preserves_columns() -> None:
    """The full feature set must remain unchanged."""

    feature_columns = make_feature_columns()

    selected = select_temporal_features(
        feature_columns,
        "all_features",
    )

    assert selected == feature_columns


def test_pollution_only_excludes_weather() -> None:
    """Weather and wind features must be removed."""

    selected = select_temporal_features(
        make_feature_columns(),
        "pollution_only",
    )

    assert "PM2.5" in selected
    assert "CO" in selected
    assert "PM2.5_missing" in selected
    assert "hour_sin" in selected
    assert "station_A" in selected

    assert "TEMP" not in selected
    assert "PRES" not in selected
    assert "WSPM" not in selected
    assert "wd_sin" not in selected
    assert "TEMP_missing" not in selected


def test_resize_to_six_hours() -> None:
    """A six-hour window must use the latest six input rows."""

    result = resize_sequence_windows(
        make_sequence_index(),
        window_size=6,
    )

    assert result.loc[
        0,
        "window_start_row",
    ] == 18

    assert result.loc[
        0,
        "window_end_row",
    ] == 23

    assert result.loc[
        0,
        "window_start_datetime",
    ] == pd.Timestamp(
        "2015-01-01 18:00:00"
    )


def test_resize_preserves_target() -> None:
    """Window resizing must not change the prediction target."""

    original = make_sequence_index()

    resized = resize_sequence_windows(
        original,
        window_size=12,
    )

    assert resized.loc[
        0,
        "target_row",
    ] == original.loc[
        0,
        "target_row",
    ]

    assert resized.loc[
        0,
        "target_datetime",
    ] == original.loc[
        0,
        "target_datetime",
    ]

    assert resized.loc[
        0,
        "target",
    ] == original.loc[
        0,
        "target",
    ]


def test_window_larger_than_available_is_rejected() -> None:
    """A requested window cannot exceed the source history."""

    with pytest.raises(
        ValueError,
        match="exceeds",
    ):
        resize_sequence_windows(
            make_sequence_index(),
            window_size=48,
        )