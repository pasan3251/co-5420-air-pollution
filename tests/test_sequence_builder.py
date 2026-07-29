"""Tests for temporal sequence construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sequence_builder import (
    build_sequence_index,
    extract_sequence,
    filter_sequence_split,
    iter_sequence_batches,
    validate_sequence_frame,
)

FEATURE_COLUMNS = [
    "feature_a",
    "feature_b",
]


def make_sequence_frame(
    *,
    hours_per_station: int = 30,
) -> pd.DataFrame:
    """Create a continuous two-station hourly dataset."""

    station_frames = []

    for station_number, station in enumerate(["StationA", "StationB"]):
        datetimes = pd.date_range(
            start="2015-08-31 00:00:00",
            periods=hours_per_station,
            freq="h",
        )

        hour_values = np.arange(
            hours_per_station,
            dtype=np.float32,
        )

        station_frame = pd.DataFrame(
            {
                "datetime": datetimes,
                "station": station,
                "target_PM2.5": (hour_values + station_number * 100),
                "feature_a": (hour_values + station_number * 100),
                "feature_b": (hour_values * 2 + station_number * 100),
            }
        )

        station_frame["split"] = np.where(
            station_frame["datetime"] < pd.Timestamp("2015-09-01 00:00:00"),
            "train",
            "validation",
        )

        station_frames.append(station_frame)

    return (
        pd.concat(
            station_frames,
            ignore_index=True,
        )
        .sort_values(["station", "datetime"])
        .reset_index(drop=True)
    )


def test_builds_exact_24_hour_window() -> None:
    """Each target must use exactly the previous 24 rows."""

    frame = make_sequence_frame()

    sequence_index = build_sequence_index(
        frame,
        FEATURE_COLUMNS,
        window_size=24,
    )

    first_record = sequence_index.iloc[0]

    assert first_record["window_start_row"] == 0
    assert first_record["window_end_row"] == 23
    assert first_record["target_row"] == 24

    assert first_record["target_datetime"] - first_record[
        "window_start_datetime"
    ] == pd.Timedelta(hours=24)

    assert first_record["target_datetime"] - first_record[
        "window_end_datetime"
    ] == pd.Timedelta(hours=1)


def test_sequence_does_not_cross_station_boundary() -> None:
    """All input rows must belong to the target station."""

    frame = make_sequence_frame()

    sequence_index = build_sequence_index(
        frame,
        FEATURE_COLUMNS,
        window_size=24,
    )

    for record in sequence_index.itertuples(index=False):
        window_stations = (
            frame.iloc[record.window_start_row : record.window_end_row + 1]["station"]
            .unique()
            .tolist()
        )

        assert window_stations == [record.station]

        assert frame.iloc[record.target_row]["station"] == record.station


def test_missing_targets_are_skipped() -> None:
    """Missing true targets must not become training samples."""

    frame = make_sequence_frame()

    missing_target_row = 25

    frame.loc[
        missing_target_row,
        "target_PM2.5",
    ] = np.nan

    sequence_index = build_sequence_index(
        frame,
        FEATURE_COLUMNS,
        window_size=24,
    )

    assert missing_target_row not in sequence_index["target_row"].tolist()


def test_split_is_determined_by_target_time() -> None:
    """A validation target may use earlier training history."""

    frame = make_sequence_frame()

    sequence_index = build_sequence_index(
        frame,
        FEATURE_COLUMNS,
        window_size=24,
    )

    first_record = sequence_index.iloc[0]

    assert first_record["split"] == "validation"

    input_splits = (
        frame.iloc[
            first_record["window_start_row"] : first_record["window_end_row"] + 1
        ]["split"]
        .unique()
        .tolist()
    )

    assert input_splits == ["train"]


def test_extract_sequence_returns_correct_values() -> None:
    """Extracted values must align with target metadata."""

    frame = make_sequence_frame()

    sequence_index = build_sequence_index(
        frame,
        FEATURE_COLUMNS,
        window_size=24,
    )

    first_input, first_target = extract_sequence(
        frame,
        sequence_index.iloc[0],
        FEATURE_COLUMNS,
    )

    assert first_input.shape == (24, 2)

    assert np.allclose(
        first_input[:, 0],
        np.arange(24, dtype=np.float32),
    )

    assert np.isclose(
        first_target,
        24.0,
    )


def test_batch_generator_shapes() -> None:
    """Batch iteration must return 3D inputs and 1D targets."""

    frame = make_sequence_frame()

    sequence_index = build_sequence_index(
        frame,
        FEATURE_COLUMNS,
        window_size=24,
    )

    batch_iterator = iter_sequence_batches(
        frame,
        sequence_index,
        FEATURE_COLUMNS,
        batch_size=4,
        shuffle=False,
    )

    inputs, targets = next(batch_iterator)

    assert inputs.shape == (4, 24, 2)
    assert targets.shape == (4,)
    assert inputs.dtype == np.float32
    assert targets.dtype == np.float32


def test_filter_sequence_split() -> None:
    """Split filtering must retain only the requested targets."""

    frame = make_sequence_frame()

    sequence_index = build_sequence_index(
        frame,
        FEATURE_COLUMNS,
        window_size=24,
    )

    validation_index = filter_sequence_split(
        sequence_index,
        "validation",
    )

    assert not validation_index.empty
    assert validation_index["split"].eq("validation").all()


def test_hourly_gap_is_rejected() -> None:
    """A temporal gap must be detected before construction."""

    frame = make_sequence_frame()

    frame = frame.drop(index=10).reset_index(drop=True)

    with pytest.raises(
        ValueError,
        match="non-hourly intervals",
    ):
        validate_sequence_frame(
            frame,
            FEATURE_COLUMNS,
        )


def test_missing_feature_is_rejected() -> None:
    """Input features must be complete before windowing."""

    frame = make_sequence_frame()

    frame.loc[5, "feature_a"] = np.nan

    with pytest.raises(
        ValueError,
        match="Input features contain",
    ):
        validate_sequence_frame(
            frame,
            FEATURE_COLUMNS,
        )
