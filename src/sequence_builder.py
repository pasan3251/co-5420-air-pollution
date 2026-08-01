"""Utilities for constructing temporal forecasting sequences."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    STATION_COLUMN,
    TARGET_OBSERVED_COLUMN,
    WINDOW_SIZE,
)

SEQUENCE_INDEX_COLUMNS = [
    "sequence_id",
    "station",
    "window_start_row",
    "window_end_row",
    "target_row",
    "window_start_datetime",
    "window_end_datetime",
    "target_datetime",
    "split",
    "target",
]


def validate_sequence_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    station_column: str = STATION_COLUMN,
    target_column: str = TARGET_OBSERVED_COLUMN,
) -> None:
    """Validate a processed hourly dataframe before window generation."""

    required_columns = {
        "datetime",
        "split",
        station_column,
        target_column,
        *feature_columns,
    }

    missing_columns = sorted(required_columns.difference(frame.columns))

    if missing_columns:
        raise ValueError(
            "Sequence frame is missing columns: " + ", ".join(missing_columns)
        )

    if target_column in feature_columns:
        raise ValueError(f"{target_column} must not be included in the input features.")

    expected_index = pd.RangeIndex(
        start=0,
        stop=len(frame),
        step=1,
    )

    if not frame.index.equals(expected_index):
        raise ValueError("The processed frame must use a zero-based RangeIndex.")

    sorted_positions = frame.sort_values(
        [station_column, "datetime"],
        kind="stable",
    ).index.to_numpy()

    if not np.array_equal(
        sorted_positions,
        np.arange(len(frame)),
    ):
        raise ValueError("The processed frame must be sorted by station and datetime.")

    duplicate_count = int(frame.duplicated(subset=[station_column, "datetime"]).sum())

    if duplicate_count != 0:
        raise ValueError(f"Found {duplicate_count} duplicated station-datetime rows.")

    time_differences = frame.groupby(
        station_column,
        sort=False,
    )["datetime"].diff()

    invalid_intervals = time_differences.notna() & time_differences.ne(
        pd.Timedelta(hours=1)
    )

    invalid_interval_count = int(invalid_intervals.sum())

    if invalid_interval_count != 0:
        raise ValueError(f"Found {invalid_interval_count} non-hourly intervals.")

    missing_feature_values = int(frame[list(feature_columns)].isna().sum().sum())

    if missing_feature_values != 0:
        raise ValueError(
            f"Input features contain {missing_feature_values} missing values."
        )


def build_sequence_index(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    window_size: int = WINDOW_SIZE,
    station_column: str = STATION_COLUMN,
    target_column: str = TARGET_OBSERVED_COLUMN,
) -> pd.DataFrame:
    """
    Construct an index for historical windows and next-hour targets.

    A sample with target row t uses feature rows:

        t - window_size, ..., t - 2, t - 1

    The target value comes from row t.
    """

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    validate_sequence_frame(
        frame,
        feature_columns,
        station_column=station_column,
        target_column=target_column,
    )

    sequence_parts: list[pd.DataFrame] = []

    for station, station_frame in frame.groupby(
        station_column,
        sort=True,
    ):
        station_rows = station_frame.index.to_numpy(dtype=np.int64)

        if len(station_rows) <= window_size:
            continue

        target_rows = station_rows[window_size:]
        window_start_rows = station_rows[:-window_size]
        window_end_rows = station_rows[window_size - 1 : -1]

        target_values = frame.loc[target_rows, target_column].to_numpy(dtype=np.float64)

        valid_target_mask = ~np.isnan(target_values)

        target_rows = target_rows[valid_target_mask]
        window_start_rows = window_start_rows[valid_target_mask]
        window_end_rows = window_end_rows[valid_target_mask]
        target_values = target_values[valid_target_mask]

        if len(target_rows) == 0:
            continue

        target_datetimes = frame.loc[target_rows, "datetime"].to_numpy(
            dtype="datetime64[ns]"
        )

        window_start_datetimes = frame.loc[window_start_rows, "datetime"].to_numpy(
            dtype="datetime64[ns]"
        )

        window_end_datetimes = frame.loc[window_end_rows, "datetime"].to_numpy(
            dtype="datetime64[ns]"
        )

        expected_start_difference = np.timedelta64(
            window_size,
            "h",
        )

        expected_end_difference = np.timedelta64(
            1,
            "h",
        )

        if not np.all(
            target_datetimes - window_start_datetimes == expected_start_difference
        ):
            raise ValueError(f"Invalid window start detected for station {station}.")

        if not np.all(
            target_datetimes - window_end_datetimes == expected_end_difference
        ):
            raise ValueError(f"Invalid window end detected for station {station}.")

        split_values = frame.loc[target_rows, "split"].astype("string").to_numpy()

        sequence_parts.append(
            pd.DataFrame(
                {
                    "station": str(station),
                    "window_start_row": (window_start_rows),
                    "window_end_row": window_end_rows,
                    "target_row": target_rows,
                    "window_start_datetime": (window_start_datetimes),
                    "window_end_datetime": (window_end_datetimes),
                    "target_datetime": target_datetimes,
                    "split": split_values,
                    "target": target_values.astype("float32"),
                }
            )
        )

    if not sequence_parts:
        return pd.DataFrame(columns=SEQUENCE_INDEX_COLUMNS)

    sequence_index = pd.concat(
        sequence_parts,
        ignore_index=True,
    )

    sequence_index.insert(
        0,
        "sequence_id",
        np.arange(
            len(sequence_index),
            dtype=np.int64,
        ),
    )

    return sequence_index[SEQUENCE_INDEX_COLUMNS]


def extract_sequence(
    frame: pd.DataFrame,
    sequence_record: pd.Series | dict[str, Any],
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, np.float32]:
    """Extract one input window and its target."""

    start_row = int(sequence_record["window_start_row"])

    end_row = int(sequence_record["window_end_row"])

    input_window = frame.iloc[start_row : end_row + 1][list(feature_columns)].to_numpy(
        dtype=np.float32
    )

    target = np.float32(sequence_record["target"])

    return input_window, target


def iter_sequence_batches(
    frame: pd.DataFrame,
    sequence_index: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    batch_size: int = 128,
    shuffle: bool = False,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield sequence batches without materializing the full dataset."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    feature_matrix = frame[list(feature_columns)].to_numpy(dtype=np.float32)

    sample_order = np.arange(
        len(sequence_index),
        dtype=np.int64,
    )

    if shuffle:
        random_generator = np.random.default_rng(seed)
        random_generator.shuffle(sample_order)

    for batch_start in range(
        0,
        len(sample_order),
        batch_size,
    ):
        batch_positions = sample_order[batch_start : batch_start + batch_size]

        batch_records = sequence_index.iloc[batch_positions]

        current_batch_size = len(batch_records)

        first_record = batch_records.iloc[0]

        sequence_length = (
            int(first_record["window_end_row"])
            - int(first_record["window_start_row"])
            + 1
        )

        inputs = np.empty(
            (
                current_batch_size,
                sequence_length,
                len(feature_columns),
            ),
            dtype=np.float32,
        )

        targets = batch_records["target"].to_numpy(dtype=np.float32)

        for batch_index, record in enumerate(batch_records.itertuples(index=False)):
            start_row = int(record.window_start_row)

            end_row = int(record.window_end_row)

            inputs[batch_index] = feature_matrix[start_row : end_row + 1]

        yield inputs, targets


def filter_sequence_split(
    sequence_index: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Return a clean sequence index for one data split."""

    valid_splits = {
        "train",
        "validation",
        "local_test",
    }

    if split not in valid_splits:
        raise ValueError(
            f"Unknown split: {split}. Expected one of {sorted(valid_splits)}."
        )

    return sequence_index.loc[sequence_index["split"] == split].reset_index(drop=True)
