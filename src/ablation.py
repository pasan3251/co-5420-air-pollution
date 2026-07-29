"""Utilities for controlled temporal ablation experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.config import POLLUTANT_COLUMNS
from src.preprocessing import TIME_FEATURE_COLUMNS

VALID_FEATURE_SETS = {
    "all_features",
    "pollution_only",
}


def select_temporal_features(
    all_feature_columns: Sequence[str],
    feature_set: str,
) -> list[str]:
    """
    Select the model-input columns for an ablation feature set.

    pollution_only retains pollutant measurements, pollutant missingness
    indicators, cyclical time features and station identity. Weather and
    wind-related features are excluded.
    """

    if feature_set not in VALID_FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {feature_set}. "
            f"Expected one of {sorted(VALID_FEATURE_SETS)}."
        )

    all_features = list(all_feature_columns)

    if feature_set == "all_features":
        return all_features

    pollutant_missing_columns = [
        f"{column}_missing"
        for column in POLLUTANT_COLUMNS
    ]

    station_columns = [
        column
        for column in all_features
        if column.startswith("station_")
    ]

    required_columns = (
        list(POLLUTANT_COLUMNS)
        + pollutant_missing_columns
        + list(TIME_FEATURE_COLUMNS)
        + station_columns
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in all_features
    ]

    if missing_columns:
        raise ValueError(
            "Required pollution-only features are missing: "
            + ", ".join(missing_columns)
        )

    required_set = set(required_columns)

    # Preserve the original preprocessor feature order.
    selected_columns = [
        column
        for column in all_features
        if column in required_set
    ]

    return selected_columns


def resize_sequence_windows(
    sequence_index: pd.DataFrame,
    window_size: int,
) -> pd.DataFrame:
    """
    Resize indexed windows while preserving the same prediction targets.

    Smaller windows use the most recent part of the original 24-hour
    history. This ensures that all window-size experiments evaluate the
    exact same target observations.
    """

    required_columns = {
        "window_start_row",
        "window_end_row",
        "window_start_datetime",
        "window_end_datetime",
        "target_datetime",
    }

    missing_columns = sorted(
        required_columns.difference(
            sequence_index.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Sequence index is missing columns: "
            + ", ".join(missing_columns)
        )

    if window_size <= 0:
        raise ValueError(
            "window_size must be positive."
        )

    original_lengths = (
        sequence_index["window_end_row"]
        - sequence_index["window_start_row"]
        + 1
    )

    unique_lengths = original_lengths.unique()

    if len(unique_lengths) != 1:
        raise ValueError(
            "The source sequence index contains "
            "multiple window lengths."
        )

    maximum_window_size = int(
        unique_lengths[0]
    )

    if window_size > maximum_window_size:
        raise ValueError(
            f"window_size {window_size} exceeds the "
            f"available {maximum_window_size}-hour history."
        )

    result = sequence_index.copy()

    result["window_start_row"] = (
        result["window_end_row"]
        - window_size
        + 1
    )

    result["window_start_datetime"] = (
        pd.to_datetime(
            result["target_datetime"]
        )
        - pd.to_timedelta(
            window_size,
            unit="h",
        )
    )

    resized_lengths = (
        result["window_end_row"]
        - result["window_start_row"]
        + 1
    )

    if not np.all(
        resized_lengths.to_numpy()
        == window_size
    ):
        raise RuntimeError(
            "Window resizing produced an invalid length."
        )

    target_datetimes = pd.to_datetime(
        result["target_datetime"]
    )

    end_datetimes = pd.to_datetime(
        result["window_end_datetime"]
    )

    if not (
        target_datetimes
        - end_datetimes
        == pd.Timedelta(hours=1)
    ).all():
        raise ValueError(
            "The input window does not end one hour "
            "before the target."
        )

    if (result["window_start_row"] < 0).any():
        raise ValueError(
            "Window resizing produced a negative row position."
        )

    return result