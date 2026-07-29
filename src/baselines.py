"""Feature preparation and prediction helpers for baseline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    NUMERIC_INPUT_COLUMNS,
    TARGET_COLUMN,
    WINDOW_SIZE,
)


def inverse_standardised_column(
    values: np.ndarray,
    preprocessor: Any,
    column: str,
) -> np.ndarray:
    """Convert one standardised numerical column to original units."""

    numeric_columns = list(preprocessor.numeric_columns)

    if column not in numeric_columns:
        raise ValueError(f"{column} is not a numerical input column.")

    scaler = preprocessor.scaler_

    if scaler is None:
        raise RuntimeError("The preprocessor does not contain a fitted scaler.")

    column_index = numeric_columns.index(column)

    original_values = (
        np.asarray(values, dtype=np.float64) * scaler.scale_[column_index]
        + scaler.mean_[column_index]
    )

    return original_values.astype(np.float32)


def extract_pm25_history(
    frame: pd.DataFrame,
    sequence_index: pd.DataFrame,
    preprocessor: Any,
    *,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """Return PM2.5 histories in original concentration units."""

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"{TARGET_COLUMN} was not found in the frame.")

    if sequence_index.empty:
        return np.empty(
            (0, window_size),
            dtype=np.float32,
        )

    start_rows = sequence_index["window_start_row"].to_numpy(dtype=np.int64)

    end_rows = sequence_index["window_end_row"].to_numpy(dtype=np.int64)

    lengths = end_rows - start_rows + 1

    if not np.all(lengths == window_size):
        raise ValueError("Sequence index contains an unexpected window size.")

    history_positions = (
        start_rows[:, None]
        + np.arange(
            window_size,
            dtype=np.int64,
        )[None, :]
    )

    scaled_pm25 = frame[TARGET_COLUMN].to_numpy(dtype=np.float32)

    scaled_history = scaled_pm25[history_positions]

    return inverse_standardised_column(
        scaled_history,
        preprocessor,
        TARGET_COLUMN,
    )


def persistence_predictions(
    pm25_history: np.ndarray,
) -> np.ndarray:
    """Predict the target using the most recent PM2.5 value."""

    history = np.asarray(
        pm25_history,
        dtype=np.float32,
    )

    if history.ndim != 2 or history.shape[1] == 0:
        raise ValueError("PM2.5 history must have shape (samples, time_steps).")

    return history[:, -1].copy()


def historical_mean_predictions(
    pm25_history: np.ndarray,
) -> np.ndarray:
    """Predict using the mean PM2.5 over the input window."""

    history = np.asarray(
        pm25_history,
        dtype=np.float32,
    )

    if history.ndim != 2 or history.shape[1] == 0:
        raise ValueError("PM2.5 history must have shape (samples, time_steps).")

    return history.mean(
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)


def build_tabular_baseline_features(
    frame: pd.DataFrame,
    sequence_index: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    numeric_columns: Sequence[str] = (NUMERIC_INPUT_COLUMNS),
    window_size: int = WINDOW_SIZE,
    delta_hours: Sequence[int] = (
        1,
        3,
        6,
        12,
    ),
) -> tuple[np.ndarray, list[str]]:
    """
    Convert temporal windows into memory-efficient tabular features.

    Generated features include:

    - all model features at the latest input hour;
    - every PM2.5 lag in the window;
    - mean, standard deviation, minimum and maximum for numerical
      features over the complete window;
    - recent PM2.5 changes.
    """

    feature_columns = list(feature_columns)
    numeric_columns = list(numeric_columns)

    if TARGET_COLUMN not in feature_columns:
        raise ValueError(f"{TARGET_COLUMN} must be included in feature_columns.")

    missing_numeric_columns = [
        column for column in numeric_columns if column not in feature_columns
    ]

    if missing_numeric_columns:
        raise ValueError(
            "Numerical columns are missing from the model "
            "features: " + ", ".join(missing_numeric_columns)
        )

    if sequence_index.empty:
        raise ValueError("Cannot construct features from an empty sequence index.")

    start_rows = sequence_index["window_start_row"].to_numpy(dtype=np.int64)

    end_rows = sequence_index["window_end_row"].to_numpy(dtype=np.int64)

    lengths = end_rows - start_rows + 1

    if not np.all(lengths == window_size):
        raise ValueError("Sequence index contains an unexpected window length.")

    history_positions = (
        start_rows[:, None]
        + np.arange(
            window_size,
            dtype=np.int64,
        )[None, :]
    )

    complete_feature_matrix = frame[feature_columns].to_numpy(dtype=np.float32)

    latest_features = complete_feature_matrix[end_rows]

    latest_feature_names = [f"latest_{column}" for column in feature_columns]

    pm25_index = feature_columns.index(TARGET_COLUMN)

    pm25_history = complete_feature_matrix[
        history_positions,
        pm25_index,
    ]

    pm25_lag_names = [
        f"PM2.5_lag_{lag}"
        for lag in range(
            window_size,
            0,
            -1,
        )
    ]

    numerical_matrix = frame[numeric_columns].to_numpy(dtype=np.float32)

    numerical_history = numerical_matrix[history_positions]

    summary_arrays = [
        numerical_history.mean(
            axis=1,
            dtype=np.float64,
        ).astype(np.float32),
        numerical_history.std(
            axis=1,
            dtype=np.float64,
        ).astype(np.float32),
        numerical_history.min(axis=1),
        numerical_history.max(axis=1),
    ]

    summary_feature_names = []

    for statistic in [
        "mean",
        "std",
        "min",
        "max",
    ]:
        summary_feature_names.extend(
            [f"{column}_{statistic}_{window_size}" for column in numeric_columns]
        )

    delta_arrays = []
    delta_feature_names = []

    latest_pm25 = pm25_history[:, -1]

    for hours in delta_hours:
        if hours >= window_size:
            continue

        earlier_pm25 = pm25_history[
            :,
            -(hours + 1),
        ]

        delta_arrays.append((latest_pm25 - earlier_pm25)[:, None])

        delta_feature_names.append(f"PM2.5_change_{hours}h")

    matrix_parts = [
        latest_features,
        pm25_history,
        *summary_arrays,
        *delta_arrays,
    ]

    feature_names = (
        latest_feature_names
        + pm25_lag_names
        + summary_feature_names
        + delta_feature_names
    )

    output = np.concatenate(
        matrix_parts,
        axis=1,
    ).astype(np.float32)

    if output.shape[1] != len(feature_names):
        raise RuntimeError(
            "Feature-name count does not match the generated feature matrix."
        )

    if not np.isfinite(output).all():
        raise ValueError(
            "Generated baseline features contain missing or infinite values."
        )

    return output, feature_names


def clip_pm25_predictions(
    predictions: np.ndarray,
) -> np.ndarray:
    """Clip impossible negative PM2.5 predictions to zero."""

    values = np.asarray(
        predictions,
        dtype=np.float64,
    )

    return np.maximum(
        values,
        0.0,
    ).astype(np.float32)
