"""Regression evaluation utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def validate_regression_arrays(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return one-dimensional regression arrays."""

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
            "y_true and y_pred must have identical shapes. "
            f"Received {true_values.shape} and "
            f"{predicted_values.shape}."
        )

    if len(true_values) == 0:
        raise ValueError("Regression arrays must not be empty.")

    if not np.isfinite(true_values).all():
        raise ValueError("y_true contains missing or infinite values.")

    if not np.isfinite(predicted_values).all():
        raise ValueError("y_pred contains missing or infinite values.")

    return true_values, predicted_values


def regression_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Calculate standard regression metrics."""

    true_values, predicted_values = validate_regression_arrays(
        y_true,
        y_pred,
    )

    mse = mean_squared_error(
        true_values,
        predicted_values,
    )

    return {
        "rmse": float(np.sqrt(mse)),
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


def stationwise_regression_metrics(
    stations: Sequence[str] | pd.Series,
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Calculate regression metrics separately for each station."""

    true_values, predicted_values = validate_regression_arrays(
        y_true,
        y_pred,
    )

    station_values = pd.Series(
        stations,
        dtype="string",
    ).reset_index(drop=True)

    if len(station_values) != len(true_values):
        raise ValueError(
            "Station values and target arrays must have identical lengths."
        )

    evaluation_frame = pd.DataFrame(
        {
            "station": station_values,
            "y_true": true_values,
            "y_pred": predicted_values,
        }
    )

    records = []

    for station, station_frame in evaluation_frame.groupby(
        "station",
        sort=True,
    ):
        metrics = regression_metrics(
            station_frame["y_true"],
            station_frame["y_pred"],
        )

        records.append(
            {
                "station": str(station),
                "samples": len(station_frame),
                **metrics,
            }
        )

    return pd.DataFrame(records)


def pollution_range_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Evaluate errors across different PM2.5 concentration ranges."""

    true_values, predicted_values = validate_regression_arrays(
        y_true,
        y_pred,
    )

    ranges = [
        ("0-35", 0.0, 35.0),
        ("35-75", 35.0, 75.0),
        ("75-150", 75.0, 150.0),
        ("150-250", 150.0, 250.0),
        ("250+", 250.0, np.inf),
    ]

    records = []

    for label, lower, upper in ranges:
        if np.isinf(upper):
            mask = true_values >= lower
        elif lower == 0.0:
            mask = (true_values >= lower) & (true_values <= upper)
        else:
            mask = (true_values > lower) & (true_values <= upper)

        sample_count = int(mask.sum())

        if sample_count == 0:
            continue

        metrics = regression_metrics(
            true_values[mask],
            predicted_values[mask],
        )

        records.append(
            {
                "pollution_range": label,
                "samples": sample_count,
                **metrics,
            }
        )

    return pd.DataFrame(records)
