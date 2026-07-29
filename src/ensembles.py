"""Prediction-alignment and ensemble utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.evaluation import regression_metrics

REQUIRED_PREDICTION_COLUMNS = {
    "sequence_id",
    "station",
    "target_datetime",
    "y_true",
    "y_pred",
}


def align_prediction_frames(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    first_name: str,
    second_name: str,
) -> pd.DataFrame:
    """Align two model-prediction frames by sequence ID."""

    for name, frame in [
        (first_name, first),
        (second_name, second),
    ]:
        missing_columns = sorted(
            REQUIRED_PREDICTION_COLUMNS.difference(
                frame.columns
            )
        )

        if missing_columns:
            raise ValueError(
                f"{name} predictions are missing columns: "
                + ", ".join(missing_columns)
            )

        if frame["sequence_id"].duplicated().any():
            raise ValueError(
                f"{name} contains duplicated sequence IDs."
            )

    first_selected = first[
        [
            "sequence_id",
            "station",
            "target_datetime",
            "y_true",
            "y_pred",
        ]
    ].rename(
        columns={
            "station": "station_first",
            "target_datetime": "target_datetime_first",
            "y_true": "y_true_first",
            "y_pred": f"prediction_{first_name}",
        }
    )

    second_selected = second[
        [
            "sequence_id",
            "station",
            "target_datetime",
            "y_true",
            "y_pred",
        ]
    ].rename(
        columns={
            "station": "station_second",
            "target_datetime": "target_datetime_second",
            "y_true": "y_true_second",
            "y_pred": f"prediction_{second_name}",
        }
    )

    aligned = first_selected.merge(
        second_selected,
        on="sequence_id",
        how="inner",
        validate="one_to_one",
    )

    if len(aligned) != len(first) or len(aligned) != len(second):
        raise ValueError(
            "The two prediction frames do not contain "
            "the same sequence IDs."
        )

    if not aligned["station_first"].equals(
        aligned["station_second"]
    ):
        raise ValueError(
            "Station alignment differs between models."
        )

    first_datetimes = pd.to_datetime(
        aligned["target_datetime_first"]
    )

    second_datetimes = pd.to_datetime(
        aligned["target_datetime_second"]
    )

    if not first_datetimes.equals(second_datetimes):
        raise ValueError(
            "Target datetime alignment differs between models."
        )

    if not np.allclose(
        aligned["y_true_first"].to_numpy(
            dtype=np.float64
        ),
        aligned["y_true_second"].to_numpy(
            dtype=np.float64
        ),
    ):
        raise ValueError(
            "True targets differ between model outputs."
        )

    result = pd.DataFrame(
        {
            "sequence_id": aligned["sequence_id"],
            "station": aligned["station_first"],
            "target_datetime": first_datetimes,
            "y_true": aligned["y_true_first"].to_numpy(
                dtype=np.float32
            ),
            f"prediction_{first_name}": aligned[
                f"prediction_{first_name}"
            ].to_numpy(dtype=np.float32),
            f"prediction_{second_name}": aligned[
                f"prediction_{second_name}"
            ].to_numpy(dtype=np.float32),
        }
    )

    return result.sort_values(
        "sequence_id"
    ).reset_index(drop=True)


def weighted_average_predictions(
    first_predictions: Sequence[float] | np.ndarray,
    second_predictions: Sequence[float] | np.ndarray,
    *,
    first_weight: float,
) -> np.ndarray:
    """Combine two prediction arrays using one convex weight."""

    if not 0.0 <= first_weight <= 1.0:
        raise ValueError(
            "first_weight must be between 0 and 1."
        )

    first_values = np.asarray(
        first_predictions,
        dtype=np.float64,
    ).reshape(-1)

    second_values = np.asarray(
        second_predictions,
        dtype=np.float64,
    ).reshape(-1)

    if first_values.shape != second_values.shape:
        raise ValueError(
            "Prediction arrays must have identical shapes."
        )

    if not np.isfinite(first_values).all():
        raise ValueError(
            "First predictions contain invalid values."
        )

    if not np.isfinite(second_values).all():
        raise ValueError(
            "Second predictions contain invalid values."
        )

    second_weight = 1.0 - first_weight

    predictions = (
        first_weight * first_values
        + second_weight * second_values
    )

    return predictions.astype(np.float32)


def search_two_model_weights(
    y_true: Sequence[float] | np.ndarray,
    first_predictions: Sequence[float] | np.ndarray,
    second_predictions: Sequence[float] | np.ndarray,
    *,
    step: float = 0.01,
) -> pd.DataFrame:
    """Evaluate convex ensemble weights using validation targets."""

    if not 0.0 < step <= 1.0:
        raise ValueError(
            "step must be greater than zero and at most one."
        )

    number_of_intervals = round(1.0 / step)

    weights = np.linspace(
        0.0,
        1.0,
        number_of_intervals + 1,
    )

    records = []

    for first_weight in weights:
        predictions = weighted_average_predictions(
            first_predictions,
            second_predictions,
            first_weight=float(first_weight),
        )

        metrics = regression_metrics(
            y_true,
            predictions,
        )

        records.append(
            {
                "feedforward_weight": float(
                    first_weight
                ),
                "lstm_weight": float(
                    1.0 - first_weight
                ),
                **metrics,
            }
        )

    return pd.DataFrame(records)