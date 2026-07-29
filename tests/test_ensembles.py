"""Tests for prediction ensembling utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ensembles import (
    align_prediction_frames,
    search_two_model_weights,
    weighted_average_predictions,
)


def make_prediction_frame(
    predictions: list[float],
) -> pd.DataFrame:
    """Create an aligned prediction fixture."""

    return pd.DataFrame(
        {
            "sequence_id": [1, 2, 3],
            "station": ["A", "A", "B"],
            "target_datetime": pd.to_datetime(
                [
                    "2015-01-01 00:00:00",
                    "2015-01-01 01:00:00",
                    "2015-01-01 02:00:00",
                ]
            ),
            "y_true": [10.0, 20.0, 30.0],
            "y_pred": predictions,
        }
    )


def test_weighted_average_predictions() -> None:
    """Convex averaging must use both model outputs."""

    result = weighted_average_predictions(
        [10.0, 20.0],
        [20.0, 40.0],
        first_weight=0.75,
    )

    assert np.allclose(
        result,
        [12.5, 25.0],
    )


def test_invalid_weight_is_rejected() -> None:
    """Weights outside zero to one must be rejected."""

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        weighted_average_predictions(
            [1.0],
            [2.0],
            first_weight=1.2,
        )


def test_prediction_frames_align() -> None:
    """Model outputs must align by sequence ID."""

    first = make_prediction_frame(
        [11.0, 19.0, 31.0]
    )

    second = make_prediction_frame(
        [10.5, 20.5, 29.0]
    )

    result = align_prediction_frames(
        first,
        second,
        first_name="first",
        second_name="second",
    )

    assert len(result) == 3

    assert result.columns.tolist() == [
        "sequence_id",
        "station",
        "target_datetime",
        "y_true",
        "prediction_first",
        "prediction_second",
    ]


def test_weight_search_can_select_first_model() -> None:
    """The exact first model should receive full weight."""

    y_true = np.array(
        [0.0, 10.0, 20.0]
    )

    first = y_true.copy()

    second = np.array(
        [20.0, 10.0, 0.0]
    )

    results = search_two_model_weights(
        y_true,
        first,
        second,
        step=0.1,
    )

    best = results.sort_values(
        "rmse"
    ).iloc[0]

    assert np.isclose(
        best["feedforward_weight"],
        1.0,
    )

    assert np.isclose(
        best["rmse"],
        0.0,
    )