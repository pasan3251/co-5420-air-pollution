"""Tests for forecasting baseline utilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.baselines import (
    build_tabular_baseline_features,
    clip_pm25_predictions,
    historical_mean_predictions,
    inverse_standardised_column,
    persistence_predictions,
)
from src.evaluation import (
    regression_metrics,
    stationwise_regression_metrics,
)


def make_baseline_frame() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create a small processed sequence dataset."""

    frame = pd.DataFrame(
        {
            "PM2.5": np.arange(
                8,
                dtype=np.float32,
            ),
            "TEMP": np.arange(
                10,
                18,
                dtype=np.float32,
            ),
            "station_Test": np.ones(
                8,
                dtype=np.float32,
            ),
        }
    )

    sequence_index = pd.DataFrame(
        {
            "window_start_row": [0, 1],
            "window_end_row": [3, 4],
            "target": [4.0, 5.0],
            "station": ["Test", "Test"],
        }
    )

    return frame, sequence_index


def make_dummy_preprocessor() -> SimpleNamespace:
    """Create a fitted scaler compatible with baseline helpers."""

    training_values = np.array(
        [
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
        ]
    )

    scaler = StandardScaler().fit(training_values)

    return SimpleNamespace(
        numeric_columns=(
            "PM2.5",
            "TEMP",
        ),
        scaler_=scaler,
    )


def test_inverse_standardised_column() -> None:
    """Standardised values must return to original units."""

    preprocessor = make_dummy_preprocessor()

    raw_values = np.array([10.0, 20.0, 30.0])

    pm25_index = 0

    scaled_values = (
        raw_values - preprocessor.scaler_.mean_[pm25_index]
    ) / preprocessor.scaler_.scale_[pm25_index]

    restored = inverse_standardised_column(
        scaled_values,
        preprocessor,
        "PM2.5",
    )

    assert np.allclose(
        restored,
        raw_values,
    )


def test_persistence_uses_latest_value() -> None:
    """Persistence must use the last input value."""

    history = np.array(
        [
            [1.0, 2.0, 3.0],
            [5.0, 7.0, 9.0],
        ]
    )

    result = persistence_predictions(history)

    assert np.allclose(
        result,
        [3.0, 9.0],
    )


def test_historical_mean_prediction() -> None:
    """Historical mean must average the complete window."""

    history = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 6.0, 9.0],
        ]
    )

    result = historical_mean_predictions(history)

    assert np.allclose(
        result,
        [2.0, 6.0],
    )


def test_tabular_feature_shape_and_names() -> None:
    """Generated matrix and names must align."""

    frame, sequence_index = make_baseline_frame()

    features, feature_names = build_tabular_baseline_features(
        frame,
        sequence_index,
        [
            "PM2.5",
            "TEMP",
            "station_Test",
        ],
        numeric_columns=[
            "PM2.5",
            "TEMP",
        ],
        window_size=4,
        delta_hours=[
            1,
            3,
        ],
    )

    expected_feature_count = 3 + 4 + 2 * 4 + 2

    assert features.shape == (
        2,
        expected_feature_count,
    )

    assert len(feature_names) == (expected_feature_count)


def test_pm25_lags_are_aligned() -> None:
    """Oldest and latest PM2.5 lags must be correct."""

    frame, sequence_index = make_baseline_frame()

    features, feature_names = build_tabular_baseline_features(
        frame,
        sequence_index,
        [
            "PM2.5",
            "TEMP",
            "station_Test",
        ],
        numeric_columns=[
            "PM2.5",
            "TEMP",
        ],
        window_size=4,
        delta_hours=[],
    )

    lag_4_index = feature_names.index("PM2.5_lag_4")

    lag_1_index = feature_names.index("PM2.5_lag_1")

    assert features[0, lag_4_index] == 0.0
    assert features[0, lag_1_index] == 3.0


def test_prediction_clipping() -> None:
    """Negative PM2.5 predictions must become zero."""

    result = clip_pm25_predictions(np.array([-4.0, 0.0, 12.0]))

    assert np.allclose(
        result,
        [0.0, 0.0, 12.0],
    )


def test_regression_metrics() -> None:
    """Regression metrics must match known values."""

    result = regression_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 4.0]),
    )

    assert np.isclose(
        result["rmse"],
        np.sqrt(1.0 / 3.0),
    )

    assert np.isclose(
        result["mae"],
        1.0 / 3.0,
    )


def test_stationwise_metrics() -> None:
    """Each station must receive an independent result row."""

    result = stationwise_regression_metrics(
        ["A", "A", "B", "B"],
        [1.0, 2.0, 5.0, 7.0],
        [1.0, 3.0, 4.0, 7.0],
    )

    assert result["station"].tolist() == [
        "A",
        "B",
    ]

    assert result["samples"].tolist() == [
        2,
        2,
    ]
