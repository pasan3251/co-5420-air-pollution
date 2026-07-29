"""Tests for recurrent temporal batch generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.temporal_dataset import (
    TemporalWindowDataset,
)


def make_temporal_data() -> tuple[
    np.ndarray,
    pd.DataFrame,
]:
    """Create a small temporal feature matrix and index."""

    feature_matrix = np.arange(
        36,
        dtype=np.float32,
    ).reshape(12, 3)

    sequence_index = pd.DataFrame(
        {
            "window_start_row": [0, 2, 4],
            "window_end_row": [3, 5, 7],
            "target": [10.0, 20.0, 30.0],
        }
    )

    return feature_matrix, sequence_index


def test_temporal_dataset_shapes() -> None:
    """The dataset must return 3D inputs and 2D targets."""

    feature_matrix, sequence_index = (
        make_temporal_data()
    )

    dataset = TemporalWindowDataset(
        feature_matrix,
        sequence_index,
        window_size=4,
        batch_size=2,
        shuffle=False,
    )

    inputs, targets = dataset[0]

    assert inputs.shape == (2, 4, 3)
    assert targets.shape == (2, 1)
    assert inputs.dtype == np.float32
    assert targets.dtype == np.float32


def test_temporal_dataset_values_are_aligned() -> None:
    """The first batch must use the indexed source rows."""

    feature_matrix, sequence_index = (
        make_temporal_data()
    )

    dataset = TemporalWindowDataset(
        feature_matrix,
        sequence_index,
        window_size=4,
        batch_size=2,
        shuffle=False,
    )

    inputs, targets = dataset[0]

    assert np.allclose(
        inputs[0],
        feature_matrix[0:4],
    )

    assert np.allclose(
        inputs[1],
        feature_matrix[2:6],
    )

    assert np.allclose(
        targets[:, 0],
        [10.0, 20.0],
    )


def test_temporal_prediction_mode() -> None:
    """Prediction datasets must return only model inputs."""

    feature_matrix, sequence_index = (
        make_temporal_data()
    )

    dataset = TemporalWindowDataset(
        feature_matrix,
        sequence_index,
        window_size=4,
        batch_size=2,
        shuffle=False,
        return_targets=False,
    )

    inputs = dataset[0]

    assert isinstance(inputs, np.ndarray)
    assert inputs.shape == (2, 4, 3)


def test_invalid_window_length_is_rejected() -> None:
    """Incorrectly indexed windows must be rejected."""

    feature_matrix, sequence_index = (
        make_temporal_data()
    )

    sequence_index.loc[
        0,
        "window_end_row",
    ] = 4

    with pytest.raises(
        ValueError,
        match="unexpected length",
    ):
        TemporalWindowDataset(
            feature_matrix,
            sequence_index,
            window_size=4,
        )