"""Tests for neural-network architectures."""

from __future__ import annotations

import numpy as np
import pytest

from src.models import (
    build_feedforward_model,
    build_gru_model,
    build_lstm_model,
    set_reproducible_seed,
)


def test_feedforward_input_and_output_shapes() -> None:
    """The model must accept 115 features and output one value."""

    set_reproducible_seed(42)

    model = build_feedforward_model(
        input_dimension=115,
        seed=42,
    )

    assert model.input_shape == (None, 115)
    assert model.output_shape == (None, 1)

    inputs = np.zeros(
        (4, 115),
        dtype=np.float32,
    )

    predictions = model(
        inputs,
        training=False,
    ).numpy()

    assert predictions.shape == (4, 1)
    assert np.isfinite(predictions).all()


def test_feedforward_can_train_on_one_batch() -> None:
    """One training batch must produce a finite result."""

    set_reproducible_seed(42)

    model = build_feedforward_model(
        input_dimension=10,
        seed=42,
    )

    generator = np.random.default_rng(42)

    inputs = generator.normal(
        size=(16, 10)
    ).astype(np.float32)

    targets = generator.normal(
        size=(16,)
    ).astype(np.float32)

    result = model.train_on_batch(
        inputs,
        targets,
    )

    assert np.isfinite(
        np.asarray(result)
    ).all()


def test_feedforward_rejects_invalid_dimension() -> None:
    """A non-positive input dimension must be rejected."""

    with pytest.raises(
        ValueError,
        match="input_dimension",
    ):
        build_feedforward_model(
            input_dimension=0
        )

def test_lstm_input_and_output_shapes() -> None:
    """The LSTM must accept 24 × 43 inputs."""

    set_reproducible_seed(42)

    model = build_lstm_model(
        input_shape=(24, 43),
        seed=42,
    )

    inputs = np.zeros(
        (4, 24, 43),
        dtype=np.float32,
    )

    predictions = model(
        inputs,
        training=False,
    ).numpy()

    assert model.input_shape == (
        None,
        24,
        43,
    )

    assert model.output_shape == (
        None,
        1,
    )

    assert predictions.shape == (
        4,
        1,
    )

    assert np.isfinite(
        predictions
    ).all()


def test_gru_input_and_output_shapes() -> None:
    """The GRU must accept 24 × 43 inputs."""

    set_reproducible_seed(42)

    model = build_gru_model(
        input_shape=(24, 43),
        seed=42,
    )

    inputs = np.zeros(
        (4, 24, 43),
        dtype=np.float32,
    )

    predictions = model(
        inputs,
        training=False,
    ).numpy()

    assert model.input_shape == (
        None,
        24,
        43,
    )

    assert model.output_shape == (
        None,
        1,
    )

    assert predictions.shape == (
        4,
        1,
    )

    assert np.isfinite(
        predictions
    ).all()


@pytest.mark.parametrize(
    "builder",
    [
        build_lstm_model,
        build_gru_model,
    ],
)
def test_recurrent_model_can_train_one_batch(
    builder,
) -> None:
    """Each recurrent model must train on one batch."""

    set_reproducible_seed(42)

    model = builder(
        input_shape=(6, 5),
        units=8,
        dense_units=4,
        seed=42,
    )

    generator = np.random.default_rng(42)

    inputs = generator.normal(
        size=(8, 6, 5)
    ).astype(np.float32)

    targets = generator.normal(
        size=(8, 1)
    ).astype(np.float32)

    result = model.train_on_batch(
        inputs,
        targets,
    )

    assert np.isfinite(
        np.asarray(result)
    ).all()