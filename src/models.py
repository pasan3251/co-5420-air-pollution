"""Neural-network architectures for PM2.5 forecasting."""

from __future__ import annotations

import tensorflow as tf


def set_reproducible_seed(
    seed: int,
    *,
    deterministic: bool = True,
) -> None:
    """
    Configure Python, NumPy and TensorFlow random seeds.

    Deterministic execution improves repeatability when the same
    software and hardware configuration is used.
    """

    tf.keras.utils.set_random_seed(seed)

    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            # Some TensorFlow builds or already-initialised devices may
            # not permit this setting. The random seed still applies.
            pass


def build_feedforward_model(
    input_dimension: int,
    *,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.20,
    l2_strength: float = 1e-5,
    seed: int = 42,
) -> tf.keras.Model:
    """Build and compile the tabular feedforward baseline."""

    if input_dimension <= 0:
        raise ValueError(
            "input_dimension must be positive."
        )

    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(
            "dropout_rate must be between 0 and 1."
        )

    if learning_rate <= 0.0:
        raise ValueError(
            "learning_rate must be positive."
        )

    regularizer = tf.keras.regularizers.L2(
        l2_strength
    )

    inputs = tf.keras.Input(
        shape=(input_dimension,),
        dtype=tf.float32,
        name="tabular_features",
    )

    hidden = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_initializer=(
            tf.keras.initializers.HeNormal(
                seed=seed,
            )
        ),
        kernel_regularizer=regularizer,
        name="dense_128",
    )(inputs)

    hidden = tf.keras.layers.Dropout(
        dropout_rate,
        seed=seed + 1,
        name="dropout_128",
    )(hidden)

    hidden = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_initializer=(
            tf.keras.initializers.HeNormal(
                seed=seed + 2,
            )
        ),
        kernel_regularizer=regularizer,
        name="dense_64",
    )(hidden)

    hidden = tf.keras.layers.Dropout(
        dropout_rate * 0.75,
        seed=seed + 3,
        name="dropout_64",
    )(hidden)

    hidden = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_initializer=(
            tf.keras.initializers.HeNormal(
                seed=seed + 4,
            )
        ),
        kernel_regularizer=regularizer,
        name="dense_32",
    )(hidden)

    outputs = tf.keras.layers.Dense(
        1,
        activation="linear",
        kernel_initializer=(
            tf.keras.initializers.GlorotUniform(
                seed=seed + 5,
            )
        ),
        name="pm25_prediction",
    )(hidden)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="feedforward_pm25_forecaster",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(
                name="rmse"
            ),
            tf.keras.metrics.MeanAbsoluteError(
                name="mae"
            ),
        ],
    )

    return model