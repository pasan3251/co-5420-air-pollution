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
    input_shape: tuple[int, ...],
    *,
    num_classes: int = 4,
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
        shape=input_shape,
        dtype=tf.float32,
        name="tabular_features",
    )

    hidden = tf.keras.layers.Flatten()(inputs)

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
        num_classes,
        activation="softmax",
        kernel_initializer=(
            tf.keras.initializers.GlorotUniform(
                seed=seed + 5,
            )
        ),
        name="aqi_class_prediction",
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
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            "accuracy"
        ],
    )

    return model


def _compile_recurrent_model(
    model: tf.keras.Model,
    *,
    learning_rate: float,
) -> tf.keras.Model:
    """Compile a recurrent regression model."""

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            "accuracy"
        ],
    )

    return model


def _validate_recurrent_arguments(
    input_shape: tuple[int, int],
    units: int,
    dense_units: int,
    dropout_rate: float,
    learning_rate: float,
) -> None:
    """Validate common recurrent-model arguments."""

    if len(input_shape) != 2:
        raise ValueError(
            "input_shape must contain time steps and features."
        )

    if input_shape[0] <= 0 or input_shape[1] <= 0:
        raise ValueError(
            "input_shape dimensions must be positive."
        )

    if units <= 0:
        raise ValueError(
            "units must be positive."
        )

    if dense_units <= 0:
        raise ValueError(
            "dense_units must be positive."
        )

    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(
            "dropout_rate must be between 0 and 1."
        )

    if learning_rate <= 0.0:
        raise ValueError(
            "learning_rate must be positive."
        )


def build_lstm_model(
    input_shape: tuple[int, int],
    *,
    num_classes: int = 4,
    units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.20,
    learning_rate: float = 1e-3,
    l2_strength: float = 1e-5,
    seed: int = 42,
) -> tf.keras.Model:
    """Build the initial LSTM forecasting model."""

    _validate_recurrent_arguments(
        input_shape,
        units,
        dense_units,
        dropout_rate,
        learning_rate,
    )

    regularizer = tf.keras.regularizers.L2(
        l2_strength
    )

    inputs = tf.keras.Input(
        shape=input_shape,
        dtype=tf.float32,
        name="temporal_features",
    )

    hidden = tf.keras.layers.LSTM(
        units,
        return_sequences=False,
        kernel_initializer=(
            tf.keras.initializers.GlorotUniform(
                seed=seed,
            )
        ),
        recurrent_initializer=(
            tf.keras.initializers.Orthogonal(
                seed=seed + 1,
            )
        ),
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
        name="lstm_64",
    )(inputs)

    hidden = tf.keras.layers.LayerNormalization(
        name="lstm_layer_normalisation",
    )(hidden)

    hidden = tf.keras.layers.Dropout(
        dropout_rate,
        seed=seed + 2,
        name="lstm_dropout",
    )(hidden)

    hidden = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_initializer=(
            tf.keras.initializers.HeNormal(
                seed=seed + 3,
            )
        ),
        kernel_regularizer=regularizer,
        name="lstm_dense_32",
    )(hidden)

    hidden = tf.keras.layers.Dropout(
        dropout_rate * 0.5,
        seed=seed + 4,
        name="lstm_dense_dropout",
    )(hidden)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer=(
            tf.keras.initializers.GlorotUniform(
                seed=seed + 5,
            )
        ),
        name="aqi_class_prediction",
    )(hidden)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="lstm_pm25_forecaster",
    )

    return _compile_recurrent_model(
        model,
        learning_rate=learning_rate,
    )


def build_gru_model(
    input_shape: tuple[int, int],
    *,
    num_classes: int = 4,
    units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.20,
    learning_rate: float = 1e-3,
    l2_strength: float = 1e-5,
    seed: int = 42,
) -> tf.keras.Model:
    """Build the initial GRU forecasting model."""

    _validate_recurrent_arguments(
        input_shape,
        units,
        dense_units,
        dropout_rate,
        learning_rate,
    )

    regularizer = tf.keras.regularizers.L2(
        l2_strength
    )

    inputs = tf.keras.Input(
        shape=input_shape,
        dtype=tf.float32,
        name="temporal_features",
    )

    hidden = tf.keras.layers.GRU(
        units,
        return_sequences=False,
        reset_after=True,
        kernel_initializer=(
            tf.keras.initializers.GlorotUniform(
                seed=seed,
            )
        ),
        recurrent_initializer=(
            tf.keras.initializers.Orthogonal(
                seed=seed + 1,
            )
        ),
        kernel_regularizer=regularizer,
        recurrent_regularizer=regularizer,
        name="gru_64",
    )(inputs)

    hidden = tf.keras.layers.LayerNormalization(
        name="gru_layer_normalisation",
    )(hidden)

    hidden = tf.keras.layers.Dropout(
        dropout_rate,
        seed=seed + 2,
        name="gru_dropout",
    )(hidden)

    hidden = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_initializer=(
            tf.keras.initializers.HeNormal(
                seed=seed + 3,
            )
        ),
        kernel_regularizer=regularizer,
        name="gru_dense_32",
    )(hidden)

    hidden = tf.keras.layers.Dropout(
        dropout_rate * 0.5,
        seed=seed + 4,
        name="gru_dense_dropout",
    )(hidden)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer=(
            tf.keras.initializers.GlorotUniform(
                seed=seed + 5,
            )
        ),
        name="aqi_class_prediction",
    )(hidden)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="gru_pm25_forecaster",
    )

    return _compile_recurrent_model(
        model,
        learning_rate=learning_rate,
    )