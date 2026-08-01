"""Train and evaluate the feedforward neural-network baseline."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "2",
)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.baselines import (
    build_tabular_baseline_features,
    clip_pm25_predictions,
)
from src.config import (
    BASELINE_FEATURE_NAMES_PATH,
    BASELINE_METRICS_PATH,
    FEEDFORWARD_HISTORY_PATH,
    FEEDFORWARD_METRICS_PATH,
    FEEDFORWARD_MODEL_PATH,
    FEEDFORWARD_MODEL_SUMMARY_PATH,
    FEEDFORWARD_PREDICTIONS_DIR,
    FEEDFORWARD_RANGE_METRICS_PATH,
    FEEDFORWARD_STATION_METRICS_PATH,
    FIGURES_DIR,
    PREPROCESSOR_PATH,
    PROCESSED_HOURLY_PATH,
    RANDOM_SEED,
    SEQUENCE_INDEX_PATH,
)
from src.evaluation import (
    pollution_range_metrics,
    regression_metrics,
    stationwise_regression_metrics,
)
from src.models import (
    build_feedforward_model,
    set_reproducible_seed,
)
from src.sequence_builder import (
    filter_sequence_split,
)

MODEL_NAME = "feedforward_nn"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the tabular feedforward PM2.5 model."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Validate training arguments."""

    if arguments.epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    if arguments.batch_size <= 0:
        raise ValueError(
            "batch-size must be positive."
        )

    if arguments.patience <= 0:
        raise ValueError(
            "patience must be positive."
        )


def save_predictions(
    sequence_index: pd.DataFrame,
    predictions: np.ndarray,
    *,
    split: str,
) -> None:
    """Save predictions for detailed later analysis."""

    output = sequence_index[
        [
            "sequence_id",
            "station",
            "target_datetime",
            "target",
        ]
    ].copy()

    output = output.rename(
        columns={"target": "y_true"}
    )

    output["y_pred"] = predictions

    output["residual"] = (
        output["y_true"]
        - output["y_pred"]
    )

    output.to_csv(
        FEEDFORWARD_PREDICTIONS_DIR
        / f"{split}_predictions.csv",
        index=False,
    )


def evaluate_split(
    sequence_index: pd.DataFrame,
    predictions: np.ndarray,
    *,
    split: str,
    epochs_trained: int,
    best_epoch: int,
    parameter_count: int,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evaluate one prediction split."""

    y_true = sequence_index[
        "target"
    ].to_numpy(dtype=np.float32)

    predictions = clip_pm25_predictions(
        predictions
    )

    overall = {
        "model": MODEL_NAME,
        "split": split,
        "samples": len(sequence_index),
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "parameters": parameter_count,
        **regression_metrics(
            y_true,
            predictions,
        ),
    }

    station_metrics = (
        stationwise_regression_metrics(
            sequence_index["station"],
            y_true,
            predictions,
        )
    )

    station_metrics.insert(
        0,
        "split",
        split,
    )

    station_metrics.insert(
        0,
        "model",
        MODEL_NAME,
    )

    range_metrics = pollution_range_metrics(
        y_true,
        predictions,
    )

    range_metrics.insert(
        0,
        "split",
        split,
    )

    range_metrics.insert(
        0,
        "model",
        MODEL_NAME,
    )

    return (
        overall,
        station_metrics,
        range_metrics,
    )


def save_training_figures(
    history_frame: pd.DataFrame,
) -> None:
    """Save loss and RMSE training curves."""

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        history_frame["epoch"],
        history_frame["loss"],
        label="Training loss",
    )

    axis.plot(
        history_frame["epoch"],
        history_frame["val_loss"],
        label="Validation loss",
    )

    axis.set_title(
        "Feedforward Neural Network: MSE Loss"
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error")
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "13_feedforward_training_loss.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        history_frame["epoch"],
        history_frame["rmse"],
        label="Training RMSE",
    )

    axis.plot(
        history_frame["epoch"],
        history_frame["val_rmse"],
        label="Validation RMSE",
    )

    axis.set_title(
        "Feedforward Neural Network: RMSE"
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("RMSE")
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "14_feedforward_training_rmse.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_comparison_figure(
    feedforward_metrics: pd.DataFrame,
) -> None:
    """Compare the feedforward model against baselines."""

    baseline_metrics = pd.read_csv(
        BASELINE_METRICS_PATH
    )

    comparison_columns = [
        "model",
        "split",
        "rmse",
    ]

    comparison = pd.concat(
        [
            baseline_metrics[
                comparison_columns
            ],
            feedforward_metrics[
                comparison_columns
            ],
        ],
        ignore_index=True,
    )

    pivot = comparison.pivot(
        index="model",
        columns="split",
        values="rmse",
    )

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    pivot.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(
        "Feedforward Network vs Forecasting Baselines"
    )
    axis.set_xlabel("Model")
    axis.set_ylabel("RMSE")
    axis.tick_params(
        axis="x",
        rotation=25,
    )
    axis.legend(
        title="Split"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "15_feedforward_baseline_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_scatter_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    seed: int,
) -> None:
    """Save a sampled local-test actual-versus-predicted plot."""

    generator = np.random.default_rng(seed)

    sample_size = min(
        5_000,
        len(y_true),
    )

    sample_positions = generator.choice(
        len(y_true),
        size=sample_size,
        replace=False,
    )

    sampled_true = y_true[
        sample_positions
    ]

    sampled_predicted = y_pred[
        sample_positions
    ]

    lower_limit = 0.0

    upper_limit = float(
        max(
            sampled_true.max(),
            sampled_predicted.max(),
        )
    )

    figure, axis = plt.subplots(
        figsize=(8, 8)
    )

    axis.scatter(
        sampled_true,
        sampled_predicted,
        alpha=0.25,
        s=10,
    )

    axis.plot(
        [lower_limit, upper_limit],
        [lower_limit, upper_limit],
        linestyle="--",
        label="Ideal prediction",
    )

    axis.set_title(
        "Feedforward Predictions on Local Test"
    )
    axis.set_xlabel("Actual PM2.5")
    axis.set_ylabel("Predicted PM2.5")
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "16_feedforward_local_test_scatter.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    """Train and evaluate the feedforward model."""

    arguments = parse_arguments()
    validate_arguments(arguments)

    required_paths = [
        PROCESSED_HOURLY_PATH,
        PREPROCESSOR_PATH,
        SEQUENCE_INDEX_PATH,
        BASELINE_FEATURE_NAMES_PATH,
        BASELINE_METRICS_PATH,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Required files are missing:\n"
            + "\n".join(missing_paths)
        )

    set_reproducible_seed(
        arguments.seed,
        deterministic=True,
    )

    FEEDFORWARD_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FEEDFORWARD_PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("TensorFlow version:", tf.__version__)

    gpu_devices = tf.config.list_physical_devices(
        "GPU"
    )

    print("GPU devices:", gpu_devices)

    print("\nLoading processed data...")

    frame = pd.read_parquet(
        PROCESSED_HOURLY_PATH
    )

    frame = (
        frame
        .sort_values(
            ["station", "datetime"]
        )
        .reset_index(drop=True)
    )

    sequence_index = pd.read_parquet(
        SEQUENCE_INDEX_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    feature_columns = (
        preprocessor.get_feature_names_out()
    )

    split_indices = {
        split: filter_sequence_split(
            sequence_index,
            split,
        )
        for split in [
            "train",
            "validation",
            "local_test",
        ]
    }

    feature_matrices = {}
    derived_feature_names: list[str] = []

    for split in [
        "train",
        "validation",
        "local_test",
    ]:
        print(
            f"Building {split} tabular features..."
        )

        (
            current_features,
            current_names,
        ) = build_tabular_baseline_features(
            frame,
            split_indices[split],
            feature_columns,
        )

        feature_matrices[split] = (
            current_features
        )

        if not derived_feature_names:
            derived_feature_names = (
                current_names
            )
        elif current_names != derived_feature_names:
            raise RuntimeError(
                "Feature order changed between splits."
            )

        print(
            f"{split} shape: "
            f"{current_features.shape}"
        )

    with BASELINE_FEATURE_NAMES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        baseline_feature_metadata = json.load(
            file
        )

    if (
        derived_feature_names
        != baseline_feature_metadata[
            "feature_names"
        ]
    ):
        raise ValueError(
            "The feedforward feature order does not "
            "match the baseline feature order."
        )

    targets = {
        split: split_indices[
            split
        ]["target"].to_numpy(
            dtype=np.float32
        )
        for split in split_indices
    }

    input_dimension = (
        feature_matrices["train"].shape[1]
    )

    model = build_feedforward_model(
        input_dimension=input_dimension,
        learning_rate=arguments.learning_rate,
        dropout_rate=arguments.dropout_rate,
        seed=arguments.seed,
    )

    summary_lines: list[str] = []

    model.summary(
        print_fn=summary_lines.append
    )

    FEEDFORWARD_MODEL_SUMMARY_PATH.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\nModel architecture")
    print("-" * 70)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                FEEDFORWARD_MODEL_PATH
            ),
            monitor="val_rmse",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_rmse",
            mode="min",
            patience=arguments.patience,
            min_delta=0.01,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_rmse",
            mode="min",
            factor=0.5,
            patience=4,
            min_delta=0.01,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    print("\nTraining feedforward model...")

    start_time = time.perf_counter()

    history = model.fit(
        feature_matrices["train"],
        targets["train"],
        validation_data=(
            feature_matrices["validation"],
            targets["validation"],
        ),
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    training_seconds = (
        time.perf_counter() - start_time
    )

    history_frame = pd.DataFrame(
        history.history
    )

    history_frame.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(history_frame) + 1,
        ),
    )

    history_frame.to_csv(
        FEEDFORWARD_HISTORY_PATH,
        index=False,
    )

    best_history_position = int(
        history_frame["val_rmse"].idxmin()
    )

    best_epoch = int(
        history_frame.loc[
            best_history_position,
            "epoch",
        ]
    )

    epochs_trained = len(history_frame)

    print(
        f"\nTraining completed in "
        f"{training_seconds:.2f} seconds."
    )

    print(
        f"Epochs trained: {epochs_trained}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        "Best training-history validation RMSE:",
        f"{history_frame['val_rmse'].min():.4f}",
    )

    # Load the checkpointed model rather than assuming the final
    # in-memory epoch is the best saved epoch.
    best_model = tf.keras.models.load_model(
        FEEDFORWARD_MODEL_PATH
    )

    del feature_matrices["train"]
    del targets["train"]
    gc.collect()

    predictions = {}

    for split in [
        "validation",
        "local_test",
    ]:
        print(
            f"Predicting {split}..."
        )

        raw_predictions = best_model.predict(
            feature_matrices[split],
            batch_size=1_024,
            verbose=0,
        ).reshape(-1)

        predictions[split] = (
            clip_pm25_predictions(
                raw_predictions
            )
        )

        save_predictions(
            split_indices[split],
            predictions[split],
            split=split,
        )

    overall_records = []
    station_frames = []
    range_frames = []

    for split in [
        "validation",
        "local_test",
    ]:
        (
            overall,
            station_metrics,
            range_metrics,
        ) = evaluate_split(
            split_indices[split],
            predictions[split],
            split=split,
            epochs_trained=epochs_trained,
            best_epoch=best_epoch,
            parameter_count=best_model.count_params(),
        )

        overall_records.append(overall)
        station_frames.append(station_metrics)
        range_frames.append(range_metrics)

    metrics_frame = pd.DataFrame(
        overall_records
    )

    station_metrics_frame = pd.concat(
        station_frames,
        ignore_index=True,
    )

    range_metrics_frame = pd.concat(
        range_frames,
        ignore_index=True,
    )

    metrics_frame.to_csv(
        FEEDFORWARD_METRICS_PATH,
        index=False,
    )

    station_metrics_frame.to_csv(
        FEEDFORWARD_STATION_METRICS_PATH,
        index=False,
    )

    range_metrics_frame.to_csv(
        FEEDFORWARD_RANGE_METRICS_PATH,
        index=False,
    )

    save_training_figures(
        history_frame
    )

    save_comparison_figure(
        metrics_frame
    )

    save_scatter_figure(
        targets["local_test"],
        predictions["local_test"],
        seed=arguments.seed,
    )

    print("\nFeedforward metrics")
    print("-" * 100)

    print(
        metrics_frame.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    validation_row = metrics_frame.loc[
        metrics_frame["split"]
        == "validation"
    ].iloc[0]

    local_test_row = metrics_frame.loc[
        metrics_frame["split"]
        == "local_test"
    ].iloc[0]

    print("\nBenchmark comparison")
    print("-" * 70)

    print(
        "Best baseline validation RMSE: 16.7294"
    )

    print(
        "Feedforward validation RMSE:",
        f"{validation_row['rmse']:.4f}",
    )

    print(
        "Best stable local-test baseline RMSE: 26.0475"
    )

    print(
        "Feedforward local-test RMSE:",
        f"{local_test_row['rmse']:.4f}",
    )

    print("\n" + "=" * 80)
    print("FEEDFORWARD TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()