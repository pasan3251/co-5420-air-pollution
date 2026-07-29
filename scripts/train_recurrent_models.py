"""Train and evaluate LSTM and GRU forecasting models."""

from __future__ import annotations

import argparse
import gc
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
    clip_pm25_predictions,
)
from src.config import (
    BASELINE_METRICS_PATH,
    FEEDFORWARD_METRICS_PATH,
    FIGURES_DIR,
    GRU_HISTORY_PATH,
    GRU_MODEL_PATH,
    GRU_MODEL_SUMMARY_PATH,
    LSTM_HISTORY_PATH,
    LSTM_MODEL_PATH,
    LSTM_MODEL_SUMMARY_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    PROCESSED_HOURLY_PATH,
    RANDOM_SEED,
    RECURRENT_METRICS_PATH,
    RECURRENT_PREDICTIONS_DIR,
    RECURRENT_RANGE_METRICS_PATH,
    RECURRENT_STATION_METRICS_PATH,
    SEQUENCE_INDEX_PATH,
    WINDOW_SIZE,
)
from src.evaluation import (
    pollution_range_metrics,
    regression_metrics,
    stationwise_regression_metrics,
)
from src.models import (
    build_gru_model,
    build_lstm_model,
    set_reproducible_seed,
)
from src.sequence_builder import (
    filter_sequence_split,
)
from src.temporal_dataset import (
    TemporalWindowDataset,
)

MODEL_BUILDERS = {
    "lstm": build_lstm_model,
    "gru": build_gru_model,
}

MODEL_PATHS = {
    "lstm": LSTM_MODEL_PATH,
    "gru": GRU_MODEL_PATH,
}

HISTORY_PATHS = {
    "lstm": LSTM_HISTORY_PATH,
    "gru": GRU_HISTORY_PATH,
}

SUMMARY_PATHS = {
    "lstm": LSTM_MODEL_SUMMARY_PATH,
    "gru": GRU_MODEL_SUMMARY_PATH,
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train LSTM and GRU PM2.5 models."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lstm", "gru"],
        default=["lstm", "gru"],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--units",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--dense-units",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Use small subsets and two epochs without "
            "saving official result files."
        ),
    )

    return parser.parse_args()


def deterministic_subset(
    sequence_index: pd.DataFrame,
    *,
    sample_count: int,
    seed: int,
) -> pd.DataFrame:
    """Select a deterministic sample for smoke testing."""

    if len(sequence_index) <= sample_count:
        return sequence_index.reset_index(
            drop=True
        )

    return (
        sequence_index
        .sample(
            n=sample_count,
            random_state=seed,
        )
        .sort_values("sequence_id")
        .reset_index(drop=True)
    )


def build_dataset(
    feature_matrix: np.ndarray,
    sequence_index: pd.DataFrame,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    return_targets: bool,
) -> TemporalWindowDataset:
    """Create one recurrent temporal dataset."""

    return TemporalWindowDataset(
        feature_matrix,
        sequence_index,
        window_size=WINDOW_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        return_targets=return_targets,
    )


def save_predictions(
    model_name: str,
    sequence_index: pd.DataFrame,
    predictions: np.ndarray,
    *,
    split: str,
) -> None:
    """Save detailed recurrent predictions."""

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

    model_directory = (
        RECURRENT_PREDICTIONS_DIR
        / model_name
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        model_directory
        / f"{split}_predictions.csv",
        index=False,
    )


def evaluate_model_split(
    model_name: str,
    sequence_index: pd.DataFrame,
    predictions: np.ndarray,
    *,
    split: str,
    epochs_trained: int,
    best_epoch: int,
    parameter_count: int,
    training_seconds: float,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evaluate one recurrent model on one split."""

    y_true = sequence_index[
        "target"
    ].to_numpy(dtype=np.float32)

    predictions = clip_pm25_predictions(
        predictions
    )

    overall = {
        "model": model_name,
        "split": split,
        "samples": len(sequence_index),
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "parameters": parameter_count,
        "training_seconds": training_seconds,
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
        model_name,
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
        model_name,
    )

    return (
        overall,
        station_metrics,
        range_metrics,
    )


def save_history_figure(
    model_name: str,
    history_frame: pd.DataFrame,
) -> None:
    """Save recurrent training and validation RMSE."""

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
        f"{model_name.upper()} Training History"
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("RMSE")
    axis.legend()

    figure.tight_layout()

    figure_number = (
        17 if model_name == "lstm" else 18
    )

    figure.savefig(
        FIGURES_DIR
        / (
            f"{figure_number}_"
            f"{model_name}_training_rmse.png"
        ),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def merge_result_file(
    path: Path,
    new_rows: pd.DataFrame,
    *,
    replaced_models: list[str],
) -> pd.DataFrame:
    """Replace selected model rows while preserving other runs."""

    if path.exists():
        existing = pd.read_csv(path)

        existing = existing.loc[
            ~existing["model"].isin(
                replaced_models
            )
        ]

        combined = pd.concat(
            [existing, new_rows],
            ignore_index=True,
        )

    else:
        combined = new_rows.copy()

    sort_columns = [
        column
        for column in [
            "model",
            "split",
            "station",
            "pollution_range",
        ]
        if column in combined.columns
    ]

    combined = combined.sort_values(
        sort_columns
    ).reset_index(drop=True)

    combined.to_csv(
        path,
        index=False,
    )

    return combined


def save_comparison_figure(
    recurrent_metrics: pd.DataFrame,
) -> None:
    """Compare recurrent models with prior models."""

    metric_frames = [
        pd.read_csv(BASELINE_METRICS_PATH)[
            ["model", "split", "rmse"]
        ],
        pd.read_csv(FEEDFORWARD_METRICS_PATH)[
            ["model", "split", "rmse"]
        ],
        recurrent_metrics[
            ["model", "split", "rmse"]
        ],
    ]

    comparison = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    pivot = comparison.pivot(
        index="model",
        columns="split",
        values="rmse",
    )

    figure, axis = plt.subplots(
        figsize=(13, 6)
    )

    pivot.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(
        "Temporal Neural Networks vs Existing Models"
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
        / "19_recurrent_model_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def train_one_model(
    model_name: str,
    feature_matrix: np.ndarray,
    split_indices: dict[str, pd.DataFrame],
    arguments: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Train and evaluate one recurrent architecture."""

    tf.keras.backend.clear_session()

    set_reproducible_seed(
        arguments.seed,
        deterministic=True,
    )

    model_builder = MODEL_BUILDERS[
        model_name
    ]

    if arguments.smoke_test:
        model_path = (
            MODELS_DIR
            / f"_smoke_{model_name}.keras"
        )
    else:
        model_path = MODEL_PATHS[
            model_name
        ]

    model = model_builder(
        input_shape=(
            WINDOW_SIZE,
            feature_matrix.shape[1],
        ),
        units=arguments.units,
        dense_units=arguments.dense_units,
        dropout_rate=arguments.dropout_rate,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
    )

    print(
        f"\n{model_name.upper()} architecture"
    )
    print("-" * 80)

    model.summary()

    if not arguments.smoke_test:
        summary_lines: list[str] = []

        model.summary(
            print_fn=summary_lines.append
        )

        SUMMARY_PATHS[
            model_name
        ].write_text(
            "\n".join(summary_lines),
            encoding="utf-8",
        )

    train_dataset = build_dataset(
        feature_matrix,
        split_indices["train"],
        batch_size=arguments.batch_size,
        shuffle=True,
        seed=arguments.seed,
        return_targets=True,
    )

    validation_dataset = build_dataset(
        feature_matrix,
        split_indices["validation"],
        batch_size=arguments.batch_size,
        shuffle=False,
        seed=arguments.seed,
        return_targets=True,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
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
            patience=3,
            min_delta=0.01,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    maximum_epochs = (
        2
        if arguments.smoke_test
        else arguments.epochs
    )

    print(
        f"\nTraining {model_name.upper()}..."
    )

    start_time = time.perf_counter()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=maximum_epochs,
        callbacks=callbacks,
        verbose=2,
    )

    training_seconds = (
        time.perf_counter()
        - start_time
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

    best_position = int(
        history_frame["val_rmse"].idxmin()
    )

    best_epoch = int(
        history_frame.loc[
            best_position,
            "epoch",
        ]
    )

    epochs_trained = len(
        history_frame
    )

    print(
        f"\n{model_name.upper()} training time: "
        f"{training_seconds:.2f} seconds"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        "Best validation-history RMSE:",
        f"{history_frame['val_rmse'].min():.4f}",
    )

    best_model = (
        tf.keras.models.load_model(
            model_path
        )
    )

    overall_records = []
    station_frames = []
    range_frames = []

    for split in [
        "validation",
        "local_test",
    ]:
        prediction_dataset = build_dataset(
            feature_matrix,
            split_indices[split],
            batch_size=arguments.batch_size * 2,
            shuffle=False,
            seed=arguments.seed,
            return_targets=False,
        )

        print(
            f"Predicting {split} with "
            f"{model_name.upper()}..."
        )

        predictions = best_model.predict(
            prediction_dataset,
            verbose=0,
        ).reshape(-1)

        predictions = clip_pm25_predictions(
            predictions
        )

        (
            overall,
            station_metrics,
            range_metrics,
        ) = evaluate_model_split(
            model_name,
            split_indices[split],
            predictions,
            split=split,
            epochs_trained=epochs_trained,
            best_epoch=best_epoch,
            parameter_count=(
                best_model.count_params()
            ),
            training_seconds=training_seconds,
        )

        overall_records.append(
            overall
        )

        station_frames.append(
            station_metrics
        )

        range_frames.append(
            range_metrics
        )

        if not arguments.smoke_test:
            save_predictions(
                model_name,
                split_indices[split],
                predictions,
                split=split,
            )

    overall_frame = pd.DataFrame(
        overall_records
    )

    station_frame = pd.concat(
        station_frames,
        ignore_index=True,
    )

    range_frame = pd.concat(
        range_frames,
        ignore_index=True,
    )

    print(
        f"\n{model_name.upper()} metrics"
    )

    print("-" * 100)

    print(
        overall_frame.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    if arguments.smoke_test:
        model_path.unlink(
            missing_ok=True
        )

    else:
        history_frame.to_csv(
            HISTORY_PATHS[model_name],
            index=False,
        )

        save_history_figure(
            model_name,
            history_frame,
        )

    del model
    del best_model
    del train_dataset
    del validation_dataset

    gc.collect()

    return (
        overall_frame,
        station_frame,
        range_frame,
    )


def main() -> None:
    """Run selected recurrent models."""

    arguments = parse_arguments()

    required_paths = [
        PROCESSED_HOURLY_PATH,
        PREPROCESSOR_PATH,
        SEQUENCE_INDEX_PATH,
        BASELINE_METRICS_PATH,
        FEEDFORWARD_METRICS_PATH,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Required project files are missing:\n"
            + "\n".join(missing_paths)
        )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RECURRENT_PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TensorFlow version:",
        tf.__version__,
    )

    print(
        "GPU devices:",
        tf.config.list_physical_devices(
            "GPU"
        ),
    )

    print("\nLoading temporal data...")

    processed = pd.read_parquet(
        PROCESSED_HOURLY_PATH
    )

    processed = (
        processed
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

    feature_matrix = processed[
        feature_columns
    ].to_numpy(dtype=np.float32)

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

    if arguments.smoke_test:
        print(
            "\nSMOKE TEST MODE: using reduced datasets"
        )

        split_indices["train"] = (
            deterministic_subset(
                split_indices["train"],
                sample_count=4_096,
                seed=arguments.seed,
            )
        )

        split_indices["validation"] = (
            deterministic_subset(
                split_indices["validation"],
                sample_count=1_024,
                seed=arguments.seed,
            )
        )

        split_indices["local_test"] = (
            deterministic_subset(
                split_indices["local_test"],
                sample_count=1_024,
                seed=arguments.seed,
            )
        )

    print("\nSequence counts")
    print("-" * 60)

    for split, current_index in (
        split_indices.items()
    ):
        print(
            f"{split:<12}: "
            f"{len(current_index):>8,}"
        )

    overall_frames = []
    station_frames = []
    range_frames = []

    for model_name in arguments.models:
        (
            overall_frame,
            station_frame,
            range_frame,
        ) = train_one_model(
            model_name,
            feature_matrix,
            split_indices,
            arguments,
        )

        overall_frames.append(
            overall_frame
        )

        station_frames.append(
            station_frame
        )

        range_frames.append(
            range_frame
        )

    if arguments.smoke_test:
        print("\n" + "=" * 80)
        print("RECURRENT SMOKE TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        return

    new_overall = pd.concat(
        overall_frames,
        ignore_index=True,
    )

    new_station = pd.concat(
        station_frames,
        ignore_index=True,
    )

    new_range = pd.concat(
        range_frames,
        ignore_index=True,
    )

    complete_overall = merge_result_file(
        RECURRENT_METRICS_PATH,
        new_overall,
        replaced_models=arguments.models,
    )

    merge_result_file(
        RECURRENT_STATION_METRICS_PATH,
        new_station,
        replaced_models=arguments.models,
    )

    merge_result_file(
        RECURRENT_RANGE_METRICS_PATH,
        new_range,
        replaced_models=arguments.models,
    )

    save_comparison_figure(
        complete_overall
    )

    print("\nComplete recurrent comparison")
    print("-" * 110)

    print(
        complete_overall.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nCurrent feedforward benchmark")
    print("-" * 60)
    print("Validation RMSE: 15.7176")
    print("Local-test RMSE: 25.2410")

    print("\n" + "=" * 80)
    print("RECURRENT TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()