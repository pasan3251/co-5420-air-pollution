"""Run controlled temporal window and weather-feature experiments."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from itertools import product
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

from src.ablation import (
    resize_sequence_windows,
    select_temporal_features,
)
from src.baselines import (
    clip_pm25_predictions,
)
from src.config import (
    PREPROCESSOR_PATH,
    PROCESSED_HOURLY_PATH,
    RANDOM_SEED,
    SEQUENCE_INDEX_PATH,
    TEMPORAL_ABLATION_FIGURE_PATH,
    TEMPORAL_ABLATION_HISTORY_DIR,
    TEMPORAL_ABLATION_MODEL_DIR,
    TEMPORAL_ABLATION_RESULTS_PATH,
    TEMPORAL_ABLATION_SUMMARY_PATH,
)
from src.evaluation import (
    regression_metrics,
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


def parse_arguments() -> argparse.Namespace:
    """Parse experiment arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run temporal window and weather-feature ablations."
        )
    )

    parser.add_argument(
        "--grid",
        action="store_true",
        help=(
            "Run the standard 6/12/24-hour by "
            "pollution/all-feature experiment grid."
        ),
    )

    parser.add_argument(
        "--model",
        choices=[
            "lstm",
            "gru",
        ],
        default="gru",
    )

    parser.add_argument(
        "--window-sizes",
        nargs="+",
        type=int,
        default=[24],
    )

    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=[
            "pollution_only",
            "all_features",
        ],
        default=["all_features"],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[RANDOM_SEED],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
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
        default=6,
    )

    parser.add_argument(
        "--evaluate-local-test",
        action="store_true",
        help=(
            "Evaluate local test only after the final "
            "configuration has been frozen."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun experiments that already exist.",
    )

    return parser.parse_args()


def create_dataset(
    feature_matrix: np.ndarray,
    sequence_index: pd.DataFrame,
    *,
    window_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    return_targets: bool,
) -> TemporalWindowDataset:
    """Create one temporal model dataset."""

    return TemporalWindowDataset(
        feature_matrix,
        sequence_index,
        window_size=window_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        return_targets=return_targets,
    )


def make_run_id(
    *,
    model: str,
    window_size: int,
    feature_set: str,
    seed: int,
) -> str:
    """Construct a stable experiment identifier."""

    return (
        f"{model}_"
        f"w{window_size}_"
        f"{feature_set}_"
        f"s{seed}"
    )


def load_existing_results() -> pd.DataFrame:
    """Load previous ablation results when available."""

    if TEMPORAL_ABLATION_RESULTS_PATH.exists():
        return pd.read_csv(
            TEMPORAL_ABLATION_RESULTS_PATH
        )

    return pd.DataFrame()


def should_skip_run(
    existing_results: pd.DataFrame,
    run_id: str,
    *,
    evaluate_local_test: bool,
    force: bool,
) -> bool:
    """Determine whether an existing run is complete."""

    if force or existing_results.empty:
        return False

    matching = existing_results.loc[
        existing_results["run_id"] == run_id
    ]

    if matching.empty:
        return False

    if not evaluate_local_test:
        return True

    return matching[
        "local_test_rmse"
    ].notna().all()


def save_result(
    result: dict[str, object],
) -> pd.DataFrame:
    """Insert or replace one experiment result."""

    existing = load_existing_results()

    if not existing.empty:
        existing = existing.loc[
            existing["run_id"]
            != result["run_id"]
        ]

    combined = pd.concat(
        [
            existing,
            pd.DataFrame([result]),
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "model",
            "window_size",
            "feature_set",
            "seed",
        ]
    ).reset_index(drop=True)

    combined.to_csv(
        TEMPORAL_ABLATION_RESULTS_PATH,
        index=False,
    )

    return combined


def update_summary(
    results: pd.DataFrame,
) -> None:
    """Save grouped seed statistics and comparison figure."""

    summary = (
        results
        .groupby(
            [
                "model",
                "window_size",
                "feature_set",
            ],
            as_index=False,
        )
        .agg(
            runs=("run_id", "count"),
            validation_rmse_mean=(
                "validation_rmse",
                "mean",
            ),
            validation_rmse_std=(
                "validation_rmse",
                "std",
            ),
            validation_rmse_min=(
                "validation_rmse",
                "min",
            ),
            validation_mae_mean=(
                "validation_mae",
                "mean",
            ),
            local_test_rmse_mean=(
                "local_test_rmse",
                "mean",
            ),
            local_test_rmse_std=(
                "local_test_rmse",
                "std",
            ),
            training_seconds_mean=(
                "training_seconds",
                "mean",
            ),
        )
        .sort_values(
            "validation_rmse_mean"
        )
    )

    summary.to_csv(
        TEMPORAL_ABLATION_SUMMARY_PATH,
        index=False,
    )

    seed_42_results = results.loc[
        results["seed"] == RANDOM_SEED
    ].copy()

    if seed_42_results.empty:
        return

    seed_42_results["label"] = (
        seed_42_results["model"].str.upper()
        + "\n"
        + seed_42_results[
            "window_size"
        ].astype(str)
        + "h\n"
        + seed_42_results[
            "feature_set"
        ]
    )

    seed_42_results = (
        seed_42_results
        .sort_values("validation_rmse")
    )

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    axis.bar(
        seed_42_results["label"],
        seed_42_results[
            "validation_rmse"
        ],
    )

    axis.axhline(
        15.717592283333612,
        linestyle="--",
        label="Feedforward validation RMSE",
    )

    axis.set_title(
        "Temporal Window and Feature Ablation"
    )

    axis.set_xlabel("Configuration")
    axis.set_ylabel("Validation RMSE")

    axis.tick_params(
        axis="x",
        rotation=25,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        TEMPORAL_ABLATION_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def train_experiment(
    *,
    model_name: str,
    window_size: int,
    feature_set: str,
    seed: int,
    processed: pd.DataFrame,
    base_sequence_index: pd.DataFrame,
    all_feature_columns: list[str],
    arguments: argparse.Namespace,
) -> dict[str, object]:
    """Train and evaluate one controlled experiment."""

    run_id = make_run_id(
        model=model_name,
        window_size=window_size,
        feature_set=feature_set,
        seed=seed,
    )

    print("\n" + "=" * 80)
    print(f"RUN: {run_id}")
    print("=" * 80)

    selected_features = (
        select_temporal_features(
            all_feature_columns,
            feature_set,
        )
    )

    resized_index = resize_sequence_windows(
        base_sequence_index,
        window_size,
    )

    split_indices = {
        split: filter_sequence_split(
            resized_index,
            split,
        )
        for split in [
            "train",
            "validation",
            "local_test",
        ]
    }

    feature_matrix = processed[
        selected_features
    ].to_numpy(dtype=np.float32)

    tf.keras.backend.clear_session()

    set_reproducible_seed(
        seed,
        deterministic=True,
    )

    builder = MODEL_BUILDERS[
        model_name
    ]

    model = builder(
        input_shape=(
            window_size,
            len(selected_features),
        ),
        units=arguments.units,
        dense_units=arguments.dense_units,
        dropout_rate=arguments.dropout_rate,
        learning_rate=arguments.learning_rate,
        seed=seed,
    )

    checkpoint_path = (
        TEMPORAL_ABLATION_MODEL_DIR
        / f"{run_id}.keras"
    )

    train_dataset = create_dataset(
        feature_matrix,
        split_indices["train"],
        window_size=window_size,
        batch_size=arguments.batch_size,
        shuffle=True,
        seed=seed,
        return_targets=True,
    )

    validation_dataset = create_dataset(
        feature_matrix,
        split_indices["validation"],
        window_size=window_size,
        batch_size=arguments.batch_size,
        shuffle=False,
        seed=seed,
        return_targets=True,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
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

    start_time = time.perf_counter()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=arguments.epochs,
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

    history_path = (
        TEMPORAL_ABLATION_HISTORY_DIR
        / f"{run_id}.csv"
    )

    history_frame.to_csv(
        history_path,
        index=False,
    )

    best_epoch_position = int(
        history_frame["val_rmse"].idxmin()
    )

    best_epoch = int(
        history_frame.loc[
            best_epoch_position,
            "epoch",
        ]
    )

    best_model = tf.keras.models.load_model(
        checkpoint_path
    )

    validation_prediction_dataset = (
        create_dataset(
            feature_matrix,
            split_indices["validation"],
            window_size=window_size,
            batch_size=(
                arguments.batch_size * 2
            ),
            shuffle=False,
            seed=seed,
            return_targets=False,
        )
    )

    validation_predictions = (
        best_model.predict(
            validation_prediction_dataset,
            verbose=0,
        )
        .reshape(-1)
    )

    validation_predictions = (
        clip_pm25_predictions(
            validation_predictions
        )
    )

    validation_targets = (
        split_indices[
            "validation"
        ]["target"].to_numpy(
            dtype=np.float32
        )
    )

    validation_metrics = regression_metrics(
        validation_targets,
        validation_predictions,
    )

    local_test_metrics = {
        "rmse": np.nan,
        "mae": np.nan,
        "r2": np.nan,
    }

    if arguments.evaluate_local_test:
        local_prediction_dataset = (
            create_dataset(
                feature_matrix,
                split_indices["local_test"],
                window_size=window_size,
                batch_size=(
                    arguments.batch_size * 2
                ),
                shuffle=False,
                seed=seed,
                return_targets=False,
            )
        )

        local_predictions = (
            best_model.predict(
                local_prediction_dataset,
                verbose=0,
            )
            .reshape(-1)
        )

        local_predictions = (
            clip_pm25_predictions(
                local_predictions
            )
        )

        local_targets = (
            split_indices[
                "local_test"
            ]["target"].to_numpy(
                dtype=np.float32
            )
        )

        local_test_metrics = regression_metrics(
            local_targets,
            local_predictions,
        )

    result = {
        "run_id": run_id,
        "model": model_name,
        "window_size": window_size,
        "feature_set": feature_set,
        "feature_count": len(
            selected_features
        ),
        "seed": seed,
        "units": arguments.units,
        "dense_units": arguments.dense_units,
        "dropout_rate": (
            arguments.dropout_rate
        ),
        "learning_rate": (
            arguments.learning_rate
        ),
        "batch_size": arguments.batch_size,
        "parameters": (
            best_model.count_params()
        ),
        "epochs_trained": len(
            history_frame
        ),
        "best_epoch": best_epoch,
        "training_seconds": (
            training_seconds
        ),
        "train_samples": len(
            split_indices["train"]
        ),
        "validation_samples": len(
            split_indices["validation"]
        ),
        "local_test_samples": len(
            split_indices["local_test"]
        ),
        "validation_rmse": (
            validation_metrics["rmse"]
        ),
        "validation_mae": (
            validation_metrics["mae"]
        ),
        "validation_r2": (
            validation_metrics["r2"]
        ),
        "local_test_rmse": (
            local_test_metrics["rmse"]
        ),
        "local_test_mae": (
            local_test_metrics["mae"]
        ),
        "local_test_r2": (
            local_test_metrics["r2"]
        ),
    }

    print("\nExperiment result")
    print("-" * 80)
    print(f"Run: {run_id}")
    print(
        f"Input shape: "
        f"({window_size}, "
        f"{len(selected_features)})"
    )
    print(
        "Validation RMSE:",
        f"{validation_metrics['rmse']:.4f}",
    )
    print(
        "Validation MAE:",
        f"{validation_metrics['mae']:.4f}",
    )
    print(
        "Best epoch:",
        best_epoch,
    )
    print(
        "Training seconds:",
        f"{training_seconds:.2f}",
    )

    del model
    del best_model
    del train_dataset
    del validation_dataset
    del feature_matrix

    gc.collect()

    return result


def main() -> None:
    """Run the requested ablation configurations."""

    arguments = parse_arguments()

    if arguments.grid:
        window_sizes = [
            6,
            12,
            24,
        ]

        feature_sets = [
            "pollution_only",
            "all_features",
        ]

        seeds = [
            RANDOM_SEED,
        ]

    else:
        window_sizes = arguments.window_sizes
        feature_sets = arguments.feature_sets
        seeds = arguments.seeds

    required_paths = [
        PROCESSED_HOURLY_PATH,
        PREPROCESSOR_PATH,
        SEQUENCE_INDEX_PATH,
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

    TEMPORAL_ABLATION_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEMPORAL_ABLATION_HISTORY_DIR.mkdir(
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

    print("\nLoading data...")

    processed = pd.read_parquet(
        PROCESSED_HOURLY_PATH
    )

    processed = (
        processed
        .sort_values(
            [
                "station",
                "datetime",
            ]
        )
        .reset_index(drop=True)
    )

    base_sequence_index = pd.read_parquet(
        SEQUENCE_INDEX_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    all_feature_columns = (
        preprocessor.get_feature_names_out()
    )

    existing_results = (
        load_existing_results()
    )

    configurations = list(
        product(
            window_sizes,
            feature_sets,
            seeds,
        )
    )

    print(
        f"Requested experiments: "
        f"{len(configurations)}"
    )

    for (
        window_size,
        feature_set,
        seed,
    ) in configurations:
        run_id = make_run_id(
            model=arguments.model,
            window_size=window_size,
            feature_set=feature_set,
            seed=seed,
        )

        if should_skip_run(
            existing_results,
            run_id,
            evaluate_local_test=(
                arguments.evaluate_local_test
            ),
            force=arguments.force,
        ):
            print(
                f"Skipping completed run: "
                f"{run_id}"
            )
            continue

        result = train_experiment(
            model_name=arguments.model,
            window_size=window_size,
            feature_set=feature_set,
            seed=seed,
            processed=processed,
            base_sequence_index=(
                base_sequence_index
            ),
            all_feature_columns=(
                all_feature_columns
            ),
            arguments=arguments,
        )

        existing_results = save_result(
            result
        )

        update_summary(
            existing_results
        )

    final_results = load_existing_results()

    if final_results.empty:
        print(
            "No ablation results were generated."
        )
        return

    ranked = final_results.sort_values(
        "validation_rmse"
    )

    print("\nValidation ranking")
    print("-" * 120)

    columns = [
        "run_id",
        "model",
        "window_size",
        "feature_set",
        "feature_count",
        "seed",
        "best_epoch",
        "training_seconds",
        "validation_rmse",
        "validation_mae",
        "validation_r2",
    ]

    print(
        ranked[columns].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    best = ranked.iloc[0]

    print("\nBest validation configuration")
    print("-" * 80)
    print("Run:", best["run_id"])
    print(
        "Validation RMSE:",
        f"{best['validation_rmse']:.4f}",
    )
    print(
        "Validation MAE:",
        f"{best['validation_mae']:.4f}",
    )

    print("\n" + "=" * 80)
    print("TEMPORAL ABLATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()