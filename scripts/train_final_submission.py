"""Refit final models and generate the Kaggle submission."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
    FINAL_FEEDFORWARD_HISTORY_PATH,
    FINAL_FEEDFORWARD_MODEL_PATH,
    FINAL_LSTM_HISTORY_PATH,
    FINAL_LSTM_MODEL_PATH,
    FINAL_MODEL_DIR,
    FINAL_PREPROCESSOR_PATH,
    FINAL_SUBMISSION_METADATA_PATH,
    FINAL_SUBMISSION_PATH,
    FINAL_SUBMISSION_VALIDATION_PATH,
    RANDOM_SEED,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
    TEST_PATH,
    TRAIN_RAW_PATH,
    WINDOW_SIZE,
)
from src.ensembles import (
    weighted_average_predictions,
)
from src.kaggle import (
    build_kaggle_submission,
    transform_kaggle_test_frame,
    validate_submission_frame,
)
from src.models import (
    build_feedforward_model,
    build_lstm_model,
    set_reproducible_seed,
)
from src.preprocessing import (
    AirPollutionPreprocessor,
    add_datetime_column,
)
from src.sequence_builder import (
    build_sequence_index,
)
from src.temporal_dataset import (
    TemporalWindowDataset,
)


def parse_arguments() -> argparse.Namespace:
    """Parse final-training arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Refit the final feedforward-LSTM ensemble "
            "and generate a Kaggle submission."
        )
    )

    parser.add_argument(
        "--feedforward-epochs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lstm-epochs",
        type=int,
        default=19,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--feedforward-weight",
        type=float,
        default=0.49,
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
            "Train one epoch on a reduced sample and "
            "write smoke_submission.csv."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    """Validate command-line values."""

    if arguments.feedforward_epochs <= 0:
        raise ValueError(
            "feedforward-epochs must be positive."
        )

    if arguments.lstm_epochs <= 0:
        raise ValueError(
            "lstm-epochs must be positive."
        )

    if arguments.batch_size <= 0:
        raise ValueError(
            "batch-size must be positive."
        )

    if not (
        0.0
        <= arguments.feedforward_weight
        <= 1.0
    ):
        raise ValueError(
            "feedforward-weight must be between 0 and 1."
        )


def history_to_frame(
    history: tf.keras.callbacks.History,
) -> pd.DataFrame:
    """Convert a Keras history object into a dataframe."""

    frame = pd.DataFrame(
        history.history
    )

    frame.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(frame) + 1,
        ),
    )

    return frame


def file_sha256(
    path: Path,
) -> str:
    """Calculate a file SHA-256 digest."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    """Execute final refitting and submission creation."""

    arguments = parse_arguments()
    validate_arguments(arguments)

    required_paths = [
        TRAIN_RAW_PATH,
        TEST_PATH,
        SAMPLE_SUBMISSION_PATH,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Required competition files are missing:\n"
            + "\n".join(missing_paths)
        )

    FINAL_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUBMISSIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "TensorFlow version:",
        tf.__version__,
    )

    gpu_devices = tf.config.list_physical_devices(
        "GPU"
    )

    print(
        "GPU devices:",
        gpu_devices,
    )

    print("\nLoading official competition files...")

    raw_train = pd.read_csv(
        TRAIN_RAW_PATH,
        low_memory=False,
    )

    raw_test = pd.read_csv(
        TEST_PATH,
        low_memory=False,
    )

    sample_submission = pd.read_csv(
        SAMPLE_SUBMISSION_PATH
    )

    print(
        "Training rows:",
        f"{len(raw_train):,}",
    )

    print(
        "Kaggle test rows:",
        f"{len(raw_test):,}",
    )

    print(
        "\nFitting final preprocessor "
        "using all training rows..."
    )

    raw_train = add_datetime_column(
        raw_train
    )

    raw_train = (
        raw_train
        .sort_values(
            [
                "station",
                "datetime",
            ]
        )
        .reset_index(drop=True)
    )

    final_preprocessor = (
        AirPollutionPreprocessor()
    )

    final_preprocessor.fit(
        raw_train
    )

    processed_train = (
        final_preprocessor.transform(
            raw_train
        )
    )

    processed_train["split"] = "train"

    feature_columns = (
        final_preprocessor
        .get_feature_names_out()
    )

    final_sequence_index = (
        build_sequence_index(
            processed_train,
            feature_columns,
            window_size=WINDOW_SIZE,
        )
    )

    print(
        "Final training sequences:",
        f"{len(final_sequence_index):,}",
    )

    print(
        "Sequence feature shape:",
        (
            WINDOW_SIZE,
            len(feature_columns),
        ),
    )

    if len(final_sequence_index) != 308_988:
        raise ValueError(
            "Unexpected final training-sequence count: "
            f"{len(final_sequence_index):,}"
        )

    print(
        "\nTransforming official Kaggle test windows..."
    )

    kaggle_test = transform_kaggle_test_frame(
        raw_test,
        final_preprocessor,
        window_size=WINDOW_SIZE,
    )

    print(
        "Kaggle tensor shape:",
        kaggle_test.sequence_tensor.shape,
    )

    if kaggle_test.sequence_tensor.shape != (
        len(raw_test),
        WINDOW_SIZE,
        len(feature_columns),
    ):
        raise ValueError(
            "Unexpected Kaggle sequence-tensor shape."
        )

    joblib.dump(
        final_preprocessor,
        FINAL_PREPROCESSOR_PATH,
    )

    training_index = final_sequence_index

    feedforward_epochs = (
        arguments.feedforward_epochs
    )

    lstm_epochs = arguments.lstm_epochs

    if arguments.smoke_test:
        print(
            "\nSMOKE TEST MODE: using 8,192 "
            "training sequences and one epoch."
        )

        training_index = (
            final_sequence_index
            .sample(
                n=min(
                    8_192,
                    len(final_sequence_index),
                ),
                random_state=arguments.seed,
            )
            .sort_values("sequence_id")
            .reset_index(drop=True)
        )

        feedforward_epochs = 1
        lstm_epochs = 1

    print(
        "\nBuilding feedforward tabular features..."
    )

    (
        train_tabular_features,
        train_tabular_names,
    ) = build_tabular_baseline_features(
        processed_train,
        training_index,
        feature_columns,
    )

    (
        test_tabular_features,
        test_tabular_names,
    ) = build_tabular_baseline_features(
        kaggle_test.flat_features,
        kaggle_test.sequence_index,
        feature_columns,
    )

    if (
        train_tabular_names
        != test_tabular_names
    ):
        raise ValueError(
            "Training and Kaggle tabular "
            "feature order differs."
        )

    if BASELINE_FEATURE_NAMES_PATH.exists():
        with BASELINE_FEATURE_NAMES_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            expected_metadata = json.load(file)

        if (
            train_tabular_names
            != expected_metadata[
                "feature_names"
            ]
        ):
            raise ValueError(
                "Final tabular feature order differs "
                "from the validated baseline order."
            )

    targets = training_index[
        "target"
    ].to_numpy(dtype=np.float32)

    print(
        "Feedforward training shape:",
        train_tabular_features.shape,
    )

    print(
        "Feedforward test shape:",
        test_tabular_features.shape,
    )

    tf.keras.backend.clear_session()

    set_reproducible_seed(
        arguments.seed,
        deterministic=True,
    )

    feedforward_model = (
        build_feedforward_model(
            input_dimension=(
                train_tabular_features.shape[1]
            ),
            seed=arguments.seed,
        )
    )

    print(
        "\nTraining final feedforward model..."
    )

    feedforward_start = (
        time.perf_counter()
    )

    feedforward_history = (
        feedforward_model.fit(
            train_tabular_features,
            targets,
            epochs=feedforward_epochs,
            batch_size=arguments.batch_size,
            shuffle=True,
            verbose=2,
        )
    )

    feedforward_seconds = (
        time.perf_counter()
        - feedforward_start
    )

    feedforward_predictions = (
        feedforward_model.predict(
            test_tabular_features,
            batch_size=1_024,
            verbose=0,
        )
        .reshape(-1)
    )

    feedforward_predictions = (
        clip_pm25_predictions(
            feedforward_predictions
        )
    )

    if arguments.smoke_test:
        feedforward_model_path = (
            FINAL_MODEL_DIR
            / "_smoke_feedforward.keras"
        )
    else:
        feedforward_model_path = (
            FINAL_FEEDFORWARD_MODEL_PATH
        )

    feedforward_model.save(
        feedforward_model_path
    )

    feedforward_history_frame = (
        history_to_frame(
            feedforward_history
        )
    )

    del feedforward_model
    del train_tabular_features
    del test_tabular_features

    gc.collect()

    tf.keras.backend.clear_session()

    print(
        "\nPreparing final LSTM dataset..."
    )

    feature_matrix = processed_train[
        feature_columns
    ].to_numpy(dtype=np.float32)

    lstm_dataset = TemporalWindowDataset(
        feature_matrix,
        training_index,
        window_size=WINDOW_SIZE,
        batch_size=arguments.batch_size,
        shuffle=True,
        seed=arguments.seed,
        return_targets=True,
    )

    set_reproducible_seed(
        arguments.seed,
        deterministic=True,
    )

    lstm_model = build_lstm_model(
        input_shape=(
            WINDOW_SIZE,
            len(feature_columns),
        ),
        units=64,
        dense_units=32,
        dropout_rate=0.20,
        learning_rate=1e-3,
        seed=arguments.seed,
    )

    print(
        "\nTraining final LSTM model..."
    )

    lstm_start = time.perf_counter()

    lstm_history = lstm_model.fit(
        lstm_dataset,
        epochs=lstm_epochs,
        verbose=2,
    )

    lstm_seconds = (
        time.perf_counter()
        - lstm_start
    )

    lstm_predictions = (
        lstm_model.predict(
            kaggle_test.sequence_tensor,
            batch_size=1_024,
            verbose=0,
        )
        .reshape(-1)
    )

    lstm_predictions = (
        clip_pm25_predictions(
            lstm_predictions
        )
    )

    if arguments.smoke_test:
        lstm_model_path = (
            FINAL_MODEL_DIR
            / "_smoke_lstm.keras"
        )
    else:
        lstm_model_path = (
            FINAL_LSTM_MODEL_PATH
        )

    lstm_model.save(
        lstm_model_path
    )

    lstm_history_frame = history_to_frame(
        lstm_history
    )

    ensemble_predictions = (
        weighted_average_predictions(
            feedforward_predictions,
            lstm_predictions,
            first_weight=(
                arguments.feedforward_weight
            ),
        )
    )

    ensemble_predictions = (
        clip_pm25_predictions(
            ensemble_predictions
        )
    )

    submission = build_kaggle_submission(
        sample_submission,
        kaggle_test.metadata,
        ensemble_predictions,
    )

    if arguments.smoke_test:
        submission_path = (
            SUBMISSIONS_DIR
            / "smoke_submission.csv"
        )
    else:
        submission_path = (
            FINAL_SUBMISSION_PATH
        )

    submission.to_csv(
        submission_path,
        index=False,
    )

    validation_summary = (
        validate_submission_frame(
            submission,
            sample_submission,
        )
    )

    print("\nSubmission validation")
    print("-" * 70)

    for key, value in (
        validation_summary.items()
    ):
        print(
            f"{key}: {value}"
        )

    print(
        "Submission path:",
        submission_path,
    )

    print(
        "SHA-256:",
        file_sha256(submission_path),
    )

    print("\nSubmission preview")
    print("-" * 70)

    print(
        submission.head().to_string(
            index=False
        )
    )

    if arguments.smoke_test:
        print("\n" + "=" * 80)
        print("FINAL PIPELINE SMOKE TEST COMPLETED")
        print("=" * 80)
        return

    feedforward_history_frame.to_csv(
        FINAL_FEEDFORWARD_HISTORY_PATH,
        index=False,
    )

    lstm_history_frame.to_csv(
        FINAL_LSTM_HISTORY_PATH,
        index=False,
    )

    with FINAL_SUBMISSION_VALIDATION_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            validation_summary,
            file,
            indent=2,
        )

    metadata = {
        "created_at_utc": (
            pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
        ),
        "tensorflow_version": (
            tf.__version__
        ),
        "gpu_devices": [
            device.name
            for device in gpu_devices
        ],
        "training_raw_rows": len(raw_train),
        "training_sequences": len(final_sequence_index),
        "test_samples": len(raw_test),
        "sequence_shape": [
            WINDOW_SIZE,
            len(feature_columns),
        ],
        "tabular_feature_count": len(train_tabular_names),
        "feedforward_epochs": int(
            feedforward_epochs
        ),
        "lstm_epochs": int(
            lstm_epochs
        ),
        "feedforward_training_seconds": float(
            feedforward_seconds
        ),
        "lstm_training_seconds": float(
            lstm_seconds
        ),
        "feedforward_weight": float(
            arguments.feedforward_weight
        ),
        "lstm_weight": float(
            1.0
            - arguments.feedforward_weight
        ),
        "random_seed": int(
            arguments.seed
        ),
        "test_target_start": (
            kaggle_test.metadata[
                "target_datetime"
            ]
            .min()
            .isoformat()
        ),
        "test_target_end": (
            kaggle_test.metadata[
                "target_datetime"
            ]
            .max()
            .isoformat()
        ),
        "submission_filename": (
            submission_path.name
        ),
        "submission_sha256": (
            file_sha256(
                submission_path
            )
        ),
        "submission_validation": (
            validation_summary
        ),
        "data_policy": (
            "Only train_raw.csv, test.csv and "
            "sample_submission.csv were accessed. "
            "test_raw.csv was not used."
        ),
    }

    with FINAL_SUBMISSION_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\nTraining summary")
    print("-" * 70)

    print(
        "Feedforward seconds:",
        f"{feedforward_seconds:.2f}",
    )

    print(
        "LSTM seconds:",
        f"{lstm_seconds:.2f}",
    )

    print(
        "Ensemble weights:",
        (
            arguments.feedforward_weight,
            1.0
            - arguments.feedforward_weight,
        ),
    )

    print("\n" + "=" * 80)
    print("FINAL KAGGLE SUBMISSION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()