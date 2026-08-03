"""Train AQI classification models."""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    PROCESSED_HOURLY_PATH,
    SEQUENCE_INDEX_PATH,
    PREPROCESSOR_PATH,
    MODELS_DIR,
    WINDOW_SIZE,
    RANDOM_SEED,
)
from src.classification_models import (
    build_lstm_model,
    build_gru_model,
    build_feedforward_model,
    set_reproducible_seed,
)
from src.sequence_builder import filter_sequence_split
from src.temporal_dataset import TemporalWindowDataset
import joblib

MODEL_BUILDERS = {
    "feedforward": build_feedforward_model,
    "lstm": build_lstm_model,
    "gru": build_gru_model,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AQI classification models.")
    parser.add_argument("--models", nargs="+", choices=["feedforward", "lstm", "gru"], default=["lstm"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def categorize_aqi(pm25_values):
    """Converts continuous PM2.5 values into 4 discrete AQI classes (0, 1, 2, 3)."""
    classes = np.zeros_like(pm25_values, dtype=np.int32)
    classes = np.where((pm25_values > 12.0) & (pm25_values <= 35.4), 1, classes)
    classes = np.where((pm25_values > 35.4) & (pm25_values <= 150.4), 2, classes)
    classes = np.where(pm25_values > 150.4, 3, classes)
    return classes


def build_dataset(
    feature_matrix: np.ndarray,
    sequence_index: pd.DataFrame,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> TemporalWindowDataset:
    return TemporalWindowDataset(
        feature_matrix,
        sequence_index,
        window_size=WINDOW_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        return_targets=True,
    )


def train_one_model(model_name: str, feature_matrix: np.ndarray, split_indices: dict, arguments: argparse.Namespace):
    tf.keras.backend.clear_session()
    set_reproducible_seed(arguments.seed, deterministic=True)
    
    model_path = MODELS_DIR / f"classification_{model_name}.keras"
    
    model = MODEL_BUILDERS[model_name](
        input_shape=(WINDOW_SIZE, feature_matrix.shape[1]),
        num_classes=4,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
    )

    print(f"\n{model_name.upper()} architecture")
    model.summary()

    train_dataset = build_dataset(feature_matrix, split_indices["train"], batch_size=arguments.batch_size, shuffle=True, seed=arguments.seed)
    val_dataset = build_dataset(feature_matrix, split_indices["validation"], batch_size=arguments.batch_size, shuffle=False, seed=arguments.seed)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=str(model_path), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=arguments.patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    print(f"\nTraining {model_name.upper()}...")
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=arguments.epochs, callbacks=callbacks, verbose=2)
    return model


def main() -> None:
    arguments = parse_arguments()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading temporal data...")
    processed = pd.read_parquet(PROCESSED_HOURLY_PATH).sort_values(["station", "datetime"]).reset_index(drop=True)
    sequence_index = pd.read_parquet(SEQUENCE_INDEX_PATH)
    
    # CRITICAL: Map continuous targets to integer classes
    sequence_index["target"] = categorize_aqi(sequence_index["target"].values)
    
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_columns = preprocessor.get_feature_names_out()
    feature_matrix = processed[feature_columns].to_numpy(dtype=np.float32)

    split_indices = {
        split: filter_sequence_split(sequence_index, split)
        for split in ["train", "validation"]
    }

    for model_name in arguments.models:
        train_one_model(model_name, feature_matrix, split_indices, arguments)

if __name__ == "__main__":
    main()
