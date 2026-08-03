"""Evaluate AQI classification models."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, confusion_matrix

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
from src.sequence_builder import filter_sequence_split
from src.temporal_dataset import TemporalWindowDataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AQI classification models.")
    parser.add_argument("--models", nargs="+", choices=["feedforward", "lstm", "gru"], default=["lstm"])
    parser.add_argument("--split", choices=["validation", "local_test", "train"], default="local_test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--ensemble", action="store_true", help="Evaluate soft-voting ensemble of all chosen models.")
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


def evaluate_one_model(model_name: str, test_dataset: TemporalWindowDataset):
    model_path = MODELS_DIR / f"classification_{model_name}.keras"
    if not model_path.exists():
        print(f"Error: Could not find trained model at {model_path}")
        return

    print(f"\nEvaluating {model_name.upper()}...")
    model = tf.keras.models.load_model(model_path)
    
    # Evaluate basic metrics
    loss, accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

    # Generate full predictions for Confusion Matrix
    print("Generating predictions...")
    y_pred_probs = model.predict(test_dataset, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # Extract true labels directly from the dataset targets
    y_true_classes = test_dataset.targets
    
    target_names = ["Good (0)", "Moderate (1)", "Unhealthy (2)", "Hazardous (3)"]
    
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(y_true_classes, y_pred_classes, target_names=target_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Pretty print confusion matrix
    print(f"{'':<15} | {'Pred: Good':<12} {'Pred: Mod':<12} {'Pred: UnH':<12} {'Pred: Haz':<12}")
    print("-" * 70)
    for i, true_name in enumerate(["True: Good", "True: Mod", "True: UnH", "True: Haz"]):
        row_str = " ".join([f"{val:<12}" for val in cm[i]])
        print(f"{true_name:<15} | {row_str}")


def main() -> None:
    arguments = parse_arguments()

    print(f"\nLoading temporal data for split: {arguments.split}...")
    processed = pd.read_parquet(PROCESSED_HOURLY_PATH).sort_values(["station", "datetime"]).reset_index(drop=True)
    sequence_index = pd.read_parquet(SEQUENCE_INDEX_PATH)
    
    # Map continuous targets to integer classes
    sequence_index["target"] = categorize_aqi(sequence_index["target"].values)
    
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_columns = preprocessor.get_feature_names_out()
    feature_matrix = processed[feature_columns].to_numpy(dtype=np.float32)

    split_index = filter_sequence_split(sequence_index, arguments.split)
    
    test_dataset = build_dataset(
        feature_matrix, 
        split_index, 
        batch_size=arguments.batch_size, 
        shuffle=False, 
        seed=arguments.seed
    )

    if arguments.ensemble:
        if len(arguments.models) < 2:
            print("Error: --ensemble requires at least 2 models.")
            return

        print(f"\nEvaluating ENSEMBLE of {arguments.models}...")
        all_probs = []
        for model_name in arguments.models:
            model_path = MODELS_DIR / f"classification_{model_name}.keras"
            if not model_path.exists():
                print(f"Error: Could not find trained model at {model_path}. Cannot ensemble.")
                return
            model = tf.keras.models.load_model(model_path)
            probs = model.predict(test_dataset, verbose=0)
            all_probs.append(probs)
            
        # Soft voting (average probabilities)
        ensemble_probs = np.mean(all_probs, axis=0)
        y_pred_classes = np.argmax(ensemble_probs, axis=1)
        y_true_classes = test_dataset.targets
        
        target_names = ["Good (0)", "Moderate (1)", "Unhealthy (2)", "Hazardous (3)"]
        
        print("\nEnsemble Classification Report:")
        print("-" * 60)
        print(classification_report(y_true_classes, y_pred_classes, target_names=target_names, zero_division=0))
        
        print("\nEnsemble Confusion Matrix:")
        print("-" * 60)
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        print(f"{'':<15} | {'Pred: Good':<12} {'Pred: Mod':<12} {'Pred: UnH':<12} {'Pred: Haz':<12}")
        print("-" * 70)
        for i, true_name in enumerate(["True: Good", "True: Mod", "True: UnH", "True: Haz"]):
            row_str = " ".join([f"{val:<12}" for val in cm[i]])
            print(f"{true_name:<15} | {row_str}")

    else:
        for model_name in arguments.models:
            evaluate_one_model(model_name, test_dataset)

if __name__ == "__main__":
    main()
