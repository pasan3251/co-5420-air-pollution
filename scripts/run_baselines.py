"""Train and evaluate forecasting baseline models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.baselines import (
    build_tabular_baseline_features,
    clip_pm25_predictions,
    extract_pm25_history,
    historical_mean_predictions,
    persistence_predictions,
)
from src.config import (
    BASELINE_FEATURE_NAMES_PATH,
    BASELINE_METRICS_PATH,
    BASELINE_PREDICTIONS_DIR,
    BASELINE_STATION_METRICS_PATH,
    FIGURES_DIR,
    GRADIENT_BOOSTING_BASELINE_PATH,
    PREPROCESSOR_PATH,
    PROCESSED_HOURLY_PATH,
    RANDOM_SEED,
    RIDGE_BASELINE_PATH,
    SEQUENCE_INDEX_PATH,
)
from src.evaluation import (
    pollution_range_metrics,
    regression_metrics,
    stationwise_regression_metrics,
)
from src.sequence_builder import (
    filter_sequence_split,
)

AVAILABLE_MODELS = [
    "persistence",
    "historical_mean",
    "ridge",
    "gradient_boosting",
]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=("Run air-pollution forecasting baselines.")
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=AVAILABLE_MODELS,
        default=AVAILABLE_MODELS,
        help="Models to train and evaluate.",
    )

    return parser.parse_args()


def save_prediction_frame(
    sequence_index: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    split: str,
) -> None:
    """Save model predictions for one split."""

    output = sequence_index[
        [
            "sequence_id",
            "station",
            "target_datetime",
            "target",
        ]
    ].copy()

    output = output.rename(columns={"target": "y_true"})

    for model_name, model_predictions in predictions.items():
        output[model_name] = model_predictions

    output_path = BASELINE_PREDICTIONS_DIR / f"{split}_predictions.csv"

    output.to_csv(
        output_path,
        index=False,
    )


def evaluate_predictions(
    *,
    model_name: str,
    split: str,
    sequence_index: pd.DataFrame,
    predictions: np.ndarray,
    training_seconds: float,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evaluate one model on one split."""

    y_true = sequence_index["target"].to_numpy(dtype=np.float32)

    predictions = clip_pm25_predictions(predictions)

    metrics = regression_metrics(
        y_true,
        predictions,
    )

    overall_record = {
        "model": model_name,
        "split": split,
        "samples": len(y_true),
        "training_seconds": training_seconds,
        **metrics,
    }

    station_metrics = stationwise_regression_metrics(
        sequence_index["station"],
        y_true,
        predictions,
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
        overall_record,
        station_metrics,
        range_metrics,
    )


def create_rmse_figure(
    metrics_frame: pd.DataFrame,
) -> None:
    """Create a validation/local-test RMSE comparison."""

    pivot = metrics_frame.pivot(
        index="model",
        columns="split",
        values="rmse",
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    pivot.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title("Baseline PM2.5 Forecasting Performance")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.tick_params(
        axis="x",
        rotation=25,
    )
    ax.legend(title="Evaluation split")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "12_baseline_rmse_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    """Run selected forecasting baselines."""

    arguments = parse_arguments()

    required_paths = [
        PROCESSED_HOURLY_PATH,
        PREPROCESSOR_PATH,
        SEQUENCE_INDEX_PATH,
    ]

    missing_paths = [str(path) for path in required_paths if not path.exists()]

    if missing_paths:
        raise FileNotFoundError(
            "Required generated files are missing:\n" + "\n".join(missing_paths)
        )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BASELINE_PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading processed data...")

    frame = pd.read_parquet(PROCESSED_HOURLY_PATH)

    frame = frame.sort_values(["station", "datetime"]).reset_index(drop=True)

    sequence_index = pd.read_parquet(SEQUENCE_INDEX_PATH)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    feature_columns = preprocessor.get_feature_names_out()

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

    print("\nSequence counts")
    print("-" * 60)

    for split, split_index in split_indices.items():
        print(f"{split:<12}: {len(split_index):>8,}")

    histories = {}

    for split in [
        "validation",
        "local_test",
    ]:
        print(f"Extracting PM2.5 history for {split}...")

        histories[split] = extract_pm25_history(
            frame,
            split_indices[split],
            preprocessor,
        )

    predictions_by_split: dict[
        str,
        dict[str, np.ndarray],
    ] = {
        "validation": {},
        "local_test": {},
    }

    training_times: dict[str, float] = {}

    if "persistence" in arguments.models:
        training_times["persistence"] = 0.0

        for split, split_predictions in predictions_by_split.items():
            split_predictions["persistence"] = persistence_predictions(
                histories[split]
            )

    if "historical_mean" in arguments.models:
        training_times["historical_mean"] = 0.0

        for split, split_predictions in predictions_by_split.items():
            split_predictions["historical_mean"] = (
                historical_mean_predictions(histories[split])
            )

    learned_models_requested = any(
        model_name in arguments.models
        for model_name in [
            "ridge",
            "gradient_boosting",
        ]
    )

    feature_names: list[str] = []

    if learned_models_requested:
        tabular_features = {}

        for split in [
            "train",
            "validation",
            "local_test",
        ]:
            print(f"Building tabular features for {split}...")

            (
                tabular_features[split],
                current_feature_names,
            ) = build_tabular_baseline_features(
                frame,
                split_indices[split],
                feature_columns,
            )

            if not feature_names:
                feature_names = current_feature_names
            elif feature_names != (current_feature_names):
                raise RuntimeError("Feature names changed between splits.")

            print(f"{split} feature shape: {tabular_features[split].shape}")

        with BASELINE_FEATURE_NAMES_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "feature_count": len(feature_names),
                    "feature_names": (feature_names),
                },
                file,
                indent=2,
            )

        y_train = split_indices["train"]["target"].to_numpy(dtype=np.float32)

        if "ridge" in arguments.models:
            print("\nTraining Ridge regression...")

            ridge_model = Ridge(
                alpha=10.0,
                solver="lsqr",
            )

            start_time = time.perf_counter()

            ridge_model.fit(
                tabular_features["train"],
                y_train,
            )

            training_times["ridge"] = time.perf_counter() - start_time

            joblib.dump(
                ridge_model,
                RIDGE_BASELINE_PATH,
            )

            for split, split_predictions in predictions_by_split.items():
                split_predictions["ridge"] = ridge_model.predict(
                    tabular_features[split]
                )

        if "gradient_boosting" in arguments.models:
            print("\nTraining histogram gradient boosting...")

            gradient_boosting_model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=RANDOM_SEED,
            )

            start_time = time.perf_counter()

            gradient_boosting_model.fit(
                tabular_features["train"],
                y_train,
            )

            training_times["gradient_boosting"] = time.perf_counter() - start_time

            joblib.dump(
                gradient_boosting_model,
                GRADIENT_BOOSTING_BASELINE_PATH,
            )

            for split, split_predictions in predictions_by_split.items():
                split_predictions["gradient_boosting"] = (
                    gradient_boosting_model.predict(
                        tabular_features[split]
                    )
                )

    metric_records = []
    station_metric_frames = []
    range_metric_frames = []

    for split, model_predictions in predictions_by_split.items():
        for model_name, predictions in model_predictions.items():
            (
                overall_record,
                station_metrics,
                range_metrics,
            ) = evaluate_predictions(
                model_name=model_name,
                split=split,
                sequence_index=split_indices[split],
                predictions=predictions,
                training_seconds=(training_times[model_name]),
            )

            metric_records.append(overall_record)

            station_metric_frames.append(station_metrics)

            range_metric_frames.append(range_metrics)

        save_prediction_frame(
            split_indices[split],
            model_predictions,
            split=split,
        )

    metrics_frame = pd.DataFrame(metric_records).sort_values(["split", "rmse"])

    station_metrics_frame = pd.concat(
        station_metric_frames,
        ignore_index=True,
    )

    range_metrics_frame = pd.concat(
        range_metric_frames,
        ignore_index=True,
    )

    metrics_frame.to_csv(
        BASELINE_METRICS_PATH,
        index=False,
    )

    station_metrics_frame.to_csv(
        BASELINE_STATION_METRICS_PATH,
        index=False,
    )

    range_metrics_frame.to_csv(
        BASELINE_METRICS_PATH.with_name("baseline_pollution_range_metrics.csv"),
        index=False,
    )

    create_rmse_figure(metrics_frame)

    print("\nOverall baseline metrics")
    print("-" * 100)

    print(
        metrics_frame.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    best_validation_row = (
        metrics_frame.loc[metrics_frame["split"] == "validation"]
        .sort_values("rmse")
        .iloc[0]
    )

    print("\nBest validation baseline")
    print("-" * 60)
    print(f"Model: {best_validation_row['model']}")
    print(
        "RMSE:",
        f"{best_validation_row['rmse']:.4f}",
    )
    print(
        "MAE:",
        f"{best_validation_row['mae']:.4f}",
    )
    print(
        "R²:",
        f"{best_validation_row['r2']:.4f}",
    )

    print("\n" + "=" * 80)
    print("BASELINE EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
