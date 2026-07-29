"""Select and evaluate a feedforward-LSTM prediction ensemble."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.baselines import clip_pm25_predictions
from src.config import (
    BASELINE_METRICS_PATH,
    ENSEMBLE_METRICS_PATH,
    ENSEMBLE_PREDICTIONS_DIR,
    ENSEMBLE_RANGE_METRICS_PATH,
    ENSEMBLE_STATION_METRICS_PATH,
    ENSEMBLE_WEIGHT_FIGURE_PATH,
    ENSEMBLE_WEIGHT_SEARCH_PATH,
    FEEDFORWARD_METRICS_PATH,
    FEEDFORWARD_PREDICTIONS_DIR,
    FINAL_MODEL_COMPARISON_FIGURE_PATH,
    RECURRENT_METRICS_PATH,
    RECURRENT_PREDICTIONS_DIR,
)
from src.ensembles import (
    align_prediction_frames,
    search_two_model_weights,
    weighted_average_predictions,
)
from src.evaluation import (
    pollution_range_metrics,
    regression_metrics,
    stationwise_regression_metrics,
)

MODEL_NAME = "feedforward_lstm_ensemble"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Select a feedforward-LSTM ensemble weight "
            "using validation predictions."
        )
    )

    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.01,
    )

    return parser.parse_args()


def load_aligned_predictions(
    split: str,
) -> pd.DataFrame:
    """Load and align feedforward and LSTM outputs."""

    feedforward_path = (
        FEEDFORWARD_PREDICTIONS_DIR
        / f"{split}_predictions.csv"
    )

    lstm_path = (
        RECURRENT_PREDICTIONS_DIR
        / "lstm"
        / f"{split}_predictions.csv"
    )

    if not feedforward_path.exists():
        raise FileNotFoundError(
            f"Missing feedforward predictions: "
            f"{feedforward_path}"
        )

    if not lstm_path.exists():
        raise FileNotFoundError(
            f"Missing LSTM predictions: {lstm_path}"
        )

    feedforward = pd.read_csv(
        feedforward_path
    )

    lstm = pd.read_csv(
        lstm_path
    )

    return align_prediction_frames(
        feedforward,
        lstm,
        first_name="feedforward",
        second_name="lstm",
    )


def evaluate_split(
    aligned: pd.DataFrame,
    *,
    split: str,
    feedforward_weight: float,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Apply the frozen ensemble weight to one split."""

    predictions = weighted_average_predictions(
        aligned["prediction_feedforward"],
        aligned["prediction_lstm"],
        first_weight=feedforward_weight,
    )

    predictions = clip_pm25_predictions(
        predictions
    )

    y_true = aligned[
        "y_true"
    ].to_numpy(dtype=np.float32)

    overall = {
        "model": MODEL_NAME,
        "split": split,
        "samples": len(aligned),
        "feedforward_weight": (
            feedforward_weight
        ),
        "lstm_weight": (
            1.0 - feedforward_weight
        ),
        **regression_metrics(
            y_true,
            predictions,
        ),
    }

    station_metrics = (
        stationwise_regression_metrics(
            aligned["station"],
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

    prediction_frame = aligned[
        [
            "sequence_id",
            "station",
            "target_datetime",
            "y_true",
            "prediction_feedforward",
            "prediction_lstm",
        ]
    ].copy()

    prediction_frame["y_pred"] = predictions

    prediction_frame["residual"] = (
        prediction_frame["y_true"]
        - prediction_frame["y_pred"]
    )

    return (
        overall,
        station_metrics,
        range_metrics,
        prediction_frame,
    )


def save_weight_figure(
    search_results: pd.DataFrame,
    best_weight: float,
) -> None:
    """Plot validation RMSE against feedforward weight."""

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        search_results["feedforward_weight"],
        search_results["rmse"],
    )

    axis.axvline(
        best_weight,
        linestyle="--",
        label=(
            "Selected feedforward weight: "
            f"{best_weight:.2f}"
        ),
    )

    axis.set_title(
        "Feedforward-LSTM Ensemble Weight Search"
    )

    axis.set_xlabel("Feedforward weight")
    axis.set_ylabel("Validation RMSE")
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        ENSEMBLE_WEIGHT_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_model_comparison_figure(
    ensemble_metrics: pd.DataFrame,
) -> None:
    """Compare all main model families."""

    baseline = pd.read_csv(
        BASELINE_METRICS_PATH
    )[
        ["model", "split", "rmse"]
    ]

    feedforward = pd.read_csv(
        FEEDFORWARD_METRICS_PATH
    )[
        ["model", "split", "rmse"]
    ]

    recurrent = pd.read_csv(
        RECURRENT_METRICS_PATH
    )[
        ["model", "split", "rmse"]
    ]

    comparison = pd.concat(
        [
            baseline,
            feedforward,
            recurrent,
            ensemble_metrics[
                ["model", "split", "rmse"]
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
        figsize=(13, 7)
    )

    pivot.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(
        "Final Model-Family Comparison"
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
        FINAL_MODEL_COMPARISON_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    """Select the validation weight and evaluate the ensemble."""

    arguments = parse_arguments()

    ENSEMBLE_PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation = load_aligned_predictions(
        "validation"
    )

    local_test = load_aligned_predictions(
        "local_test"
    )

    search_results = search_two_model_weights(
        validation["y_true"],
        validation["prediction_feedforward"],
        validation["prediction_lstm"],
        step=arguments.weight_step,
    )

    search_results.to_csv(
        ENSEMBLE_WEIGHT_SEARCH_PATH,
        index=False,
    )

    best_row = (
        search_results
        .sort_values(
            [
                "rmse",
                "mae",
                "feedforward_weight",
            ]
        )
        .iloc[0]
    )

    selected_weight = float(
        best_row["feedforward_weight"]
    )

    print("Selected using validation only")
    print("-" * 60)
    print(
        "Feedforward weight:",
        f"{selected_weight:.2f}",
    )
    print(
        "LSTM weight:",
        f"{1.0 - selected_weight:.2f}",
    )
    print(
        "Validation search RMSE:",
        f"{best_row['rmse']:.4f}",
    )

    overall_records = []
    station_frames = []
    range_frames = []

    for split, aligned in [
        ("validation", validation),
        ("local_test", local_test),
    ]:
        (
            overall,
            station_metrics,
            range_metrics,
            prediction_frame,
        ) = evaluate_split(
            aligned,
            split=split,
            feedforward_weight=(
                selected_weight
            ),
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

        prediction_frame.to_csv(
            ENSEMBLE_PREDICTIONS_DIR
            / f"{split}_predictions.csv",
            index=False,
        )

    metrics = pd.DataFrame(
        overall_records
    )

    station_metrics = pd.concat(
        station_frames,
        ignore_index=True,
    )

    range_metrics = pd.concat(
        range_frames,
        ignore_index=True,
    )

    metrics.to_csv(
        ENSEMBLE_METRICS_PATH,
        index=False,
    )

    station_metrics.to_csv(
        ENSEMBLE_STATION_METRICS_PATH,
        index=False,
    )

    range_metrics.to_csv(
        ENSEMBLE_RANGE_METRICS_PATH,
        index=False,
    )

    save_weight_figure(
        search_results,
        selected_weight,
    )

    save_model_comparison_figure(
        metrics
    )

    print("\nEnsemble metrics")
    print("-" * 100)

    print(
        metrics.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nReference results")
    print("-" * 60)
    print("LSTM validation RMSE:       15.6505")
    print("Feedforward validation RMSE: 15.6733")
    print("Feedforward local RMSE:      25.1482")
    print("LSTM local RMSE:             25.9367")

    print("\n" + "=" * 80)
    print("ENSEMBLE EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()