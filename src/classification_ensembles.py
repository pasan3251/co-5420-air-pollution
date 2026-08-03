"""Classification ensemble utilities for soft-voting models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def weighted_average_probabilities(
    probs_list: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """Compute the weighted average of predicted probabilities."""
    
    if len(probs_list) != len(weights):
        raise ValueError("Number of probability arrays must match number of weights.")
        
    for w in weights:
        if not 0.0 <= w <= 1.0:
            raise ValueError("All weights must be between 0 and 1.")
            
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Weights must sum to 1.0.")
        
    ensemble_probs = np.zeros_like(probs_list[0], dtype=np.float64)
    
    for probs, weight in zip(probs_list, weights):
        if probs.shape != probs_list[0].shape:
            raise ValueError("All probability arrays must have identical shapes.")
            
        ensemble_probs += np.asarray(probs, dtype=np.float64) * weight
        
    return ensemble_probs.astype(np.float32)


def search_three_model_weights(
    y_true: Sequence[int] | np.ndarray,
    first_probs: np.ndarray,
    second_probs: np.ndarray,
    third_probs: np.ndarray,
    *,
    step: float = 0.05,
    first_name: str = "feedforward",
    second_name: str = "lstm",
    third_name: str = "gru",
) -> pd.DataFrame:
    """Evaluate convex ensemble weights for 3 models using validation accuracy."""

    if not 0.0 < step <= 1.0:
        raise ValueError("step must be greater than zero and at most one.")

    records = []
    
    # 3D simplex grid search
    for w1 in np.arange(0.0, 1.0 + step/2, step):
        for w2 in np.arange(0.0, 1.0 - w1 + step/2, step):
            w3 = 1.0 - w1 - w2
            
            # Account for floating point precision issues
            if w3 < -1e-5:
                continue
            w3 = max(0.0, w3)
            
            # Re-normalize to exactly 1.0
            total = w1 + w2 + w3
            fw1, fw2, fw3 = w1/total, w2/total, w3/total

            ensemble_probs = weighted_average_probabilities(
                [first_probs, second_probs, third_probs],
                [fw1, fw2, fw3],
            )
            
            ensemble_classes = np.argmax(ensemble_probs, axis=1)
            accuracy = accuracy_score(y_true, ensemble_classes)
            
            records.append({
                f"{first_name}_weight": float(fw1),
                f"{second_name}_weight": float(fw2),
                f"{third_name}_weight": float(fw3),
                "accuracy": float(accuracy),
            })

    df = pd.DataFrame(records)
    return df.sort_values("accuracy", ascending=False).reset_index(drop=True)
