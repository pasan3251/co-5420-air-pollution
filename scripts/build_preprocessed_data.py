"""Build the leakage-safe hourly feature dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    METRICS_DIR,
    PREPROCESSOR_PATH,
    PROCESSED_DATA_DIR,
    PROCESSED_HOURLY_PATH,
    TARGET_OBSERVED_COLUMN,
    TRAIN_RAW_PATH,
)
from src.preprocessing import (
    AirPollutionPreprocessor,
    add_datetime_column,
    assign_time_split,
)


def main() -> None:
    """Fit the preprocessor and generate processed data."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading raw training data...")

    raw = pd.read_csv(
        TRAIN_RAW_PATH,
        low_memory=False,
    )

    raw = add_datetime_column(raw)

    raw = raw.sort_values(
        [
            "station",
            "datetime",
        ]
    ).reset_index(drop=True)

    raw["split"] = assign_time_split(raw["datetime"])

    training_rows = raw.loc[raw["split"] == "train"].copy()

    print(f"Raw rows: {len(raw):,}")
    print(f"Training rows used to fit: {len(training_rows):,}")

    preprocessor = AirPollutionPreprocessor()

    print("Fitting training-only preprocessing parameters...")

    preprocessor.fit(training_rows)

    print("Transforming the complete chronological dataset...")

    processed = preprocessor.transform(raw)

    processed["split"] = assign_time_split(processed["datetime"])

    feature_columns = preprocessor.get_feature_names_out()

    feature_missing_count = int(processed[feature_columns].isna().sum().sum())

    if feature_missing_count != 0:
        raise ValueError("Processed features contain missing values.")

    raw_missing_targets = int(raw["PM2.5"].isna().sum())

    processed_missing_targets = int(processed[TARGET_OBSERVED_COLUMN].isna().sum())

    if raw_missing_targets != processed_missing_targets:
        raise ValueError("The original target missingness was not preserved.")

    print("Saving processed hourly dataset...")

    processed.to_parquet(
        PROCESSED_HOURLY_PATH,
        index=False,
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    split_summary = processed.groupby(
        "split",
        observed=True,
    ).agg(
        rows=("datetime", "size"),
        start_datetime=("datetime", "min"),
        end_datetime=("datetime", "max"),
        stations=("station", "nunique"),
        valid_targets=(
            TARGET_OBSERVED_COLUMN,
            "count",
        ),
    )

    split_summary["missing_targets"] = (
        split_summary["rows"] - split_summary["valid_targets"]
    )

    split_summary_path = METRICS_DIR / "preprocessing_split_summary.csv"

    split_summary.to_csv(split_summary_path)

    scaler = preprocessor.scaler_

    if scaler is None:
        raise RuntimeError("The fitted scaler is unavailable.")

    metadata = {
        "raw_rows": len(raw),
        "processed_rows": len(processed),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "training_medians": (preprocessor.medians_),
        "station_categories": (preprocessor.station_categories_),
        "scaler_mean": {
            column: float(value)
            for column, value in zip(
                preprocessor.numeric_columns,
                scaler.mean_,
            )
        },
        "scaler_standard_deviation": {
            column: float(value)
            for column, value in zip(
                preprocessor.numeric_columns,
                np.sqrt(scaler.var_),
            )
        },
        "missing_feature_values_after_processing": (feature_missing_count),
        "missing_targets_preserved": (processed_missing_targets),
        "processed_dataset_path": str(PROCESSED_HOURLY_PATH),
        "preprocessor_path": str(PREPROCESSOR_PATH),
    }

    metadata_path = METRICS_DIR / "preprocessing_metadata.json"

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\nSplit summary")
    print("-" * 60)
    print(split_summary.to_string())

    print("\nPreprocessing summary")
    print("-" * 60)
    print(f"Processed shape: {processed.shape}")
    print(f"Model features: {len(feature_columns)}")
    print(f"Feature NaNs: {feature_missing_count}")
    print(
        "Preserved missing targets:",
        processed_missing_targets,
    )
    print(f"Dataset: {PROCESSED_HOURLY_PATH}")
    print(f"Preprocessor: {PREPROCESSOR_PATH}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
