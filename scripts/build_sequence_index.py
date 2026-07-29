"""Build temporal sequence metadata for model training."""

from __future__ import annotations

import json

import joblib
import pandas as pd

from src.config import (
    PREPROCESSOR_PATH,
    PROCESSED_HOURLY_PATH,
    SEQUENCE_INDEX_PATH,
    SEQUENCE_METADATA_PATH,
    SEQUENCE_STATION_SUMMARY_PATH,
    SEQUENCE_SUMMARY_PATH,
    TARGET_OBSERVED_COLUMN,
    WINDOW_SIZE,
)
from src.sequence_builder import (
    build_sequence_index,
    extract_sequence,
    filter_sequence_split,
)


def main() -> None:
    """Create and validate the temporal sequence index."""

    if not PROCESSED_HOURLY_PATH.exists():
        raise FileNotFoundError(
            "Processed hourly data was not found. Run:\n"
            "python -m scripts.build_preprocessed_data"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "The fitted preprocessor was not found. Run:\n"
            "python -m scripts.build_preprocessed_data"
        )

    print("Loading processed hourly data...")

    processed = pd.read_parquet(PROCESSED_HOURLY_PATH)

    processed = processed.sort_values(["station", "datetime"]).reset_index(drop=True)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    feature_columns = preprocessor.get_feature_names_out()

    if TARGET_OBSERVED_COLUMN in feature_columns:
        raise ValueError(
            "The observed target column is present in the model feature list."
        )

    print(f"Hourly rows: {len(processed):,}")

    print(f"Input features: {len(feature_columns)}")

    print(f"Window size: {WINDOW_SIZE} hours")

    print("Constructing sequence index...")

    sequence_index = build_sequence_index(
        processed,
        feature_columns,
        window_size=WINDOW_SIZE,
    )

    if sequence_index.empty:
        raise ValueError("No valid sequences were generated.")

    duplicate_target_rows = int(sequence_index["target_row"].duplicated().sum())

    if duplicate_target_rows != 0:
        raise ValueError("A target row was assigned to more than one sequence.")

    invalid_window_lengths = (
        sequence_index["window_end_row"] - sequence_index["window_start_row"] + 1
        != WINDOW_SIZE
    )

    if invalid_window_lengths.any():
        raise ValueError("At least one sequence has an invalid window length.")

    print("Saving sequence index...")

    SEQUENCE_INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sequence_index.to_parquet(
        SEQUENCE_INDEX_PATH,
        index=False,
    )

    split_summary = sequence_index.groupby(
        "split",
        observed=True,
    ).agg(
        sequences=("sequence_id", "count"),
        stations=("station", "nunique"),
        first_target=(
            "target_datetime",
            "min",
        ),
        last_target=(
            "target_datetime",
            "max",
        ),
        target_mean=("target", "mean"),
        target_median=("target", "median"),
        target_standard_deviation=(
            "target",
            "std",
        ),
    )

    split_order = [
        "train",
        "validation",
        "local_test",
    ]

    split_summary = split_summary.reindex(split_order)

    split_summary.to_csv(SEQUENCE_SUMMARY_PATH)

    station_summary = (
        sequence_index.groupby(
            ["split", "station"],
            observed=True,
        )
        .agg(
            sequences=("sequence_id", "count"),
            first_target=(
                "target_datetime",
                "min",
            ),
            last_target=(
                "target_datetime",
                "max",
            ),
            target_mean=("target", "mean"),
        )
        .reset_index()
    )

    station_summary.to_csv(
        SEQUENCE_STATION_SUMMARY_PATH,
        index=False,
    )

    samples_by_split = {
        split: int((sequence_index["split"] == split).sum()) for split in split_order
    }

    metadata = {
        "window_size": WINDOW_SIZE,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "hourly_rows": len(processed),
        "station_count": int(processed["station"].nunique()),
        "total_sequences": len(sequence_index),
        "samples_by_split": samples_by_split,
        "skipped_missing_targets": int(processed[TARGET_OBSERVED_COLUMN].isna().sum()),
        "sequence_index_path": str(SEQUENCE_INDEX_PATH),
        "input_shape": [
            WINDOW_SIZE,
            len(feature_columns),
        ],
    }

    with SEQUENCE_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\nSequence summary")
    print("-" * 80)
    print(split_summary.to_string())

    print("\nExpected current-dataset counts")
    print("-" * 80)
    print("Train:       257,639")
    print("Validation:   25,458")
    print("Local test:   25,891")
    print("Total:       308,988")

    print("\nActual counts")
    print("-" * 80)

    for split, sample_count in samples_by_split.items():
        print(f"{split:<12}: {sample_count:>8,}")

    print(f"{'total':<12}: {len(sequence_index):>8,}")

    first_train_record = filter_sequence_split(
        sequence_index,
        "train",
    ).iloc[0]

    first_input, first_target = extract_sequence(
        processed,
        first_train_record,
        feature_columns,
    )

    print("\nFirst sequence check")
    print("-" * 80)
    print(
        "Station:",
        first_train_record["station"],
    )
    print(
        "Window:",
        first_train_record["window_start_datetime"],
        "to",
        first_train_record["window_end_datetime"],
    )
    print(
        "Target time:",
        first_train_record["target_datetime"],
    )
    print(
        "Input shape:",
        first_input.shape,
    )
    print(
        "Target:",
        float(first_target),
    )

    if first_input.shape != (
        WINDOW_SIZE,
        len(feature_columns),
    ):
        raise ValueError("The first sequence has an invalid input shape.")

    print("\n" + "=" * 80)
    print("SEQUENCE INDEX COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
