"""Verify the structure of the supplied CO5420 competition data."""

from __future__ import annotations

import pandas as pd

from src.config import (
    SAMPLE_SUBMISSION_PATH,
    TEST_PATH,
    TRAIN_RAW_PATH,
)

EXPECTED_TRAIN_COLUMNS = [
    "No",
    "year",
    "month",
    "day",
    "hour",
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "wd",
    "WSPM",
    "station",
]

SEQUENCE_FEATURES = [
    "year",
    "month",
    "day",
    "hour",
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "wd",
    "WSPM",
]


def require(condition: bool, message: str) -> None:
    """Raise a clear error when a dataset requirement is not satisfied."""
    if not condition:
        raise ValueError(message)


def verify_files_exist() -> None:
    """Check that all required files exist."""
    required_paths = [
        TRAIN_RAW_PATH,
        TEST_PATH,
        SAMPLE_SUBMISSION_PATH,
    ]

    missing = [str(path) for path in required_paths if not path.exists()]

    require(
        not missing,
        "Required dataset files are missing:\n" + "\n".join(missing),
    )


def verify_train_data(train: pd.DataFrame) -> None:
    """Validate raw training-data structure."""
    require(
        train.columns.tolist() == EXPECTED_TRAIN_COLUMNS,
        "train_raw.csv has unexpected columns.",
    )

    require(
        train.shape == (315_648, 18),
        f"Unexpected train shape: {train.shape}",
    )

    timestamps = pd.to_datetime(
        train[["year", "month", "day", "hour"]],
        errors="coerce",
    )

    require(
        timestamps.notna().all(),
        "Some training timestamps could not be constructed.",
    )

    station_times = pd.DataFrame(
        {
            "station": train["station"],
            "timestamp": timestamps,
        }
    )

    duplicate_count = station_times.duplicated().sum()

    require(
        duplicate_count == 0,
        f"Found {duplicate_count} duplicate station-timestamp records.",
    )

    print("\nTraining data")
    print("-" * 60)
    print(f"Shape: {train.shape}")
    print(f"Stations: {train['station'].nunique()}")
    print(f"Start: {timestamps.min()}")
    print(f"End:   {timestamps.max()}")

    missing_percent = train.isna().mean().mul(100).sort_values(ascending=False)

    print("\nMissing values (%):")
    print(missing_percent[missing_percent > 0].round(3).to_string())


def verify_test_data(test: pd.DataFrame) -> None:
    """Validate Kaggle test-data structure."""
    require(
        test.shape == (4_103, 386),
        f"Unexpected test shape: {test.shape}",
    )

    require(
        test.columns[0] == "id",
        "The first test column must be id.",
    )

    require(
        test.columns[1] == "station",
        "The second test column must be station.",
    )

    for lag in range(1, 25):
        expected_columns = {f"{feature}_lag_{lag}" for feature in SEQUENCE_FEATURES}

        actual_columns = {
            column for column in test.columns if column.endswith(f"_lag_{lag}")
        }

        require(
            actual_columns == expected_columns,
            f"Unexpected columns for lag {lag}.",
        )

    require(
        test["id"].is_unique,
        "Test IDs are not unique.",
    )

    print("\nKaggle test data")
    print("-" * 60)
    print(f"Shape: {test.shape}")
    print(f"Stations: {test['station'].nunique()}")
    print("Lag range: 24 hours to 1 hour")


def verify_submission(
    test: pd.DataFrame,
    submission: pd.DataFrame,
) -> None:
    """Validate sample-submission format."""
    require(
        submission.shape == (4_103, 2),
        f"Unexpected submission shape: {submission.shape}",
    )

    require(
        submission.columns.tolist() == ["id", "PM2.5"],
        "Submission columns must be exactly: id, PM2.5",
    )

    require(
        submission["id"].equals(test["id"]),
        "Sample-submission IDs do not match test IDs or order.",
    )

    print("\nSample submission")
    print("-" * 60)
    print(f"Shape: {submission.shape}")
    print("Columns: id, PM2.5")
    print("ID order matches test.csv")


def main() -> None:
    """Run all dataset checks."""
    verify_files_exist()

    print("Loading competition files...")

    train = pd.read_csv(TRAIN_RAW_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)
    submission = pd.read_csv(
        SAMPLE_SUBMISSION_PATH,
        low_memory=False,
    )

    verify_train_data(train)
    verify_test_data(test)
    verify_submission(test, submission)

    print("\n" + "=" * 60)
    print("DATA VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
