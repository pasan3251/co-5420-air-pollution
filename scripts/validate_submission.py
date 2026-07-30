"""Validate the generated final Kaggle submission."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    FINAL_SUBMISSION_PATH,
    SAMPLE_SUBMISSION_PATH,
)
from src.kaggle import (
    validate_submission_frame,
)


def main() -> None:
    """Validate format, IDs and prediction values."""

    if not FINAL_SUBMISSION_PATH.exists():
        raise FileNotFoundError(
            "Final submission does not exist. Run:\n"
            "python -m scripts.train_final_submission"
        )

    submission = pd.read_csv(
        FINAL_SUBMISSION_PATH
    )

    sample = pd.read_csv(
        SAMPLE_SUBMISSION_PATH
    )

    summary = validate_submission_frame(
        submission,
        sample,
    )

    digest = hashlib.sha256(
        FINAL_SUBMISSION_PATH.read_bytes()
    ).hexdigest()

    print("Final submission validation")
    print("-" * 60)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("sha256:", digest)
    print("path:", FINAL_SUBMISSION_PATH)

    print("\nSUBMISSION IS VALID")


if __name__ == "__main__":
    main()