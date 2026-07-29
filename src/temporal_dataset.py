"""Memory-efficient temporal batches for recurrent models."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import tensorflow as tf


class TemporalWindowDataset(tf.keras.utils.PyDataset):
    """
    Generate temporal batches from a feature matrix and sequence index.

    The complete 3D sequence dataset is not stored in memory. Each batch
    is assembled from the processed hourly feature matrix when requested.
    """

    def __init__(
        self,
        feature_matrix: np.ndarray,
        sequence_index: pd.DataFrame,
        *,
        window_size: int,
        batch_size: int = 256,
        shuffle: bool = False,
        seed: int = 42,
        return_targets: bool = True,
    ) -> None:
        """Initialise the temporal dataset."""

        super().__init__()

        features = np.asarray(
            feature_matrix,
            dtype=np.float32,
        )

        if features.ndim != 2:
            raise ValueError(
                "feature_matrix must be a two-dimensional array."
            )

        if not np.isfinite(features).all():
            raise ValueError(
                "feature_matrix contains missing or infinite values."
            )

        if sequence_index.empty:
            raise ValueError(
                "sequence_index must not be empty."
            )

        required_columns = {
            "window_start_row",
            "window_end_row",
            "target",
        }

        missing_columns = required_columns.difference(
            sequence_index.columns
        )

        if missing_columns:
            raise ValueError(
                "sequence_index is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        if window_size <= 0:
            raise ValueError(
                "window_size must be positive."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        start_rows = sequence_index[
            "window_start_row"
        ].to_numpy(dtype=np.int64)

        end_rows = sequence_index[
            "window_end_row"
        ].to_numpy(dtype=np.int64)

        window_lengths = (
            end_rows - start_rows + 1
        )

        if not np.all(
            window_lengths == window_size
        ):
            raise ValueError(
                "At least one indexed window has an "
                "unexpected length."
            )

        if start_rows.min() < 0:
            raise ValueError(
                "Window row positions must not be negative."
            )

        if end_rows.max() >= len(features):
            raise ValueError(
                "A window row position exceeds the "
                "feature matrix bounds."
            )

        targets = sequence_index[
            "target"
        ].to_numpy(dtype=np.float32)

        if not np.isfinite(targets).all():
            raise ValueError(
                "Sequence targets contain missing or "
                "infinite values."
            )

        self.feature_matrix = features
        self.start_rows = start_rows
        self.targets = targets

        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_targets = return_targets

        self.window_offsets = np.arange(
            window_size,
            dtype=np.int64,
        )

        self.sample_order = np.arange(
            len(sequence_index),
            dtype=np.int64,
        )

        self.random_generator = (
            np.random.default_rng(seed)
        )

        if self.shuffle:
            self.random_generator.shuffle(
                self.sample_order
            )

    def __len__(self) -> int:
        """Return the number of batches per epoch."""

        return math.ceil(
            len(self.sample_order)
            / self.batch_size
        )

    def __getitem__(
        self,
        batch_index: int,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Build and return one batch."""

        if batch_index < 0 or batch_index >= len(self):
            raise IndexError(
                f"Batch index {batch_index} is out of range."
            )

        batch_start = (
            batch_index * self.batch_size
        )

        batch_end = min(
            batch_start + self.batch_size,
            len(self.sample_order),
        )

        selected_samples = self.sample_order[
            batch_start:batch_end
        ]

        selected_start_rows = self.start_rows[
            selected_samples
        ]

        row_positions = (
            selected_start_rows[:, None]
            + self.window_offsets[None, :]
        )

        inputs = self.feature_matrix[
            row_positions
        ].astype(
            np.float32,
            copy=False,
        )

        if not self.return_targets:
            return inputs

        targets = self.targets[
            selected_samples
        ][:, None]

        return inputs, targets

    def on_epoch_end(self) -> None:
        """Shuffle the training sample order between epochs."""

        if self.shuffle:
            self.random_generator.shuffle(
                self.sample_order
            )

    @property
    def sample_count(self) -> int:
        """Return the number of forecasting samples."""

        return len(self.sample_order)

    @property
    def input_shape(self) -> tuple[int, int]:
        """Return the shape of one temporal input sample."""

        return (
            self.window_size,
            self.feature_matrix.shape[1],
        )