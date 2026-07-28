"""Leakage-safe preprocessing for the air-pollution dataset."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    LOCAL_TEST_START,
    NUMERIC_INPUT_COLUMNS,
    STATION_COLUMN,
    TARGET_COLUMN,
    TARGET_OBSERVED_COLUMN,
    TIME_COLUMNS,
    VALIDATION_START,
    WIND_DIRECTION_COLUMN,
)

WIND_DIRECTION_TO_DEGREES = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}


TIME_FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

WIND_FEATURE_COLUMNS = [
    "wd_sin",
    "wd_cos",
    "wd_missing",
]


def add_datetime_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct and validate a datetime column."""

    result = frame.copy()

    if "datetime" not in result.columns:
        missing_columns = [
            column for column in TIME_COLUMNS if column not in result.columns
        ]

        if missing_columns:
            raise ValueError(
                "Cannot construct datetime. Missing columns: "
                + ", ".join(missing_columns)
            )

        result["datetime"] = pd.to_datetime(
            result[TIME_COLUMNS],
            errors="raise",
        )

    else:
        result["datetime"] = pd.to_datetime(
            result["datetime"],
            errors="raise",
        )

    return result


def assign_time_split(
    datetimes: pd.Series,
) -> pd.Categorical:
    """Assign train, validation and local-test periods."""

    datetime_values = pd.to_datetime(
        datetimes,
        errors="raise",
    )

    validation_start = pd.Timestamp(VALIDATION_START)
    local_test_start = pd.Timestamp(LOCAL_TEST_START)

    split = pd.Series(
        "local_test",
        index=datetimes.index,
        dtype="object",
    )

    split.loc[datetime_values < local_test_start] = "validation"

    split.loc[datetime_values < validation_start] = "train"

    return pd.Categorical(
        split,
        categories=[
            "train",
            "validation",
            "local_test",
        ],
        ordered=True,
    )


@dataclass
class AirPollutionPreprocessor:
    """Preprocess hourly observations without future leakage."""

    numeric_columns: tuple[str, ...] = tuple(NUMERIC_INPUT_COLUMNS)

    station_column: str = STATION_COLUMN
    wind_direction_column: str = WIND_DIRECTION_COLUMN
    target_column: str = TARGET_COLUMN
    target_observed_column: str = TARGET_OBSERVED_COLUMN

    medians_: dict[str, float] = field(
        default_factory=dict,
        init=False,
    )

    station_categories_: list[str] = field(
        default_factory=list,
        init=False,
    )

    scaler_: StandardScaler | None = field(
        default=None,
        init=False,
        repr=False,
    )

    feature_columns_: list[str] = field(
        default_factory=list,
        init=False,
    )

    fitted_: bool = field(
        default=False,
        init=False,
    )

    def fit(
        self,
        frame: pd.DataFrame,
    ) -> AirPollutionPreprocessor:
        """Fit imputation and scaling parameters on training data."""

        prepared = self._prepare_frame(frame)

        medians = prepared[list(self.numeric_columns)].median()

        if medians.isna().any():
            invalid_columns = medians[medians.isna()].index.tolist()

            raise ValueError(
                "Cannot calculate training medians for: " + ", ".join(invalid_columns)
            )

        self.medians_ = {column: float(value) for column, value in medians.items()}

        self.station_categories_ = sorted(
            prepared[self.station_column].dropna().astype(str).unique().tolist()
        )

        if not self.station_categories_:
            raise ValueError("No station categories were found.")

        engineered = self._engineer_features(
            prepared,
            apply_scaling=False,
        )

        self.scaler_ = StandardScaler()

        self.scaler_.fit(engineered[list(self.numeric_columns)])

        missing_indicator_columns = [
            f"{column}_missing" for column in self.numeric_columns
        ]

        station_feature_columns = [
            f"station_{station}" for station in self.station_categories_
        ]

        self.feature_columns_ = (
            list(self.numeric_columns)
            + TIME_FEATURE_COLUMNS
            + WIND_FEATURE_COLUMNS
            + missing_indicator_columns
            + station_feature_columns
        )

        self.fitted_ = True

        return self

    def transform(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply fitted preprocessing parameters."""

        self._require_fitted()

        prepared = self._prepare_frame(frame)

        transformed = self._engineer_features(
            prepared,
            apply_scaling=True,
        )

        output_columns = [
            "datetime",
            self.station_column,
            self.target_observed_column,
            *self.feature_columns_,
        ]

        result = transformed[output_columns].copy()

        feature_missing_count = int(result[self.feature_columns_].isna().sum().sum())

        if feature_missing_count != 0:
            raise ValueError(
                "Preprocessed feature matrix still contains "
                f"{feature_missing_count} missing values."
            )

        return result

    def fit_transform(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fit the preprocessor and transform the same frame."""

        return self.fit(frame).transform(frame)

    def get_feature_names_out(self) -> list[str]:
        """Return model-input feature names."""

        self._require_fitted()

        return self.feature_columns_.copy()

    def _prepare_frame(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate, timestamp and sort an hourly dataframe."""

        required_columns = [
            self.station_column,
            self.wind_direction_column,
            *self.numeric_columns,
        ]

        missing_columns = [
            column for column in required_columns if column not in frame.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing preprocessing columns: " + ", ".join(missing_columns)
            )

        prepared = add_datetime_column(frame)

        if prepared[self.station_column].isna().any():
            raise ValueError("Station values must not be missing.")

        prepared[self.station_column] = prepared[self.station_column].astype(str)

        if self.target_observed_column not in prepared:
            prepared[self.target_observed_column] = prepared[self.target_column]

        prepared = prepared.sort_values(
            [
                self.station_column,
                "datetime",
            ]
        ).reset_index(drop=True)

        return prepared

    def _engineer_features(
        self,
        frame: pd.DataFrame,
        *,
        apply_scaling: bool,
    ) -> pd.DataFrame:
        """Impute and generate temporal/categorical features."""

        result = frame.copy()

        numeric_columns = list(self.numeric_columns)

        for column in numeric_columns:
            result[f"{column}_missing"] = result[column].isna().astype("float32")

        # Forward fill uses only earlier values from the same station.
        result[numeric_columns] = result.groupby(
            self.station_column,
            sort=False,
        )[numeric_columns].ffill()

        # Leading missing values cannot be forward-filled. They use
        # medians learned exclusively from the training split.
        result[numeric_columns] = result[numeric_columns].fillna(self.medians_)

        self._add_time_features(result)
        self._add_wind_features(result)
        self._add_station_features(result)

        if apply_scaling:
            if self.scaler_ is None:
                raise RuntimeError("The numerical scaler is unavailable.")

            scaled_values = self.scaler_.transform(result[numeric_columns])

            result[numeric_columns] = scaled_values.astype("float32")

        else:
            result[numeric_columns] = result[numeric_columns].astype("float32")

        return result

    def _add_time_features(
        self,
        frame: pd.DataFrame,
    ) -> None:
        """Add cyclical time features in place."""

        datetime_values = frame["datetime"]

        hour = datetime_values.dt.hour.astype(float)
        day_of_week = datetime_values.dt.dayofweek.astype(float)
        day_of_year = datetime_values.dt.dayofyear.astype(float) - 1.0

        frame["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype("float32")

        frame["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype("float32")

        frame["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0).astype(
            "float32"
        )

        frame["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0).astype(
            "float32"
        )

        frame["day_of_year_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25).astype(
            "float32"
        )

        frame["day_of_year_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25).astype(
            "float32"
        )

    def _add_wind_features(
        self,
        frame: pd.DataFrame,
    ) -> None:
        """Encode compass direction as circular features."""

        wind_values = frame[self.wind_direction_column].astype("string").str.upper()

        degrees = wind_values.map(WIND_DIRECTION_TO_DEGREES)

        unknown_direction = degrees.isna()

        radians = np.deg2rad(degrees.fillna(0.0).astype(float))

        frame["wd_sin"] = np.sin(radians).astype("float32")

        frame["wd_cos"] = np.cos(radians).astype("float32")

        frame["wd_missing"] = unknown_direction.astype("float32")

    def _add_station_features(
        self,
        frame: pd.DataFrame,
    ) -> None:
        """Add stable station one-hot features."""

        for station in self.station_categories_:
            column = f"station_{station}"

            frame[column] = frame[self.station_column].eq(station).astype("float32")

    def _require_fitted(self) -> None:
        """Raise an error when transform is called before fit."""

        if not self.fitted_:
            raise RuntimeError("The preprocessor must be fitted first.")
