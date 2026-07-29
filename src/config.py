"""Central configuration for the CO5420 air-pollution project."""

from pathlib import Path

# Project directories
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Input files
TRAIN_RAW_PATH = RAW_DATA_DIR / "train_raw.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"

# Project constants
TARGET_COLUMN = "PM2.5"
STATION_COLUMN = "station"
ID_COLUMN = "id"

TIME_COLUMNS = ["year", "month", "day", "hour"]

POLLUTANT_COLUMNS = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
]

WEATHER_COLUMNS = [
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "wd",
    "WSPM",
]

WINDOW_SIZE = 24
RANDOM_SEED = 42

# Leakage-safe chronological split boundaries
VALIDATION_START = "2015-09-01 00:00:00"
LOCAL_TEST_START = "2015-12-01 00:00:00"

# The original observed PM2.5 value is preserved separately because
# the PM2.5 feature itself will be imputed and scaled.
TARGET_OBSERVED_COLUMN = "target_PM2.5"

WIND_DIRECTION_COLUMN = "wd"

NUMERIC_INPUT_COLUMNS = [
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
    "WSPM",
]

PROCESSED_HOURLY_PATH = PROCESSED_DATA_DIR / "hourly_features.parquet"

PREPROCESSOR_PATH = PROCESSED_DATA_DIR / "air_pollution_preprocessor.joblib"

# Sequence construction
SEQUENCE_INDEX_PATH = PROCESSED_DATA_DIR / "sequence_index.parquet"

SEQUENCE_SUMMARY_PATH = METRICS_DIR / "sequence_split_summary.csv"

SEQUENCE_STATION_SUMMARY_PATH = METRICS_DIR / "sequence_station_summary.csv"

SEQUENCE_METADATA_PATH = METRICS_DIR / "sequence_metadata.json"

# Baseline model artefacts
RIDGE_BASELINE_PATH = PROCESSED_DATA_DIR / "ridge_baseline.joblib"

GRADIENT_BOOSTING_BASELINE_PATH = (
    PROCESSED_DATA_DIR / "hist_gradient_boosting_baseline.joblib"
)

BASELINE_FEATURE_NAMES_PATH = METRICS_DIR / "baseline_feature_names.json"

BASELINE_METRICS_PATH = METRICS_DIR / "baseline_metrics.csv"

BASELINE_STATION_METRICS_PATH = METRICS_DIR / "baseline_station_metrics.csv"

BASELINE_PREDICTIONS_DIR = PREDICTIONS_DIR / "baselines"
