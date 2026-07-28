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
