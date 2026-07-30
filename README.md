# CO5420 Air Pollution Forecasting

Deep-learning project for one-hour-ahead PM2.5 forecasting from the previous
24 hours of pollutant, meteorological, calendar, and station observations.

- Competition: [CO5420 Air Pollution Forecasting Using Temporal NNs](https://www.kaggle.com/competitions/co-5420-air-pollution-forecasting-using-temporal-n-ns)
- Team: `DevOps` (five members)
- Competition metric: root mean squared error (RMSE; lower is better)
- Selected submission: `data/submissions/submission_ensemble.csv`
- Best recorded public score: **14.74409 RMSE**
- Recorded public rank: **8th**

The public leaderboard is provisional. Model selection is based on
chronological validation, with a later local-test period used once as a
generalisation check.

## Project status

| Stage | Status | Main result |
|---|---|---|
| Data audit and EDA | Complete | Verified hourly structure, missingness, target distribution, and split boundaries |
| Leakage-safe preprocessing | Complete | 43 model-input features with no missing inputs |
| Sequence construction | Complete | 308,988 valid 24-hour forecasting sequences |
| Traditional baselines | Complete | Persistence, historical mean, Ridge, and gradient boosting |
| Neural models | Complete | Feedforward NN, LSTM, and GRU |
| Temporal ablation | Complete | 6/12/24-hour and pollution-only/all-feature comparisons |
| Robustness analysis | Complete | Selected GRU evaluated across three random seeds |
| Model ensemble | Complete | Validation-selected feedforward–LSTM ensemble |
| Kaggle submission | Complete | Structurally validated 4,103-row CSV |
| Extended AQI task | Pending | Classification experiment proposed for the extended component |
| Presentation and viva | Pending | Slides, contribution evidence, and viva preparation |

## Prediction task

For a target time \(t\) at one monitoring station, the input is:

```text
t - 24, t - 23, ..., t - 2, t - 1
```

and the target is:

```text
PM2.5 at t
```

The official test set contains 4,103 independent flattened windows.
`lag_24` is the oldest observation and `lag_1` is the most recent.

## Dataset

The training data contains:

- 315,648 hourly rows;
- 12 Beijing monitoring stations;
- observations from 2013-03-01 through 2016-02-29;
- 309,276 observed PM2.5 targets;
- 6,372 missing PM2.5 observations.

Every station has a complete hourly timestamp sequence, with no duplicated
station–timestamp records. Missingness occurs in sensor values rather than in
the timeline itself. PM2.5 is strongly right-skewed, with a median of 56,
mean of 80.36, and maximum of 999.

## Data policy and competition integrity

The approved modelling and submission workflow uses only:

- `data/raw/train_raw.csv`
- `data/raw/test.csv`
- `data/raw/sample_submission.csv`

`test_raw.csv` is excluded from training, tuning, local submission scoring,
and final-submission selection because it exposes values from the competition
test period. Raw and generated data files are excluded from Git.

## Chronological evaluation

Rows and sequences are never randomly split. Membership is assigned by the
prediction target timestamp:

| Split | Target period | Sequences |
|---|---|---:|
| Training | Before 2015-09-01 | 257,639 |
| Validation | 2015-09-01 to 2015-11-30 | 25,458 |
| Local test | 2015-12-01 to 2016-02-29 | 25,891 |

The validation split controls architecture and ensemble decisions. The
local-test split is retained as a final internal check.

## Leakage-safe preprocessing

1. Sort rows by station and datetime.
2. Preserve the observed target separately as `target_PM2.5`.
3. Forward-fill numerical inputs using earlier observations from the same
   station only.
4. Fill remaining leading gaps with medians learned from the training split.
5. Never backward-fill.
6. Fit numerical scaling using the training split only.
7. Encode hour, weekday, and day of year cyclically.
8. Encode wind direction using sine and cosine components.
9. Represent station identity with one-hot features.
10. Retain numerical missingness indicators.

The processed hourly representation has 43 features:

- 11 scaled numerical variables;
- 6 cyclical time features;
- 3 wind-direction features;
- 11 numerical missingness indicators;
- 12 station indicators.

Each sample contains 24 consecutive input hours from exactly one station.
Samples with missing observed targets are excluded. A lightweight sequence
index is stored, and three-dimensional batches are generated on demand.

## Models and results

### Traditional baselines

Ridge and histogram gradient boosting use 115 engineered features:

- the latest 43 processed values;
- 24 PM2.5 lags;
- mean, standard deviation, minimum, and maximum for 11 numerical variables;
- PM2.5 changes over 1, 3, 6, and 12 hours.

| Model | Validation RMSE | Local-test RMSE |
|---|---:|---:|
| Ridge | 16.7889 | **26.0475** |
| Gradient boosting | **16.7294** | 31.1928 |
| Persistence | 17.6065 | 27.4611 |
| Historical mean | 50.7812 | 79.2034 |

### Neural models

The feedforward model uses the 115 engineered tabular features. The LSTM and
GRU process the unflattened `24 × 43` sequence directly. All neural models use
Adam, mean squared error, gradient clipping, early stopping, learning-rate
reduction, and best-checkpoint restoration.

| Model | Validation RMSE | Validation MAE | Local-test RMSE | Local-test MAE |
|---|---:|---:|---:|---:|
| Feedforward | 15.6733 | 8.7399 | 25.1482 | 11.9499 |
| LSTM | **15.6505** | **8.3936** | **25.9367** | **11.4252** |
| GRU | 16.0906 | 8.4478 | 27.4224 | 11.8377 |

The LSTM is the strongest recurrent architecture. The feedforward model
generalises better on local-test RMSE, showing that explicit lag and summary
features remain valuable for this one-hour horizon.

## Temporal and feature ablation

A controlled GRU grid compared 6-, 12-, and 24-hour windows:

| Window | Pollution-only RMSE | All-feature RMSE | Improvement from all features |
|---:|---:|---:|---:|
| 6 hours | 17.2793 | 16.3545 | 5.35% |
| 12 hours | 17.2064 | 16.2914 | 5.32% |
| 24 hours | 17.0593 | **16.1782** | 5.16% |

Meteorological inputs improve validation RMSE at every history length. The
24-hour all-feature GRU is best, although its gain over 12 hours is small
relative to the additional computation.

The selected GRU configuration was also tested across three seeds:

| Seed | Validation RMSE | Local-test RMSE |
|---:|---:|---:|
| 42 | 16.1782 | 27.2517 |
| 123 | 15.9583 | 26.7436 |
| 2026 | 16.2768 | 27.3588 |
| Mean ± sample SD | **16.1378 ± 0.1631** | **27.1180 ± 0.3287** |

## Selected feedforward–LSTM ensemble

The final model combines the two strongest complementary neural models:

\[
\hat{y}
=
0.49\hat{y}_{\mathrm{feedforward}}
+
0.51\hat{y}_{\mathrm{LSTM}}
\]

The weight was selected by searching from 0.00 to 1.00 in increments of 0.01
using validation RMSE only.

| Model | Validation RMSE | Validation MAE | Local-test RMSE | Local-test MAE |
|---|---:|---:|---:|---:|
| Feedforward | 15.6733 | 8.7399 | 25.1482 | 11.9499 |
| LSTM | 15.6505 | 8.3936 | 25.9367 | 11.4252 |
| **Selected ensemble** | **15.3365** | **8.2919** | **24.8339** | **11.3179** |

The ensemble improves validation RMSE by 2.01% relative to the LSTM and
local-test RMSE by 1.25% relative to the feedforward model.

## Selected Kaggle submission

The retained competition file is:

```text
data/submissions/submission_ensemble.csv
```

Submission checks:

- 4,103 rows and two columns: `id`, `PM2.5`;
- IDs exactly match `sample_submission.csv`;
- no duplicate IDs;
- no missing, infinite, or negative predictions;
- prediction range: 5.6579 to 577.8348;
- prediction mean: 76.6361;
- SHA-256:
  `19a6c9ff68d12cdd7f17d87d88845d7f2eedfa122994921e0d2c6afb472db219`;
- best recorded public score: **14.74409 RMSE**.

The public leaderboard result is reported as an observed competition outcome,
not as a tuning target.

## Notebook submission package

The module-submission notebook is:

```text
notebooks/07_kaggle_final_submission.ipynb
```

It is standalone and shows the complete workflow: competition-data loading,
leakage-safe preprocessing, 24-hour window construction, feedforward and LSTM
training, validation-only ensemble analysis, official-test prediction, and
submission-file creation. On Kaggle it writes:

```text
/kaggle/working/submission.csv
```

The locally executed matching CSV is:

```text
data/submissions/final_kaggle_notebook_submission.csv
```

All 14 code cells were executed successfully in the WSL GPU environment.
The resulting CSV contains 4,103 valid predictions and has SHA-256:
`a8038724c365dcf73e1fd199fc238c095c0525051af6e4e39ea8b767f70f6a07`.

## WSL/GPU environment

Development and neural training use Ubuntu under WSL:

```bash
source /home/wijer/miniconda3/etc/profile.d/conda.sh
conda activate co5420-air-gpu
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

The verified environment uses Python 3.11, TensorFlow 2.21.0, and an NVIDIA
GPU.

## Reproduce the workflow

Run from the repository root:

```bash
source /home/wijer/miniconda3/etc/profile.d/conda.sh
conda activate co5420-air-gpu

python -m scripts.verify_data
python -m scripts.build_preprocessed_data
python -m scripts.build_sequence_index
python -m scripts.run_baselines
python -m scripts.train_feedforward_nn
python -m scripts.train_recurrent_models
python -m scripts.run_temporal_ablation \
  --grid \
  --model gru \
  --batch-size 512
python -m scripts.evaluate_ensemble
python scripts/generate_submission.py --save-components
```

Run project checks:

```bash
ruff check src scripts tests
python -m pytest -q
```

## Notebook guide

| Notebook | Purpose |
|---|---|
| `01_data_audit.ipynb` | Dataset integrity, missingness, distributions, station patterns, and split design |
| `02_baselines.ipynb` | Traditional baselines and engineered feature design |
| `03_lstm.ipynb` | LSTM architecture, training behaviour, and error analysis |
| `04_gru.ipynb` | GRU performance, comparison with LSTM, and seed robustness |
| `05_ablation_analysis.ipynb` | Window-length and meteorological-feature ablation |
| `06_final_submission.ipynb` | Final model comparison, ensemble selection, and selected submission validation |
| `07_kaggle_final_submission.ipynb` | Standalone Kaggle workflow that trains the 49/51 ensemble and creates `submission.csv` |

## Repository structure

```text
data/
├── raw/                 # ignored competition inputs
├── processed/           # ignored reproducible intermediates
└── submissions/         # ignored Kaggle CSV files
notebooks/               # reader-facing analysis notebooks
results/
├── figures/             # generated visual evidence
├── metrics/             # tracked experiment summaries
└── predictions/         # ignored prediction-level outputs
scripts/                 # reproducible command-line workflows
src/                     # reusable preprocessing, modelling, and evaluation
tests/                   # automated safeguards
experiment_log.csv       # experiment trail
```

## Remaining deliverables

1. Import `07_kaggle_final_submission.ipynb` into Kaggle, attach the
   competition data, enable a GPU, run all cells, and save the final version.
2. Complete the proposed AQI classification extension using one cited AQI
   standard.
3. Finalise station-wise and severe-pollution interpretation.
4. Prepare presentation slides, contribution evidence, AI-use disclosure,
   and individual-viva notes.

All AI-assisted material must be reviewed, tested, understood, and explainable
by the team.
