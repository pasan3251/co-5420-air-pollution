# CO5420 Air Pollution Project

Repository scaffold for the Neural Networks and Deep Learning project.

## Structure

- `data/` for raw and processed datasets
- `notebooks/` for exploratory work, baselines, and model experiments
- `src/` for reusable preprocessing, modeling, evaluation, and submission code
- `results/` for figures, metrics, and prediction outputs

## Starter workflow

1. Audit the dataset in `notebooks/01_data_audit.ipynb`
2. Build baselines in `notebooks/02_baselines.ipynb`
3. Train recurrent models in `notebooks/03_lstm.ipynb` and `notebooks/04_gru.ipynb`
4. Run ablation analysis in `notebooks/05_ablation_analysis.ipynb`
5. Prepare the final Kaggle submission in `notebooks/06_final_submission.ipynb`

## Dataset policy

The modelling pipeline uses only:

- `data/raw/train_raw.csv`
- `data/raw/test.csv`
- `data/raw/sample_submission.csv`

Raw and processed datasets are excluded from Git because they can be
reproduced from the official competition files.

`test_raw.csv` is not used by the modelling, validation, feature-engineering,
or submission pipelines.

## Prediction task

Given 24 consecutive hourly observations from one monitoring station,
predict the station's PM2.5 concentration for the following hour.

## Reproducibility

All experiments must record:

- Author
- Git branch
- Model
- Feature set
- Window size
- Data split
- Random seed
- Validation RMSE
- Validation MAE
- Important observations

## Leakage-safe preprocessing

The preprocessing pipeline follows these rules:

1. Observations are sorted by station and datetime.
2. The original PM2.5 observation is preserved as `target_PM2.5`.
3. Numerical input features are forward-filled only within the same station.
4. Remaining leading missing values use medians calculated from the training split.
5. Backward filling is not used because it can expose future measurements.
6. The numerical scaler is fitted using the training split only.
7. Hour, day of week, and day of year are encoded cyclically.
8. Wind direction is represented using sine and cosine components.
9. Station identity is represented using one-hot features.
10. Missing-value indicators are retained as model inputs.

The chronological split is:

- Training: before 2015-09-01
- Validation: 2015-09-01 to 2015-11-30
- Local test: from 2015-12-01

Split membership is based on the prediction target timestamp.

## Temporal sequence construction

Each forecasting sample uses 24 historical hourly observations from one
monitoring station to predict PM2.5 at the following hour.

For a target time `t`, the input is:

```text
t - 24, t - 23, ..., t - 2, t - 1
```
and the prediction target is:

```text
PM2.5 at t
```

Sequence-building rules:

1. A sequence never crosses a station boundary.
2. All 24 input timestamps must be consecutive.
3. The target timestamp must be exactly one hour after the final input.
4. Samples with missing true PM2.5 targets are excluded.
5. target_PM2.5 is never included in the input feature list.
6. Split membership is determined by the target timestamp.
7. Validation and local-test windows may use earlier historical observations from the preceding split because those values would be available at prediction time.
8. The full 3D training tensor is not saved. A lightweight sequence index is stored and batches are generated when required.


Current input shape:

```text
24 time steps × 43 features
```

## Forecasting baselines

The project evaluates four traditional forecasting baselines:

1. **Persistence:** use PM2.5 from the final input hour.
2. **Historical mean:** use mean PM2.5 over the previous 24 hours.
3. **Ridge regression:** learn a regularised linear relationship from lag,
   latest-hour, temporal, station, weather and summary features.
4. **Histogram gradient boosting:** learn nonlinear relationships from the
   same tabular feature representation.

Ridge and gradient boosting use 115 derived features:

- latest values of all 43 sequence features;
- all 24 historical PM2.5 lags;
- mean, standard deviation, minimum and maximum for 11 numerical features;
- PM2.5 changes over 1, 3, 6 and 12 hours.

Model selection uses validation RMSE. The local-test period is retained as
a final internal generalisation check and is not used for hyperparameter
selection.

Reported metrics include:

- RMSE;
- MAE;
- R-squared;
- station-wise performance;
- performance across PM2.5 concentration ranges.

## Feedforward neural-network baseline

The feedforward network uses the same 115 engineered features as Ridge
regression and histogram gradient boosting. This provides a fair comparison
between traditional models and neural learning without recurrent sequence
processing.

Architecture:

```text
Input(115)
→ Dense(128, ReLU)
→ Dropout(0.20)
→ Dense(64, ReLU)
→ Dropout(0.15)
→ Dense(32, ReLU)
→ Dense(1, Linear)
```

Training configuration:

- Adam optimizer
- Mean squared error loss
- Validation RMSE for model selection
- Gradient clipping with norm 1.0
- Early stopping
- Learning-rate reduction on validation plateaus
- Best-model checkpointing
- Random seed 42

The dense model predicts PM2.5 directly in its original concentration units.
Negative predictions are clipped to zero during evaluation.


### Feedforward results

| Split | RMSE | MAE | R² |
|---|---:|---:|---:|
| Validation | 15.6733 | 8.7399 | 0.9679 |
| Local test | 25.1482 | 11.9499 | 0.9516 |

## LSTM and GRU temporal models

The recurrent models receive the original temporal structure:

```text
24 historical hours × 43 features
```

Both models use equivalent configurations for a fair comparison.

LSTM architecture

```text
Input(24, 43)
→ LSTM(64)
→ Layer Normalisation
→ Dropout(0.20)
→ Dense(32, ReLU)
→ Dropout(0.10)
→ Dense(1, Linear)
```

GRU architecture

```text
Input(24, 43)
→ GRU(64)
→ Layer Normalisation
→ Dropout(0.20)
→ Dense(32, ReLU)
→ Dropout(0.10)
→ Dense(1, Linear)
```

Training uses:

- Adam optimisation;
- mean squared error loss;
- validation RMSE for checkpoint selection;
- gradient clipping;
- early stopping;
- learning-rate reduction;
- deterministic random seed 42;
- identical train, validation and local-test sequences.

Temporal batches are generated on demand instead of storing the complete three-dimensional dataset in memory.

### Recurrent results

| Model | Validation RMSE | Validation MAE | Local-test RMSE | Local-test MAE |
|---|---:|---:|---:|---:|
| LSTM | 15.6505 | 8.3936 | 25.9367 | 11.4252 |
| GRU | 16.0906 | 8.4478 | 27.4224 | 11.8377 |

## Temporal ablation design

A controlled GRU experiment evaluates two project questions:

1. How much historical context is useful for one-hour-ahead prediction?
2. Do meteorological variables improve forecasting beyond pollutant history?

The experiment grid contains:

| Window | Pollution-only | All features |
|---:|---:|---:|
| 6 hours | Yes | Yes |
| 12 hours | Yes | Yes |
| 24 hours | Yes | Yes |

All configurations use the same target samples, train-validation split,
architecture, optimiser, units, batch size, stopping policy and random seed.

The pollution-only feature set retains pollutant measurements, pollutant
missing-value indicators, cyclical time features and station identity. It
excludes temperature, pressure, dew point, rain, wind speed and wind
direction.

Configuration selection uses validation RMSE only. Local-test evaluation is
performed only after the final configuration has been frozen.

### Temporal ablation findings

Meteorological information improved GRU validation RMSE at every tested
window length:

| Window | Pollution-only RMSE | All-features RMSE |
|---:|---:|---:|
| 6 hours | 17.2793 | 16.3545 |
| 12 hours | 17.2064 | 16.2914 |
| 24 hours | 17.0593 | 16.1782 |

The 24-hour all-features GRU produced the best single-run validation result.
However, the improvement over 12 hours was small relative to its additional
training time.

The selected recurrent configuration is:

- Architecture: GRU
- Window: 24 hours
- Features: all 43 features
- GRU units: 64
- Dense units: 32
- Dropout: 0.20
- Batch size: 512
- Learning rate: 0.001

### Random-seed robustness

| Seed | Validation RMSE | Validation MAE | Best epoch |
|---:|---:|---:|---:|
| 42 | 16.178228 | 8.501350 | 19 |
| 123 | 15.958253 | 8.434038 | 18 |
| 2026 | 16.276832 | 8.450451 | 11 |

Mean validation RMSE: 16.137771069870812

Validation RMSE standard deviation: 0.16309733424471895