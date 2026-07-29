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
| Validation | 15.7176 | 8.7623 | 0.9677 |
| Local test | 25.2410 | 11.9686 | 0.9512 |