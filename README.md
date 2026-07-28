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