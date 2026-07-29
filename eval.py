import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math

# file_name = 'submission.csv'
# file_name = 'submission-feat-eng.csv'
file_name = 'submission_ensemble.csv'
# file_name = 'submission_regularized_ensemble.csv'

test = pd.read_csv('data/raw/test.csv')
sub = pd.read_csv(f'data/submissions/{file_name}')

# The target hour is 1 hour after lag_1
test['target_datetime'] = pd.to_datetime(test[['year_lag_1', 'month_lag_1', 'day_lag_1', 'hour_lag_1']].rename(columns={'year_lag_1':'year', 'month_lag_1':'month', 'day_lag_1':'day', 'hour_lag_1':'hour'})) + pd.Timedelta(hours=1)

test_raw = pd.read_csv('data/raw/test_raw.csv')
test_raw['target_datetime'] = pd.to_datetime(test_raw[['year', 'month', 'day', 'hour']])

# merge target
merged = test[['id', 'station', 'target_datetime']].merge(
    test_raw[['station', 'target_datetime', 'PM2.5']], 
    on=['station', 'target_datetime'], 
    how='left'
)

# merge predictions
final = merged.merge(sub, on='id', suffixes=('_true', '_pred'))

# compute rmse for non-nulls
valid = final.dropna(subset=['PM2.5_true', 'PM2.5_pred'])
if len(valid) == 0:
    print('No valid overlapping targets found!')
else:
    rmse = math.sqrt(mean_squared_error(valid['PM2.5_true'], valid['PM2.5_pred']))
    print(f'RMSE {file_name}: {rmse}')
    print(f'Calculated over {len(valid)} rows out of {len(final)}')
