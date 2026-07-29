import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import DEVICE, SEED, BATCH_SIZE, EPOCHS, WINDOW_SIZE, SUBMISSIONS_DIR
from src.utils import set_seed, save_submission, compute_metrics
from src.data_loader import load_and_preprocess_train_val, build_dataloaders, preprocess_test_data, create_sliding_windows
from src.models import ResidualLSTM, BiLSTMAttention, TemporalConvolutionalNetwork
from src.train import run_training_pipeline, predict_test, evaluate
from src.gbdt import train_lightgbm_model, predict_lightgbm

def main():
    set_seed(SEED)
    print(f"=== Beijing PM2.5 Advanced Super-Pipeline ===")
    print(f"Device: {DEVICE}")

    # 1. Data Prep & Physics Feature Engineering
    print("\n[1/6] Preprocessing, Dewpoint Spread & Physics Ratios...")
    train_df, val_df, scaler, feature_cols = load_and_preprocess_train_val(cutoff_date='2015-03-01 00:00:00')
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # 2. PyTorch DataLoaders
    print("\n[2/6] Constructing PyTorch DataLoaders...")
    train_loader, val_loader = build_dataloaders(train_df, val_df, feature_cols, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE)

    input_dim = len(feature_cols)

    # 3. Model 1: Baseline Residual LSTM (with Early Stopping)
    print("\n[3/6] Training Model 1: Baseline Residual LSTM...")
    model_lstm = ResidualLSTM(input_size=input_dim, hidden_size=64, num_layers=2, dropout=0.2)
    model_lstm, lstm_rmse = run_training_pipeline(
        model=model_lstm, train_loader=train_loader, val_loader=val_loader,
        scaler=scaler, feature_cols=feature_cols, epochs=EPOCHS, lr=0.001, use_huber=True, patience=3
    )

    # 4. Model 2: BiLSTM + Multi-Head Attention + Station Embeddings (with Early Stopping)
    print("\n[4/6] Training Model 2: BiLSTM + Multi-Head Attention + Station Embeddings...")
    model_attn = BiLSTMAttention(
        input_size=input_dim, embedding_dim=4, hidden_size=64,
        num_layers=2, num_heads=4, dropout=0.2
    )
    model_attn, attn_rmse = run_training_pipeline(
        model=model_attn, train_loader=train_loader, val_loader=val_loader,
        scaler=scaler, feature_cols=feature_cols, epochs=EPOCHS, lr=0.001, use_huber=True, patience=3
    )

    # 5. Model 3: Temporal Convolutional Network (TCN) (with Early Stopping)
    print("\n[5/6] Training Model 3: Temporal Convolutional Network (TCN)...")
    model_tcn = TemporalConvolutionalNetwork(input_size=input_dim, num_channels=[32, 64, 64], kernel_size=3, dropout=0.2)
    model_tcn, tcn_rmse = run_training_pipeline(
        model=model_tcn, train_loader=train_loader, val_loader=val_loader,
        scaler=scaler, feature_cols=feature_cols, epochs=EPOCHS, lr=0.001, use_huber=True, patience=3
    )

    # 6. Model 4: LightGBM GBDT Regressor
    print("\n[6/6] Training Model 4: LightGBM GBDT Regressor...")
    X_train_seq, st_train, y_train = create_sliding_windows(train_df, feature_cols, WINDOW_SIZE)
    X_val_seq, st_val, y_val = create_sliding_windows(val_df, feature_cols, WINDOW_SIZE)
    
    model_lgb, lgb_rmse = train_lightgbm_model(
        X_train_seq, st_train, y_train,
        X_val_seq, st_val, y_val,
        scaler, feature_cols, n_estimators=600, learning_rate=0.03
    )

    # Inference & Submissions
    print("\n=== Generating Predictions & Submissions ===")
    test_loader, test_ids = preprocess_test_data(scaler, feature_cols, window_size=WINDOW_SIZE)

    preds_lstm = np.clip(predict_test(model_lstm, test_loader, scaler, feature_cols), 0, None)
    preds_attn = np.clip(predict_test(model_attn, test_loader, scaler, feature_cols), 0, None)
    preds_tcn = np.clip(predict_test(model_tcn, test_loader, scaler, feature_cols), 0, None)
    preds_lgb = np.clip(predict_lightgbm(model_lgb, test_loader, scaler, feature_cols), 0, None)

    save_submission(test_ids, preds_lstm, "submission_modular_lstm.csv")
    save_submission(test_ids, preds_attn, "submission_modular_bilstm_attn.csv")
    save_submission(test_ids, preds_tcn, "submission_modular_tcn.csv")
    save_submission(test_ids, preds_lgb, "submission_lightgbm.csv")

    # Super-Ensemble Blend (BiLSTM-Attn 35%, LightGBM 35%, TCN 20%, LSTM 10%)
    super_ensemble = 0.35 * preds_attn + 0.35 * preds_lgb + 0.20 * preds_tcn + 0.10 * preds_lstm
    save_submission(test_ids, super_ensemble, "submission_super_ensemble.csv")

    # Super-Ensemble Blend (BiLSTM-Attn 10%, LightGBM 35%, TCN 20%, LSTM 35%)
    super_ensemble = 0.10 * preds_attn + 0.35 * preds_lgb + 0.20 * preds_tcn + 0.35 * preds_lstm
    save_submission(test_ids, super_ensemble, "submission_super_ensemble_new.csv")

    # Evaluation against test_raw.csv ground truth
    print("\n=== GROUND TRUTH TEST EVALUATION ===")
    test = pd.read_csv(os.path.join(ROOT_DIR, 'data/raw/test.csv'))
    test_raw = pd.read_csv(os.path.join(ROOT_DIR, 'data/raw/test_raw.csv'))

    test['target_datetime'] = pd.to_datetime(
        test[['year_lag_1', 'month_lag_1', 'day_lag_1', 'hour_lag_1']]
        .rename(columns={'year_lag_1':'year', 'month_lag_1':'month', 'day_lag_1':'day', 'hour_lag_1':'hour'})
    ) + pd.Timedelta(hours=1)
    test_raw['target_datetime'] = pd.to_datetime(test_raw[['year', 'month', 'day', 'hour']])

    merged = test[['id', 'station', 'target_datetime']].merge(
        test_raw[['station', 'target_datetime', 'PM2.5']], on=['station', 'target_datetime'], how='left'
    )

    for name, filename in [
        ("Modular LSTM", "submission_modular_lstm.csv"),
        ("BiLSTM Attention", "submission_modular_bilstm_attn.csv"),
        ("TCN", "submission_modular_tcn.csv"),
        ("LightGBM GBDT", "submission_lightgbm.csv"),
        ("Super Ensemble", "submission_super_ensemble.csv")
    ]:
        sub_filepath = os.path.join(SUBMISSIONS_DIR, filename)
        if os.path.exists(sub_filepath):
            sub_df = pd.read_csv(sub_filepath)
            final_df = merged.merge(sub_df, on='id', suffixes=('_true', '_pred'))
            valid_df = final_df.dropna(subset=['PM2.5_true', 'PM2.5_pred'])
            rmse_score = np.sqrt(np.mean((valid_df['PM2.5_true'] - valid_df['PM2.5_pred'])**2))
            print(f"{name:22s} ({filename}): Ground Truth Test RMSE = {rmse_score:.4f}")

if __name__ == '__main__':
    main()
