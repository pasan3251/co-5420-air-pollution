#!/usr/bin/env bash

set -euo pipefail

echo "Activating project checks..."

python - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)

if not gpus:
    raise RuntimeError("No TensorFlow GPU was detected.")
PY

echo "Running tests..."
python -m pytest -q

echo "Training feedforward network..."
python -m scripts.train_feedforward_nn \
    --epochs 80 \
    --batch-size 512 \
    --learning-rate 0.001 \
    --dropout-rate 0.20 \
    --patience 10 \
    --seed 42

echo "Training LSTM..."
python -m scripts.train_recurrent_models \
    --models lstm \
    --epochs 50 \
    --batch-size 512 \
    --units 64 \
    --dense-units 32 \
    --dropout-rate 0.20 \
    --learning-rate 0.001 \
    --patience 8 \
    --seed 42

echo "Training GRU..."
python -m scripts.train_recurrent_models \
    --models gru \
    --epochs 50 \
    --batch-size 512 \
    --units 64 \
    --dense-units 32 \
    --dropout-rate 0.20 \
    --learning-rate 0.001 \
    --patience 8 \
    --seed 42

echo "Running GPU ablation grid..."
python -m scripts.run_temporal_ablation \
    --grid \
    --model gru \
    --batch-size 512 \
    --epochs 40 \
    --units 64 \
    --dense-units 32 \
    --dropout-rate 0.20 \
    --learning-rate 0.001 \
    --patience 6 \
    --force

echo "Running selected GRU across seeds..."
python -m scripts.run_temporal_ablation \
    --model gru \
    --window-sizes 24 \
    --feature-sets all_features \
    --seeds 42 123 2026 \
    --batch-size 512 \
    --epochs 40 \
    --units 64 \
    --dense-units 32 \
    --dropout-rate 0.20 \
    --learning-rate 0.001 \
    --patience 6 \
    --force

echo "All requested GPU experiments completed."
