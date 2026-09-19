#!/usr/bin/env bash
# Phase 4: train the three setups back-to-back for a clean comparison.
# Each run is independent — a crash in one does not stop the others.
cd /app/src || exit 1
STEPS=1000000

echo "[$(date)] === START ppo (baseline) ==="
python train_ppo.py --timesteps $STEPS --n-envs 8 || echo "[$(date)] ppo FAILED"

echo "[$(date)] === START curiosity (game + curiosity) ==="
python train_curiosity.py --timesteps $STEPS --n-envs 8 || echo "[$(date)] curiosity FAILED"

echo "[$(date)] === START pure_curiosity (curiosity only) ==="
python train_curiosity.py --timesteps $STEPS --extrinsic-coef 0 --n-envs 8 || echo "[$(date)] pure_curiosity FAILED"

echo "[$(date)] === ALL DONE ==="
