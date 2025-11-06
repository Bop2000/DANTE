#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6
export TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=2
export TF_FORCE_GPU_ALLOW_GROWTH=1
export DANTE_DEBUG_TIMING=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1

python -u experiments/rosenbrock_40d_opt.py \
  --data-dir data_raw/Rosenbrock-40d \
  --acquisitions 10000 \
  --samples-per-acquisition 20 \
  --epochs 500 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --rollout-round 200 \
  --ratio 0.8 \
  --nte-roots 3 \
  --turn 0.1 \
  2>&1 | tee -a ~/rosenbrock40_train.log

python -u experiments/schwefel_40d_opt.py \
  --data-dir data_raw/Schwefel-40d \
  --acquisitions 20000 \
  --samples-per-acquisition 20 \
  --epochs 500 \
  --batch-size 128 \
  --learning-rate 1e-3 \
  --rollout-round 200 \
  --ratio 1.0 \
  --nte-roots 3 \
  --turn 1.0 \
  2>&1 | tee -a ~/schwefel40_train.log
