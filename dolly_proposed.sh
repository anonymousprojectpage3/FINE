#!/bin/bash
set -e

python eval_dolly_proposed.py \
  --base_model Qwen/Qwen3-4B \
  --lora_weights_path ./train_results/dolly_data01_100/qwen4b/proposed/5 \
  --global_test_path ./DATA/data01_100/dolly/5/global_test.json \
  --num_clients 5 \
  --mode proposed \
  --train_pool_root ./DATA/data01_100/dolly/5 \
  --proposed_db_root ./train_results/dolly_data01_100/qwen4b/proposed/5/input_prompt \
  --k_min 1 \
  --k_max 3 \
  --seeds 42,43,44 \
  --save_dir ./eval_results/dolly_data01_100/qwen4b/proposed


