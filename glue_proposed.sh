#!/bin/bash
set -e

SEEDS="42,43,44"

python train_glue_proposed.py icl_fl \
  --global_model meta-llama/Llama-3.2-3B \
  --data_path ./DATA/data01_300/GLUE \
  --output_dir ./train_results/glue_data01_300/llama3b/proposed_e1_r10 \
  --shot 0 \
  --num_clients 5 \
  --num_communication_rounds 10 \
  --local_num_epochs 1 \
  --local_learning_rate 3e-4 \
  --use_bf16 False \
  --allow_tf32 True 

