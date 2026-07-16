#!/bin/bash
set -e

SEEDS="42,43,44"

python train_mmlu_proposed.py icl_fl \
  --global_model meta-llama/Llama-3.2-3B \
  --data_path ./DATA/data01_100/MMLU \
  --output_dir ./train_results/mmlu_data01_100/llama3b/proposed_e1_r10 \
  --shot 0 \
  --num_clients 5 \
  --num_communication_rounds 10 \
  --local_num_epochs 1 \
  --local_learning_rate 3e-4 \
  --use_bf16 False \
  --allow_tf32 True 
python eval_mmlu_proposed.py \
  --base_model meta-llama/Llama-3.2-3B \
  --global_test_path ./DATA/data01_100/MMLU/5/global_test.json \
  --num_clients 5 \
  --mode proposed \
  --train_pool_root ./DATA/data01_100/MMLU/5 \
  --proposed_db_root ./train_results/mmlu_data01_100/llama3b/proposed_e1_r10/5/input_prompt \
  --lora_weights_path ./train_results/mmlu_data01_100/llama3b/proposed_e1_r10/5 \
  --k_min 1 --k_max 3 \
  --seeds $SEEDS \
  --save_dir ./eval_results/mmlu_data01_100/llama3b/proposed_e1_r10

