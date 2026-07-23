# FINE: Federated Meta In-Context Learning via Informative Example Curation

Official implementation of **FINE**, a federated Meta-ICL framework that combines
mutual information based example curation with consensus-aware federated aggregation
to improve ICL capability across heterogeneous distributed clients.

[Project Page](https://anonymousprojectpage3.github.io/FINE/)

## Overview

FINE consists of two components:
1. **MI-based example curation** — automatically constructs an informative
   client-specific example database using a mutual information criterion.
2. **Consensus-aware federated aggregation** — integrates ICL adaptation
   signals across heterogeneous clients through parameter-level consensus.

## Requirements

```bash
pip install torch transformers datasets pandas numpy
```

## Training

```bash
bash glue_proposed.sh    # GLUE
bash mmlu_proposed.sh    # MMLU
bash dolly_proposed.sh   # Dolly-15k
```

Each script runs federated Meta-ICL training (`train_*_proposed.py`) with MI-based
example curation and consensus-aware aggregation over multiple communication rounds.

## Evaluation

```bash
python eval_glue_proposed.py \
  --base_model <model_name> \
  --lora_weights_path <path_to_trained_lora> \
  --global_test_path <path_to_test_json> \
  --mode proposed \
  --seeds 42,43,44
```

Analogous `eval_mmlu_proposed.py` and `eval_dolly_proposed.py` scripts are provided
for MMLU and Dolly-15k.
