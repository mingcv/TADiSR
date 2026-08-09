#!/usr/bin/env bash
set -euo pipefail

# Fill in these paths before running. See README.md for the complete protocol.
accelerate launch train_cogview4_realce.py \
  --train_folders /path/to/FTSR \
  --test_folder /path/to/RealCE \
  --realce_train_list data/RealCE/aligned_list_train.txt \
  --realce_eval_list data/RealCE/aligned_list_eval.txt \
  --pretrained_model_name_or_path weights/CogView4 \
  --prompt_embeddings weights/CogView4/saved_prompt_tokens_nocfg.pt \
  --output_dir output/TADiSR/TADiSR_CogView4_RealCE \
  --learning_rate 2e-5 \
  --train_batch_size 1
