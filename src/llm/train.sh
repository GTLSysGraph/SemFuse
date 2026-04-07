#!/bin/bash

MODEL_PATH="/your_model_path/"
OUTPUT_DIR="saves/llama3/"

llamafactory-cli train \
  --stage sft \
  --model_name_or_path "$MODEL_PATH" \
  --do_train True \
  --preprocessing_num_workers 16 \
  --dataset deepseek_distill_het \
  --template llama3 \
  --finetuning_type lora \
  --output_dir "$OUTPUT_DIR" \
  --cutoff_len 2048 \
  --lora_rank 32 \
  --lora_dropout 0.1 \
  --lora_alpha 64 \
  --lora_target all \
  --flash_attn fa2 \
  --overwrite_cache \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --save_steps 100 \
  --learning_rate 1e-4 \
  --warmup_steps 100 \
  --train_on_prompt False \
  --max_grad_norm 1.0 \
  --num_train_epochs 5.0 \
  --plot_loss True \
  --bf16 True \
  --ddp_find_unused_parameters False \
  --report_to none