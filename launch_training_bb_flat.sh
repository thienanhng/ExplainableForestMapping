#!/bin/bash

experiment=bb_flat_seed_0

python train.py \
        --input_sources SI2017 ALTI \
        --target_source TLM5c \
        --train_csv_fn data/csv/SI2017_ALTI_TLM5c_train_with_counts.csv \
        --val_csv_fn data/csv/SI2017_ALTI_TLM5c_val.csv \
        --batch_size 16 \
        --num_epochs 18 \
        --lr 1e-5 1e-6 1e-6 1e-7 \
        --learning_schedule 3 5 5 5 \
        --n_negative_samples 5 10 20 40 80 160 320 320 320 \
        --negative_sampling_schedule 2 2 2 2 2 2 2 2 2 \
        --decision f \
        --lambda_bin 1 \
        --patch_size 128 \
        --padding 64 \
        --inference_batch_size 1 \
        --num_workers 2 \
        --output_dir output/$experiment \
        --no_user_input \
        --random_seed 0 \
        --resume_training \
        --starting_model_fn output/bb_flat_seed_0/training/bb_flat_seed_0_model.pt \
        --starting_metrics_fn output/bb_flat_seed_0/training/bb_flat_seed_0_metrics.pt \
        > log_bb_flat_training.txt
