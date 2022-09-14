#!/bin/bash

experiment=sb_rulem

python train.py \
        --input_sources SI2017 ALTI \
        --interm_target_sources TH TCD1 \
        --train_csv_fn data/csv/SI2017_ALTI_TH_TCD1_TLM5c_train_with_counts.csv \
        --val_csv_fn data/csv/SI2017_ALTI_TH_TCD1_TLM5c_val.csv \
        --batch_size 16 \
        --num_epochs 15 \
        --lr 1e-6 1e-6 1e-7 \
        --learning_schedule 5 5 5 \
        --n_negative_samples 10 20 40 80 160 320 320 320 \
        --negative_sampling_schedule 1 2 2 2 2 2 2 2 \
        --decision h \
        --penalize_correction \
        --regression_loss MSElog MSE \
        --lambda_bin 1 \
        --lambda_sem 0.75 0.5 0.25 \
        --lambda_corr 1.0 \
        --epsilon_rule 0.2 \
        --patch_size 128 \
        --padding 64 \
        --inference_batch_size 1 \
        --num_workers 2 \
        --output_dir output/$experiment \
        --resume_training \
        --starting_point sb_epoch_4 \
        --no_user_input \
        --random_seed 0 \
        > log_${experiment}_training.txt
