#!/bin/bash

python train.py \
        --input_sources SI2017 ALTI \
        --target_source TLM5c \
        --train_csv_fn data/csv/SI2017_ALTI_TLM5c_train_with_counts.csv \
        --val_csv_fn data/csv/SI2017_ALTI_TLM5c_val.csv \
        --batch_size 16 \
        --num_epochs 20 \
        --lr 1e-5 1e-6 1e-6 1e-7 \
        --learning_schedule 5 5 5 5 \
        --n_negative_samples 0 5 10 20 40 80 160 320 320 320 \
        --negative_sampling_schedule 2 2 2 2 2 2 2 2 2 2 \
        --decision h \
        --lambda_bin 1 \
        --num_workers 2 \
        --output_dir output/bb_hierarchical_new \
        #> log_bb_training.txt
