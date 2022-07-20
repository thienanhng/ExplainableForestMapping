#!/bin/bash

# trained models:
# BB: bb_seed_0
# BB with single task: bb_flat_seed_0
# BB without DEM: bb_wo_alti_seed_0
# BB with weight binary loss: bb_weighted_bin_loss_seed_0

experiment=bb_weighted_bin_loss_seed_0
set=test

python infer.py \
        --input_sources SI2017 ALTI \
        --target_source TLM5c \
        --batch_size 1 \
        --padding 64 \
        --save_hard \
        --overwrite \
        --csv_fn data/csv/SI2017_ALTI_TLM5c_${set}.csv \
        --model_fn output/$experiment/training/${experiment}_model.pt \
        --output_dir output/$experiment/inference/epoch_19/${set} \
        --num_workers 2 \
        --evaluate \
        --random_seed 0 \
        > log_${experiment}_${set}_inference.txt
        # 
        # --save_soft \
        # --save_error_map \
        # --csv_fn data/csv/SI2017_ALTI_${set}.csv \
        
