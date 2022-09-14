#!/bin/bash

# trained models:
# BB: bb
# BB with single task: bb_flat
# BB without DEM: bb_dem_ablation

experiment=bb
set=test

python infer.py \
        --input_sources SI2017 ALTI \
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
        
