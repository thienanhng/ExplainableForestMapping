#!/bin/bash

# trained models:
# sb: sb_seed_0
# sbcorr+: sb_corrp_seed_0
# sbrule-: sb_rulem_seed_0

experiment=sb_seed_0
set=test_with_context
python infer.py \
        --input_sources SI2017 ALTI \
        --interm_target_sources TH TCD1 \
        --batch_size 1 \
        --padding 64 \
        --overwrite \
        --save_hard \
        --csv_fn data/csv/SI2017_ALTI_${set}.csv \
        --model_fn output/$experiment/training/${experiment}_model.pt \
        --output_dir output/$experiment/inference/epoch_19/${set} \
        --num_workers 2 \
        --random_seed 0 \
        > log_${experiment}_${set}_inference.txt
        #--save_soft \
        #--save_error_map \
        #--evaluate \
        #--save_corr \
        #--csv_fn data/csv/SI2017_ALTI_TH_TCD1_TLM5c_${set}.csv \