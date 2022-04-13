#!/bin/bash

# trained models:
# sb: sb_hierarchical_MSElog1em1_MSE_doubling_negatives
# sbcorr+: sb_hierarchical_MSElog1em1_MSE_doubling_negatives_lpen_2em1
# sbrule-: sb_hierarchical_MSElog1em1_MSE_doubling_negatives_eps_2em1

experiment=sb_hierarchical_MSElog1em1_MSE_doubling_negatives
python infer.py \
        --input_sources SI2017 ALTI \
        --target_source TLM5c \
        --interm_target_sources TH TCD1 \
        --batch_size 16 \
        --save_hard \
        --save_soft \
        --save_error_map \
        --evaluate \
        --overwrite \
        --csv_fn data/csv/SI2017_ALTI_TH_TCD1_TLM5c_test.csv \
        --model_fn output/$experiment/training/${experiment}_model.pt \
        --output_dir output/$experiment/inference/epoch_19/test \
        --num_workers 2 \
        #> log_${experiment}_inference.txt