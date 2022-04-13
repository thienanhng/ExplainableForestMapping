#!/bin/bash

# trained models:
# bb: bb_hierarchical

python infer.py \
        --input_sources SI2017 ALTI \
        --target_source TLM5c \
        --batch_size 16 \
        --save_hard \
        --save_soft \
        --save_error_map \
        --evaluate \
        --overwrite \
        --csv_fn data/csv/SI2017_ALTI_TLM5c_test.csv \
        --model_fn output/bb_hierarchical/training/bb_hierarchical_model.pt \
        --output_dir output/bb_hierarchical/inference/epoch_19/test \
        --num_workers 2 \
        #> log_bb_inference.txt
