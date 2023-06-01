#!/bin/bash

for dataset_dir in 'pata/data' ; do # add here the path to the folder containing dataset folders
for vis_encoder in 'ViT-B/32'; do # You can choose among 'ViT-B/32' and 'ViT-L/14'
for split_seed in 500; do # This indicate the split for TRZSL, i.e., 500, 0, or 200. For other learning setting this is 500 as default.
for dataset_name in RESICS45; do # DTD Flowers102 EuroSAT FGVCAircraft MNIST
for model in clip; do 
for optim_seed in 1; do # To simply make inference with CLIP we do not need multiple seeds
    
    export OPTIM_SEED="$optim_seed"
    export VIS_ENCODER="$vis_encoder"
    export DATASET_NAME="$dataset_name"
    export SPLIT_SEED="$split_seed"
    export MODEL="$model"
    export DATASET_DIR="$dataset_dir"

    python3 ./run_main_clip.py \
            --model_config ${model}_config.yml \
            --learning_paradigm trzsl # Choose among ul, ssl, and trzsl

done
done
done
done
done
done