#!/bin/bash

for dataset_dir in '/path/data' ; do # add here the path to the folder containing dataset folders
for vis_encoder in 'ViT-B/32'; do # You can choose among 'ViT-B/32' and 'ViT-L/14'
for split_seed in 500; do # This indicate the split for TRZSL, i.e., 500, 0, or 200. For other learning setting this is 500 as default.
for dataset_name in RESICS45; do # DTD Flowers102 EuroSAT FGVCAircraft MNIST
for model in textual_fpl; do # Choose among: textual_fpl, visual_fpl, multimodal_fpl, iterative_textual_fpl, iterative_visual_fpl, iterative_multimodal_fpl, grip_textual, grip_visual, grip_multimodal
for optim_seed in 1; do # 1 2 3 4 5 are the seeds we used
    
    export OPTIM_SEED="$optim_seed"
    export VIS_ENCODER="$vis_encoder"
    export DATASET_NAME="$dataset_name"
    export SPLIT_SEED="$split_seed"
    export MODEL="$model"
    export DATASET_DIR="$dataset_dir"
    
    # Set accelerate configguration file to to accelerate_config.yml when running on GPUs
    accelerate launch --config_file methods_config/accelerate_localtest_config.yml run_main_ul.py --model_config ${model}_config.yml \
                      --learning_paradigm ul # Choose among ul, ssl, and trzsl

done
done
done
done
done
done