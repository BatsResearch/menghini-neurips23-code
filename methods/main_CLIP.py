import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from logging import StreamHandler
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import scipy.stats as st
import torch
import yaml
from accelerate import Accelerator
from requests.adapters import HTTPAdapter
from torch import nn
from urllib3.util import Retry

from data import CustomDataset, dataset_custom_prompts
from methods import ClipBaseline
from utils import (
    Config,
    dataset_object,
    evaluate_predictions,
    get_class_names,
    get_labeled_and_unlabeled_data,
    save_parameters,
    save_predictions,
    store_results,
)

accelerator = Accelerator()

logger_ = logging.getLogger()
logger_.level = logging.INFO
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")


class AccelerateHandler(StreamHandler):
    def __init__(self, stream):
        super().__init__(stream)

    def emit(self, record):
        if accelerator.is_local_main_process:
            super().emit(record)


stream_handler = AccelerateHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger_.addHandler(stream_handler)

log = logging.getLogger(__name__)

def workflow(dataset_dir, obj_conf):
    
    dataset = obj_conf.DATASET_NAME
    # Get class names of target task
    classes, seen_classes, unseen_classes = get_class_names(
        dataset, 
        dataset_dir, 
        obj_conf.SPLIT_SEED
    )
    # Create dict classes to pass as variable
    dict_classes = {
        "classes": classes,
        "seen_classes": seen_classes,
        "unseen_classes": unseen_classes,
    }
    # Log number of classes
    log.info(f"\n----------------------DATA INFO-----------------------\n")
    log.info(f"Number of classes split {obj_conf.SPLIT_SEED}: {len(classes)}")
    log.info(f"Number of seen classes split {obj_conf.SPLIT_SEED}: {len(seen_classes)}")
    log.info(f"Number of unseen classes split {obj_conf.SPLIT_SEED}: {len(unseen_classes)}")
    # Path for images
    data_folder = f"{dataset_dir}/{dataset}"
    log.info(f"Data folder: {data_folder}")
    log.info(f"\n-------------------------------------------------------------\n")
    
    # Get labeled data (seen classes)
    # Get unlabeled data (unseen classes)
    # Get test data (both seen and unseen classes)
    _, _, test_data = get_labeled_and_unlabeled_data(
        dataset, data_folder, seen_classes, unseen_classes, classes
    )

    # Create datasets
    test_labeled_files, test_labeles = zip(*test_data)
    label_to_idx = {c: idx for idx, c in enumerate(classes)}

    DatasetObject = dataset_object(obj_conf.DATASET_NAME)

    # Test set (test seen and unseen)
    test_dataset = DatasetObject(
        test_labeled_files,
        data_folder,
        transform=None,
        augmentations=None,
        train=False,
        labels=None,
        label_map=label_to_idx,
    )
    # Log info data
    log.info(f"\n----------------------TRAINING DATA INFO-----------------------\n")
    log.info(f"Len test data: {len(test_dataset.filepaths)}")
    log.info(f"\n-------------------------------------------------------------\n")
    # Define model
    device = "cuda" if torch.cuda.is_available() else "cpu"


    log.info(f"\n----------------------MODEL INFO-----------------------\n")
    log.info(f"The model in use is: {obj_conf.MODEL}")
    model = ClipBaseline(obj_conf, label_to_idx, device=device, **dict_classes)

    # Validate on test set (standard)
    std_predictions, images, predictions, prob_preds = model.test_predictions(test_dataset)
    # Submit predictions (standard)
    std_response = evaluate_predictions(
        obj_conf,
        std_predictions,
        test_labeled_files,
        test_labeles,
        unseen_classes,
        seen_classes
    )
    log.info(f"ZSL accuracy: {std_response}")

    # Store model results
    store_results(obj_conf, std_response)

    dictionary_predictions = {
        'images' : images, 
        'predictions' : predictions,
        'labels' : test_labeles,
        'logits' : prob_preds,
    }

    save_predictions(dictionary_predictions, obj_conf, iteration=None)

 
def main():
    parser = argparse.ArgumentParser(description="Run JPL task")
    parser.add_argument(
        "--model_config",
        type=str,
        default="model_config.yml",
        help="Name of model config file",
    )
    parser.add_argument(
        "--learning_paradigm",
        type=str,
        default="trzsl",
        help="Choose among trzsl, ssl, and ul",
    )

    args = parser.parse_args()

    with open(f"methods_config/{args.model_config}", "r") as file:
        config = yaml.safe_load(file)

    # Cast configs to object
    obj_conf = Config(config)

    # Set seed seed
    optim_seed = int(os.environ["OPTIM_SEED"])
    obj_conf.OPTIM_SEED = optim_seed
    # Set backbone
    obj_conf.VIS_ENCODER = os.environ["VIS_ENCODER"]
    # Set dataset name
    obj_conf.DATASET_NAME = os.environ["DATASET_NAME"]
    # Set dataset dir
    obj_conf.DATASET_DIR = os.environ["DATASET_DIR"]
    # Set model name
    obj_conf.MODEL = os.environ["MODEL"]
    # Set split seed
    obj_conf.SPLIT_SEED = int(os.environ["SPLIT_SEED"])
    # Set dataset's template for textual prompts
    obj_conf.PROMPT_TEMPLATE = dataset_custom_prompts[obj_conf.DATASET_NAME]
    # Set data dir
    dataset_dir = obj_conf.DATASET_DIR
    # Set learning paradigm
    obj_conf.LEARNING_PARADIGM = args.learning_paradigm

    # Set the file path for the log file
    log_file = f"logs/{obj_conf.DATASET_NAME}_{obj_conf.MODEL}_{obj_conf.VIS_ENCODER.replace('/', '-')}.log"
    # Create a FileHandler and set the log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    # Add the FileHandler to the logger
    logger_.addHandler(file_handler)
    
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Dataset dir: {dataset_dir}")
    # Check dataset directory exists
    if not Path(dataset_dir).exists():
        print(dataset_dir)
        raise Exception("`dataset_dir` does not exist..")

    # Set random seeds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(obj_conf.OPTIM_SEED)
    random.seed(obj_conf.OPTIM_SEED)
    torch.manual_seed(obj_conf.OPTIM_SEED)
    accelerator.wait_for_everyone()
    # Seed for cuda
    if torch.cuda.is_available():
        torch.cuda.manual_seed(obj_conf.OPTIM_SEED)
        torch.cuda.manual_seed_all(obj_conf.OPTIM_SEED)
        accelerator.wait_for_everyone()

    torch.backends.cudnn.benchmark = True

    workflow(dataset_dir, obj_conf)


if __name__ == "__main__":
    main()
