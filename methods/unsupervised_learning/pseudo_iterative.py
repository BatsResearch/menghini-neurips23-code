import copy
import logging
import math
import pickle

import clip
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from accelerate import Accelerator
from PIL import Image
from torch import nn


accelerator = Accelerator()

from data import CustomDataset
from models import CustomImageEncoder, ImagePrefixModel
from methods import TeacherStudent
from utils import (
    dataset_object, 
    evaluate_predictions, 
    make_scheduler, 
    pseudolabel_top_k,
    seed_worker, 
    save_parameters,
    save_pseudo_labels,
)

g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class PseudoIterative(TeacherStudent):
    def __init__(
        self,
        config,
        label_to_idx,
        data_folder,
        classes,
        seen_classes,
        unseen_classes,
        device,
    ):
        super().__init__(
            config, label_to_idx, data_folder, classes, seen_classes, unseen_classes, device
        )


    def train(
        self,
        train_data,
        val_data,
        unlabeled_data,
        test_data,
        test_labeled_files,
        test_labeles,
    ):
        # Number of total iterations to cover all unlabeled data
        num_iter = int(100/self.config.STEP_QUANTILE)
        num_samples = int(len(unlabeled_data) / num_iter)
        # Initialize the number of pseudo-labels per class
        n_per_class = int(num_samples / len(self.unseen_classes))
        n_unseen = len(self.unseen_classes)
        if n_per_class * n_unseen <= len(unlabeled_data.filepaths):
            # self.num_pseudo_labels_per_class =  n_per_class
            self.config.N_PSEUDOSHOTS = n_per_class
        else:
            # self.num_pseudo_labels_per_class =  math.floor(len(unlabeled_data.filepaths)/n_unseen)
            self.config.N_PSEUDOSHOTS = math.floor(
                len(unlabeled_data.filepaths) / n_unseen
            )

        log.info(f"We select {self.config.N_PSEUDOSHOTS} pseudolabel per each unseen classes.")
        log.info(f"The number of unseen classes is: {len(self.unseen_classes)}.")
        log.info(f"Thus we expect an initial number of pseudo labeles equal to {len(self.unseen_classes) * self.config.N_PSEUDOSHOTS}.")
        # Create a safe copy of labeled/unlabeled data
        original_train_data = copy.deepcopy(train_data)
        # log.info(f"Training data labels: {original_train_data.labels}")
        original_unlabeled_data = copy.deepcopy(unlabeled_data)
        # Original val
        original_val_data = copy.deepcopy(val_data)

        # Initialize here first batch of pseudo labels
        #self.create_training_dataset(train_data, unlabeled_data)
        #log.info(f"The original train data has size: {len(original_train_data.filepaths)}.")
        #log.info(f"Plus: {len(unlabeled_data.filepaths)}.")

        for niter in range(1, num_iter + 1):
            log.info(f"NUM PSEUDO SHOTS: {self.config.N_PSEUDOSHOTS}")
            pseudolabel_top_k(
                self.config.DATASET_NAME,
                self.config.N_PSEUDOSHOTS,
                self.config.PROMPT_TEMPLATE,
                unlabeled_data,
                self.unseen_classes,
                self.transform,
                self.clip_model,
                self.label_to_idx,
                self.device,
                self.config.VIS_ENCODER,
                self.config.SPLIT_SEED,
            )
            
            log.info(f"Plus: {len(unlabeled_data.filepaths)}.")
            filename = f"pseudolabels/{self.config.DATASET_NAME}_CLIP_{self.config.VIS_ENCODER.replace('/', '')}_iter_{niter}_pseudolabels_spl_{self.config.SPLIT_SEED}.pickle"
            with open(filename, "wb") as f:
                pickle.dump({"filepaths": unlabeled_data.filepaths, "labels": unlabeled_data.labels}, f)

            # Exploit all the available unlabeled data
            if self.config.ALL_UNLABELED:
                n_per_class = int((niter + 1) * num_samples / n_unseen)
                log.info(f"n_per_class: {n_per_class}")
                if n_per_class * n_unseen <= len(original_unlabeled_data.filepaths):
                    log.info(f"if n_per_class: {n_per_class}")
                    self.config.N_PSEUDOSHOTS = n_per_class
                else:
                    log.info(f"else new val: {len(original_unlabeled_data.filepaths) / n_unseen}")
                    # We are making a stong assumption about the distribution of unlabeled data
                    self.config.N_PSEUDOSHOTS = math.floor(
                        len(original_unlabeled_data.filepaths) / n_unseen
                    )

            unlabeled_data = original_unlabeled_data
            original_unlabeled_data = copy.deepcopy(unlabeled_data)