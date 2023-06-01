import logging

import clip
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from PIL import Image
from torch import nn
import math

accelerator = Accelerator()

from methods.transductive_zsl import MultimodalPrompt
from utils import (
    dataset_object,
    make_scheduler, 
    pseudolabel_top_k,
)


log = logging.getLogger(__name__)


class MultimodalFPL(MultimodalPrompt):
    def __init__(
        self, 
        config, 
        label_to_idx,
         data_folder,
        classes, 
        seen_classes, 
        unseen_classes, 
        device
    ):
        """This class defines self-trainig UPT's training and evaluation.
        :param config: dictionaries of prameters in models_config/upt_baseline_pseudo_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        """

        super().__init__(
            config, label_to_idx, classes, seen_classes, unseen_classes, device
        )

        self.data_folder = data_folder

    def create_training_dataset(self, train_data, unlabeled_data=None):
        """This function creates the dataset for training. Specifically, it
        merges pseudo-labels for unseen data and labeled data for seen classes.
        :param train_data: Dataset object - training seen classes (defined in zsl_jpl line 323)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for
                               unseen classes (defined in zsl_jpl line 328)
        """

        # Get pseudo-labels for unlabeled data from unseen classes
        train_unseen_dataset = pseudolabel_top_k(
            self.config,
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
            self.config.SPLIT_SEED
        )
        
        # Define the lists of traiing data from seen and unseen classes
        unseen_imgs = train_unseen_dataset.filepaths
        unseen_labs = train_unseen_dataset.labels

        # Use a portion of the pseudo-labeled data to build a validation set
        if self.config.N_PSEUDOSHOTS >= 10:
            np.random.seed(self.config.validation_seed)
            train_indices = np.random.choice(
                range(len(unseen_imgs)),
                size=int(len(unseen_imgs) * self.config.ratio_train_val),
                replace=False,
            )
            val_indices = list(
                set(range(len(unseen_imgs))).difference(set(train_indices))
            )

            self.val_unseen_files = np.array(unseen_imgs)[val_indices]
            self.val_unseen_labs = np.array(unseen_labs)[val_indices]

            unseen_imgs = list(np.array(unseen_imgs)[train_indices])
            unseen_labs = list(np.array(unseen_labs)[train_indices])

        else:
            self.val_unseen_files = None
            self.val_unseen_labs = None

        seen_imgs = train_data.filepaths
        seen_labs = [self.label_to_idx[l] for l in train_data.labels]

        self.balance_param = math.sqrt(len(seen_imgs) / len(unseen_imgs))

        train_data.filepaths = list(unseen_imgs) + list(seen_imgs)
        train_data.labels = list(unseen_labs) + list(seen_labs)
        train_data.label_id = True

    def define_loss_function(self, logits, labs, teacher=False):
        
        loss_ce_seen = self.cross_entropy(logits, labs, self.seen_classes)
        loss_ce_unseen = self.cross_entropy(logits, labs, self.unseen_classes)

        return loss_ce_seen + self.balance_param * loss_ce_unseen

    def cross_entropy(self, logits, labels, classes):
        """This loss computes the probability mass on the
        opposite set of classes for each sample.
        :param logits: continuous vector
        :param labels: class ids
        """

        ids = [self.label_to_idx[c] for c in classes]

        # Get indices of unseen and seen samples in the batch
        samples = []

        for idx, l in enumerate(labels):
            if l in ids:
                samples.append(idx)

        if samples:
            error = self.loss_func(logits[samples], labels[samples])
        else:
            error = 0

        return error

    def reindex_predicted_labels(self, idx_preds, only_unlabelled=False):
        """This function returns the correct index of predictions to compute
        model's accuracy.
        :param idx_pred: list of predictions ids
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """

        if only_unlabelled:
            return [self.unseen_classes[i.item()] for i in idx_preds]
        else:
            return [self.classes[i.item()] for i in idx_preds]

    def reindex_true_labels(self, label, only_unlabelled=False):
        """This function returns the correct index of true labels.
        :param label: list of labels from data loader
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """

        if only_unlabelled:
            return torch.tensor(
                [self.unseen_classes.index(self.classes[l.item()]) for l in label]
            )
        else:
            return torch.tensor([l for l in label])

    def get_pseudo_labels(self, unlabeled_examples):
        log.info(f"Num unlabeled data: {len(unlabeled_examples)}")
        # Get prediction on unlabeled data
        std_preds = self.test_predictions(
            unlabeled_examples, standard_zsl=True
        )

        DatasetObject = dataset_object(self.config.DATASET_NAME)
        # 4. Take top-16 pseudo-labels to finetune the student
        pseudo_unseen_examples = DatasetObject(
            std_preds["id"],
            self.data_folder,
            transform=self.transform,
            augmentations=None,
            train=True,
            labels=None,
            label_map=self.label_to_idx,
            class_folder=True,
            original_filepaths=unlabeled_examples.filepaths,
        )

        pseudo_labels = self.assign_pseudo_labels(
            self.config.N_PSEUDOSHOTS, pseudo_unseen_examples
        )

        return pseudo_labels

    def assign_pseudo_labels(self, k, unlabeled_data):

        # to find the top k for each class, each class has it's own "leaderboard"
        top_k_leaderboard = {
            self.label_to_idx[self.unseen_classes[i]]: []
            for i in range(len(self.unseen_classes))
        }  # maps class idx -> (confidence, image_path) tuple

        classes = self.unseen_classes
        for img_path in unlabeled_data.filepaths:
            # log.info(f"IMAGEPATH: {img_path}")
            img = Image.open(img_path).convert("RGB")
            img = torch.unsqueeze(self.transform(img), 0).to(self.device)
            with torch.no_grad():
                # Get text and image prompts using UPT
                # coop_embeddings, vpt_embeddings, vpt_deep_embeddings = self.model(0)
                # Calculate text prompts
                # text_features = self.text_encoder(coop_embeddings, classes)
                text_features, image_features = self.model(img, classes)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # Calculate image prompts
                # image_features = self.image_encoder(img, vpt_embeddings, deep_embds=vpt_deep_embeddings)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            probs = logits.softmax(dim=-1)
            idx_preds = torch.argmax(logits, dim=1)
            pred_id = idx_preds.item()
            pred = self.label_to_idx[self.unseen_classes[idx_preds.item()]]

            """if predicted class has empty leaderboard, or if the confidence is high
            enough for predicted class leaderboard, add the new example
            """
            prob_score = probs[0][pred_id]
            if len(top_k_leaderboard[pred]) < k:
                top_k_leaderboard[pred].append((prob_score, img_path))
            elif (
                top_k_leaderboard[pred][-1][0] < prob_score
            ):  # if the confidence in predicted class "qualifies" for top-k
                # default sorting of tuples is by first element
                top_k_leaderboard[pred] = sorted(
                    top_k_leaderboard[pred] + [(probs[0][pred_id], img_path)],
                    reverse=True,
                )[:k]
            else:
                # sort the other classes by confidence score
                order_of_classes = sorted(
                    [
                        (probs[0][j], j)
                        for j in range(len(self.unseen_classes))
                        if j != pred_id
                    ],
                    reverse=True,
                )
                for score, index in order_of_classes:
                    index_dict = self.label_to_idx[self.unseen_classes[index]]
                    # log.info(f"{classnames[index]}")
                    # log.info(f"{index_dict}")
                    if len(top_k_leaderboard[index_dict]) < k:
                        top_k_leaderboard[index_dict].append(
                            (probs[0][index], img_path)
                        )
                    elif top_k_leaderboard[index_dict][-1][0] < probs[0][index]:
                        # default sorting of tuples is by first element
                        top_k_leaderboard[index_dict] = sorted(
                            top_k_leaderboard[index_dict]
                            + [((probs[0][index], img_path))],
                            reverse=True,
                        )[:k]

        new_imgs = []
        new_labels = []
        # loop through, and rebuild the dataset
        for index, leaderboard in top_k_leaderboard.items():
            new_imgs += [tup[1] for tup in leaderboard]
            new_labels += [index for _ in leaderboard]

        unlabeled_data.filepaths = new_imgs
        unlabeled_data.labels = new_labels
        unlabeled_data.label_id = True

        return unlabeled_data
