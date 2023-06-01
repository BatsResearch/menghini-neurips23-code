import copy
from functools import reduce
import logging
import math
from operator import mul

import clip
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
from accelerate import Accelerator
from PIL import Image
from torch import nn
from torch.nn.modules.utils import _pair

accelerator = Accelerator()

from models import (
    CustomImageEncoder, 
    CustomTextEncoder, 
    ImagePrefixModel,
    TextPrefixModel,
    UPTModel,
)
from utils import (
    make_scheduler, 
    seed_worker, 
    save_parameters,
    save_pseudo_labels,
)


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)

class TrainingStrategy(object):
    def __init__(
        self, 
        config, 
        label_to_idx, 
        classes, 
        seen_classes, 
        unseen_classes, 
        device
    ):
        """ This class defines functions for the training strategies.

        :param config: dictionaries of prameters in models_config/vpt_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        """

        self.config = config
        self.classes = classes
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes
        self.label_to_idx = label_to_idx

        self.device = device
        self.clip_model, self.transform = clip.load(
            self.config.VIS_ENCODER, device=self.device
        )
        self.template = self.config.PROMPT_TEMPLATE

    def declare_custom_encoder(self):
        """ This function declares the custom encoder
        needed depending on the prompt modality.

        :param modality: either text or image
        """

        if self.config.MODALITY == 'image':
            self.visual_transformer = self.clip_model.visual
            self.image_encoder = CustomImageEncoder(self.visual_transformer
            ).to(self.device)
            log.info(f"Freeze visual encoder.")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        elif self.config.MODALITY == 'text':
            if torch.cuda.is_available():
                self.text_encoder = CustomTextEncoder(
                    self.clip_model, self.device, torch.float16
                ).to(self.device)  
            else:
                self.text_encoder = CustomTextEncoder(
                    self.clip_model, self.device, torch.half
                ).to(self.device)

            log.info(f"Freeze text encoder.")
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        elif self.config.MODALITY == 'multi':
            self.visual_transformer = self.clip_model.visual
            self.image_encoder = CustomImageEncoder(self.visual_transformer).to(self.device)
            if torch.cuda.is_available():
                self.text_encoder = CustomTextEncoder(
                    self.clip_model, self.device, torch.float16
                ).to(self.device)
            else:
                self.text_encoder = CustomTextEncoder(
                    self.clip_model, self.device, torch.half
                ).to(self.device)

            log.info(f"Freeze visual encoder.")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

            log.info(f"Freeze text encoder.")
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def initialize_prompts_parameters(self):
        """ This function initialized the prompt parameters
        depending on the prompt modality.

        :param modality: either text or image
        """

        if self.config.MODALITY == 'image':
            width = self.visual_transformer.class_embedding.size()[0]
            scale = width**-0.5
            if self.config.VIS_PREFIX_INIT == "normal":
                vis_initial_prefix = scale * torch.randn(self.config.PREFIX_SIZE, width)

            elif self.config.VIS_PREFIX_INIT == "uniform":
                val = math.sqrt(6.0 / float(3 * reduce(mul, (16, 16), 1) + width))  # noqa
                vis_initial_prefix = torch.zeros(self.config.PREFIX_SIZE, width)
                vis_initial_prefix = scale * nn.init.uniform_(vis_initial_prefix, -val, val)

            self.vis_initial_prefix = vis_initial_prefix

        elif self.config.MODALITY == 'text':
            # Prefix initialization
            prefix_dim = (
                1,
                self.config.PREFIX_SIZE,
                self.clip_model.token_embedding.embedding_dim,
            )
            self.initial_prefix = torch.normal(
                self.config.MEAN_INIT, self.config.VAR_INIT, size=prefix_dim
            ).to(self.device)

        elif self.config.MODALITY == 'multi':
            # Get relevant dimensions
            vpt_dim = self.clip_model.visual.conv1.weight.shape[0]
            coop_dim = self.clip_model.ln_final.weight.shape[0]

            # Initialize the coop prompt
            self.coop_embeddings = torch.empty(
                1, 
                self.config.TEXT_PREFIX_SIZE, 
                coop_dim,
                dtype=self.dtype).to(self.device)
            nn.init.normal_(self.coop_embeddings, std=0.02)

            # Initialize the vpt prompt
            clip_patchsize = self.clip_model.visual.conv1.weight.shape[-1]
            clip_patchsize = _pair(clip_patchsize)
            val = math.sqrt(6. / float(3 * reduce(mul, clip_patchsize, 1) + vpt_dim))  # noqa

            self.vpt_embeddings = torch.zeros(
                1, 
                self.config.VISION_PREFIX_SIZE, 
                vpt_dim, 
                dtype=self.dtype).to(self.device)
            # xavier_uniform initialization
            nn.init.uniform_(self.vpt_embeddings.data, -val, val)

            if self.config.VPT_DEEP:
                self.vision_layers = len([k for k in self.clip_model.state_dict().keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])

                self.vpt_embeddings_deep = torch.zeros(
                        self.vision_layers-1, 
                        self.config.VISION_PREFIX_SIZE, 
                        vpt_dim, 
                        dtype=self.dtype).to(self.device)
                # xavier_uniform initialization
                nn.init.uniform_(self.vpt_embeddings_deep.data, -val, val)
            else:
                self.vpt_embeddings_deep = None


    def define_model(self, classes=None):
        """ This function initialized the model
        depending on the prompt modality.

        :param modality: either text or image
        :param classes: the list of classes for textual model
        """

        if self.config.MODALITY == 'image':
            # Define model
            self.model = ImagePrefixModel(
                self.vis_initial_prefix,
                self.image_encoder,
                device=self.device,
            ).to(self.device)

        elif self.config.MODALITY == 'text':
            # Define model
            self.model = TextPrefixModel(
                self.initial_prefix,
                self.text_encoder,
                [" ".join(c.split("_")) for c in classes],
                device=self.device, 
            ).to(self.device)

        elif self.config.MODALITY == 'multi':

            # Define model
            self.model = UPTModel(
                self.coop_embeddings,
                self.vpt_embeddings,
                self.vpt_embeddings_deep,
                self.image_encoder,
                self.text_encoder,
                self.classes,
                self.config.TRANSFORMER_DIM, 
                device=self.device,
                dtype=self.dtype
            ).to(self.device)

        for i, parameter in enumerate(self.model.parameters()):
            if parameter.requires_grad:
                log.info(f"Shape of parameters {i}: {parameter.shape}")

        if self.config.OPTIM == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.LR,
                weight_decay=self.config.DECAY,
                momentum=0.9,
            )

        self.scheduler = make_scheduler(self.optimizer, self.config)
        self.loss_func = torch.nn.CrossEntropyLoss()
    
    def create_training_dataset(self, train_data, unlabeled_data=None):
        """This function create the dataset for training. Specifically, it
        merges pseudo-labels for unseen data and labeled data for seen classes.

        :param train_data: Dataset object - training seen classes (defined in zsl_jpl line 323)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for
                               unseen classes (defined in zsl_jpl line 328)
        """
        self.val_unseen_files = None
        return train_data
    
    def train(
        self,
        train_data,
        val_data,
        unlabeled_data=None,
        only_unlabelled=False,
        only_seen=False,
        iterative=False,
    ):
        """This function defines the training of self.model.

        :param train_data: Dataset object - training dataset of labeled data for
                           seen classes (defined in zsl_jpl line 323)
        :param val_data: Dataset object - validation dataset of labeled data for
                         seen classes (defined in zsl_jpl line 334)
        :param unlabeled_data: Dataset object - dataset of unlabeled data for
                               unseen classes (defined in zsl_jpl line 328)
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """

        # Define training dataset
        if not iterative:
            self.create_training_dataset(train_data, unlabeled_data)
            if self.config.MODALITY == 'text':
                self.define_model(self.classes)
            else:
                self.define_model()

        log.info(f"[self.train] Training data: {len(train_data.filepaths)}")
        
        # Declare the data pre processing for train and validation data
        train_data.transform = self.transform
        val_data.transform = self.transform

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        if self.val_unseen_files is not None:
        
            unseen_imgs = list(self.val_unseen_files)
            unseen_labs = list(self.val_unseen_labs)

            val_data.filepaths = list(unseen_imgs)
            val_data.labels = list(unseen_labs)
            val_data.label_id = True

        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=self.config.BATCH_SIZE
        )

        accelerator.wait_for_everyone()

        self.model, self.optimizer, train_loader, val_loader = accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader
        )

        best_val_accuracy = 0
        best_prompt = None
        loss = None
        if val_loader is not None:
            log.info(f"Size of validation dataset: {len(val_data.filepaths)}")

        for epoch in range(self.config.EPOCHS):
            log.info(f"Run Epoch {epoch}")
            total_loss = 0
            accum_iter = self.config.ACCUMULATION_ITER

            loss, total_loss, epoch_parameters = self._train_epoch(
                loss,
                total_loss,
                train_loader,
                accum_iter,
                epoch,
                only_unlabelled=only_unlabelled,
                only_seen=only_seen,
            )
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                log.info(f"Loss Epoch {epoch}: {total_loss/(len(train_loader))}")

            accelerator.free_memory()

            if val_loader is not None:
                val_accuracy = self._run_validation(val_loader, only_unlabelled)
                log.info(f"Validation accuracy after Epoch {epoch}: {val_accuracy}")
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_prompt = epoch_parameters
            else:
                best_val_accuracy = None
                best_prompt = epoch_parameters

            if self.config.MODALITY == 'text':
                # After validation on seen classes redefine the set of training classes
                self.model.classes = self.classes

        return best_val_accuracy, epoch_parameters

    def fixed_iterative_train(self,
        train_data,
        val_data,
        unlabeled_data,
        only_seen=False,
    ):
        # Number of total iterations to cover all unlabeled data
        num_iter = int(100/self.config.STEP_QUANTILE)
        num_samples = int(len(unlabeled_data) / num_iter)
        # Initialize the number of pseudo-labels per class
        n_per_class = int(num_samples / len(self.classes))
        n_unseen = len(self.classes)

        log.info(f"We select {self.config.N_PSEUDOSHOTS} pseudolabel per each unseen classes.")
        log.info(f"The number of unseen classes is: {len(self.classes)}.")
        log.info(f"Thus we expect an initial number of pseudo labeles equal to {len(self.classes) * self.config.N_PSEUDOSHOTS}.")

        # Create a safe copy of labeled/unlabeled data
        original_train_data = copy.deepcopy(train_data)
        # log.info(f"Training data labels: {original_train_data.labels}")
        original_unlabeled_data = copy.deepcopy(unlabeled_data)
        # Original val
        original_val_data = copy.deepcopy(val_data)

        # 1. Get training data (also get pseudo-labeles from CLIP)
        self.create_training_dataset(train_data, unlabeled_data)
        log.info(f"The original train data has size: {len(original_train_data.filepaths)}.")
        log.info(f"Only with unlabeled is: {len(unlabeled_data.filepaths)}.")
        log.info(f"Current train data is: {len(train_data.filepaths)}.")

        # Save pseudolabels
        log.info(f"Saving pseudo-labels for init")
        save_pseudo_labels(
            unlabeled_data.filepaths, 
            unlabeled_data.labels, 
            self.config, 
            iteration=0,
        )
        log.info(f"Unlabeled is: {len(unlabeled_data.filepaths)}.")

        for niter in range(1, num_iter + 1):
            log.info(f"Start {niter} round of training..")

            train_data.filepaths = [
                f for i, f in enumerate(original_train_data.filepaths)
            ]
            train_data.labels = [l for i, l in enumerate(original_train_data.labels)]
           
            log.info(f"Unlabeled is {len(unlabeled_data.filepaths)} at iter: {niter}.")
            self.update_training_set(train_data, unlabeled_data)
            log.info(f"Train data is {len(train_data.filepaths)} at iter: {niter}.")

            # 2. Define model
            if self.config.MODALITY == 'text':
                self.define_model(self.classes)
            else:
                self.define_model()

            log.info(f"[MODEL] Initialization iter {niter}")

            # 3. Train model
            log.info(f"[MODEL] Start model training iter {niter}..")
            t_best_val_accuracy, t_best_prompt = self.train(
                train_data, val_data, only_seen=only_seen, iterative=True,
            )
            log.info(f"[MODEL] Training completed iter {niter}.")

            log.info(f"[MODEL] Collecting model pseudo-labels on unlabeled data..")
            unlabeled_data = self.get_pseudo_labels(
                original_unlabeled_data
            )

             # Save pseudolabels
            log.info(f"Saving pseudo-labels for iteration {niter}")
            save_pseudo_labels(
                unlabeled_data.filepaths, 
                unlabeled_data.labels, 
                self.config, 
                iteration=niter,
            )

            save_parameters(
                t_best_prompt,
                self.config, 
                iteration=niter
            )

            val_data = original_val_data
            original_val_data = copy.deepcopy(val_data)
        
        return t_best_val_accuracy, t_best_prompt

    def grip_train(
        self,
        train_data,
        val_data,
        unlabeled_data,
        only_seen=False,
    ):

        # Number of total iterations to cover all unlabeled data
        num_iter = int(100/self.config.STEP_QUANTILE)
        num_samples = int(len(unlabeled_data) / num_iter)
        # Initialize the number of pseudo-labels per class
        n_per_class = int(num_samples / len(self.classes))
        n_unseen = len(self.classes)
        if n_per_class * n_unseen <= len(unlabeled_data.filepaths):
            # self.num_pseudo_labels_per_class =  n_per_class
            self.config.N_PSEUDOSHOTS = n_per_class
        else:
            # self.num_pseudo_labels_per_class =  math.floor(len(unlabeled_data.filepaths)/n_unseen)
            self.config.N_PSEUDOSHOTS = math.floor(
                len(unlabeled_data.filepaths) / n_unseen
            )

        log.info(f"We select {self.config.N_PSEUDOSHOTS} pseudolabel per each unseen classes.")
        log.info(f"The number of unseen classes is: {len(self.classes)}.")
        log.info(f"Thus we expect an initial number of pseudo labeles equal to {len(self.classes) * self.config.N_PSEUDOSHOTS}.")
        # Create a safe copy of labeled/unlabeled data
        original_train_data = copy.deepcopy(train_data)
        # log.info(f"Training data labels: {original_train_data.labels}")
        original_unlabeled_data = copy.deepcopy(unlabeled_data)
        # Original val
        original_val_data = copy.deepcopy(val_data)

        # Initialize here first batch of pseudo labels
        self.create_training_dataset(train_data, unlabeled_data)
        log.info(f"The original train data has size: {len(original_train_data.filepaths)}.")
        log.info(f"Plus: {len(unlabeled_data.filepaths)}.")

        # Save pseudolabels
        log.info(f"Saving pseudo-labels for iteration {0}")
        save_pseudo_labels(
            unlabeled_data.filepaths, 
            unlabeled_data.labels, 
            self.config, 
            0,
        )
        log.info(f"Unlabeled is: {len(unlabeled_data.filepaths)}.")

        for niter in range(1, num_iter + 1):
            log.info(f"Start {niter} round of training..")

            train_data.filepaths = [
                f for i, f in enumerate(original_train_data.filepaths)
            ]
            train_data.labels = [l for i, l in enumerate(original_train_data.labels)]

            self.update_training_set(train_data, unlabeled_data)

            # 1. Initialize model            
            if self.config.MODALITY == 'text':
                self.define_model(self.classes)
            else:
                self.define_model()
            log.info(f"[TEACHER] Initialization..")

            # # Validation with seen and unseen.
            # if self.val_unseen_files is not None:
            #     seen_imgs = original_val_data.filepaths
            #     seen_labs = [self.label_to_idx[l] for l in original_val_data.labels]
                
            #     unseen_imgs = list(self.val_unseen_files)
            #     unseen_labs = list(self.val_unseen_labs)

            #     val_data.filepaths = list(unseen_imgs) + list(seen_imgs)
            #     val_data.labels = list(unseen_labs) + list(seen_labs)
            #     val_data.label_id = True

            # 2. Train teacher with labeled seen and pseudo-labeled unseen
            log.info(f"[TEACHER] Start model training..")
            t_best_val_accuracy, t_best_prompt = self.train(
                train_data, val_data, only_seen=only_seen, iterative=True,
            )
            log.info(f"[TEACHER] Training completed.")

            # Increase the number of pseudolabels
            n_per_class = int((niter + 1) * num_samples / n_unseen)
            if n_per_class * n_unseen <= len(original_unlabeled_data.filepaths):
                self.config.N_PSEUDOSHOTS = n_per_class
            else:
                # We are making a stong assumption about the distribution of unlabeled data
                self.config.N_PSEUDOSHOTS = math.floor(
                    len(original_unlabeled_data.filepaths) / n_unseen
                )

            # 3. Get teacher pseudo-labels
            log.info(f"[TEACHER] Collecting teacher pseudo-labels on unlabeled data..")
            unlabeled_data = self.get_pseudo_labels(
                original_unlabeled_data
            )

            save_pseudo_labels(
                unlabeled_data.filepaths, 
                unlabeled_data.labels, 
                self.config, 
                niter,
            )

            save_parameters(t_best_prompt, self.config, iteration=niter)

            val_data = original_val_data
            original_val_data = copy.deepcopy(val_data)

        return t_best_val_accuracy, t_best_prompt

    def define_loss_function(self, logits, labs):
        return self.loss_func(logits, labs)

    def backpropagate(self):
        self.optimizer.step()
        self.model.zero_grad()

    def update_scheduler(self):
        current_lr = self.scheduler.get_last_lr()
        self.scheduler.step()

    def unwrap_model(self):
        return accelerator.unwrap_model(self.model)

    def training_model(self, img):
        """This function allows to customize the model to use while trainig

        :param img: Tensor of images form Dataloader
        """
        return self.model(img)

    def update_training_set(self, train_data, unlabeled_data):
        
        # Get pseudo-labels for unlabeled data from unseen classes
        train_unseen_dataset = unlabeled_data
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

        train_data.filepaths = list(unseen_imgs)
        train_data.labels = list(unseen_labs)
        train_data.label_id = True
        log.info(f"UPDATE DATASET: size = {len(train_data.filepaths)}")
        log.info(f"UPDATE UNSEEN DATASET: size = {len(unseen_imgs)}")
