import logging

import clip
import numpy as np
import copy
import pandas as pd
import scipy.stats as st
import torch
from accelerate import Accelerator


accelerator = Accelerator()

from methods.transductive_zsl import TrainingStrategy
from utils import make_scheduler, seed_worker


g = torch.Generator()
g.manual_seed(0)

log = logging.getLogger(__name__)


class MultimodalPrompt(TrainingStrategy):
    def __init__(
        self, 
        config, 
        label_to_idx, 
        classes, 
        seen_classes, 
        unseen_classes, 
        device
    ):
        """This class defines UPT's training and evaluation.
        :param config: dictionaries of prameters in models_config/vpt_baseline_config.yml
        :param label_to_idx: dictionary (key, value):(class name, id)
        :param classes: list of class names
        :param seen_classes: list of seen classes' names
        :param unseen_classes: list of unseen classes' names
        :param device: device in use
        """
        super().__init__(
            config, label_to_idx, classes, seen_classes, unseen_classes, device
        )
        
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.declare_custom_encoder()
        
        # creates initial prefixes, linear layers, and lightweight transformer
        self.initialize_prompts_parameters()

    def reindex_predicted_labels(self, idx_preds, only_unlabelled=False):
        """This function returns the correct index of predictions to compute
        model's accuracy.
        :param idx_pred: list of predictions ids
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """
        return [self.seen_classes[i.item()] for i in idx_preds]

    def reindex_true_labels(self, label, only_unlabelled=False):
        """This function returns the correct index of true labels.
        :param label: list of labels from data loader
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        """

        return torch.tensor(
            [self.seen_classes.index(self.classes[l.item()]) for l in label]
        )

    def _train_epoch(
        self,
        loss,
        total_loss,
        train_loader,
        accum_iter,
        epoch,
        only_unlabelled=False,
        only_seen=False,
    ):
        """This function defines the training epoch of self.model.
        :param loss: float loss (average across batches)
        :param total_loss: float total loss
        :param train_loader: Dataloader object - training data defined in self.train
        :param accum_iter: number of accumulation steps minimum 1
        :param epoch: current epoch
        :param only_unlabelled: boolean. It is True if the training only involves
                                pseudo-labeled unseen data
        :param teachet: boolean. Added to use this function in more subclasses
        """
        if only_unlabelled:
            classes = self.unseen_classes
        elif only_seen:
            classes = self.seen_classes
        else:
            classes = self.classes
        log.info(f"[self._train_epoch] Number of prompts: {len(classes)}")

        predictions = []
        labels = []
        for i, (img, _, _, label, img_path) in enumerate(train_loader):
            # Get text and image prompts using UPT
            # coop_embeddings, vpt_embeddings, vpt_deep_embeddings = self.model(0)
            # Calculate text prompts
            # text_features = self.text_encoder(coop_embeddings, classes)
            text_features, image_features = self.model(img, classes)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Calculate image prompts
            # image_features = self.image_encoder(img, vpt_embeddings, deep_embds=vpt_deep_embeddings)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            idx_preds = torch.argmax(logits, dim=1)

            real_preds = self.reindex_predicted_labels(idx_preds, only_unlabelled)

            predictions += real_preds
            labels += [self.classes[i.item()] for i in label]

            labs = self.reindex_true_labels(label, only_unlabelled)
            labs = labs.to(self.device)
            loss = self.define_loss_function(logits, labs)
            total_loss += loss.item()

            accelerator.wait_for_everyone()

            loss = loss / accum_iter
            accelerator.backward(loss)

            # Accumulate grandient
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                self.backpropagate()

        accelerator.wait_for_everyone()

        predictions = torch.tensor([self.label_to_idx[p] for p in predictions]).to(
            self.device
        )
        labels = torch.tensor([self.label_to_idx[l] for l in labels]).to(self.device)

        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)

        # Get harmonic mean
        idx_seen = [self.label_to_idx[c] for c in self.seen_classes]
        seen_true = [i for i, c in enumerate(labels_outputs) if c in idx_seen]
        seen_preds = predictions_outputs[seen_true]
        seen_labs = labels_outputs[seen_true]
        seen_accuracy = torch.sum(seen_preds == seen_labs) / len(seen_true)

        idx_unseen = [self.label_to_idx[c] for c in self.unseen_classes]
        unseen_true = [i for i, c in enumerate(labels_outputs) if c in idx_unseen]
        unseen_preds = predictions_outputs[unseen_true]
        unseen_labs = labels_outputs[unseen_true]
        unseen_accuracy = torch.sum(unseen_preds == unseen_labs) / len(unseen_true)

        if only_unlabelled:
            accuracy = unseen_accuracy
            log.info(f"Training UNSEEN accuracy after Epoch {epoch}: {unseen_accuracy}")
        else:
            if only_seen:
                accuracy = seen_accuracy
                log.info(f"Training SEEN accuracy after Epoch {epoch}: {accuracy}")
            else:
                accuracy = st.hmean([unseen_accuracy.cpu(), seen_accuracy.cpu()])

                # accuracy = torch.sum(predictions_outputs == labels_outputs)/len(predictions_outputs)
                log.info(f"Training SEEN accuracy after Epoch {epoch}: {seen_accuracy}")
                log.info(
                    f"Training UNSEEN accuracy after Epoch {epoch}: {unseen_accuracy}"
                )
                log.info(f"Training HARMONIC accuracy after Epoch {epoch}: {accuracy}")

        self.update_scheduler()

        unwrapped_model = self.unwrap_model()
        epoch_parameters = [
            unwrapped_model.transformer.state_dict(),
            unwrapped_model.proj_coop_pre.state_dict(),
            unwrapped_model.proj_coop_post.state_dict(),
            unwrapped_model.proj_vpt_pre.state_dict(),
            unwrapped_model.proj_vpt_post.state_dict(),
            unwrapped_model.coop_embeddings.detach().cpu().numpy(),
            None if unwrapped_model.vpt_embeddings_deep is None else unwrapped_model.vpt_embeddings_deep.detach().cpu().numpy(),
            unwrapped_model.vpt_embeddings.detach().cpu().numpy(),
        ]

        return loss, total_loss, epoch_parameters

    def _run_validation(
        self, val_loader, only_unlabelled=False, only_seen=False
    ):
        """This function computes the validation accuracy on labeled seen data.
        :param val_loder: Dataloader object - validation dataset
        """

        # Define text queries
        if self.val_unseen_files is not None:
            val = False
        else:
            val = True

        predictions = []
        labels = []
        classes = self.seen_classes if self.val_unseen_files is None else self.classes
        log.info(f"[self._run_validation] Number of prompts: {len(classes)}")

        for img, _, _, label, img_path in val_loader:
            # Get text and image prompts using UPT
            # coop_embeddings, vpt_embeddings, vpt_deep_embeddings = self.model(0)
            # Calculate text prompts
            # text_features = self.text_encoder(coop_embeddings, classes) # TODO: Should this be all classes?
            text_features, image_features = self.model(img, classes)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Calculate image prompts
            # image_features = self.image_encoder(img, vpt_embeddings, deep_embds=vpt_deep_embeddings)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            idx_preds = torch.argmax(logits, dim=1)
            if self.val_unseen_files is not None:
                real_preds = [self.classes[i.item()] for i in idx_preds]
            else:
                real_preds = [self.seen_classes[i.item()] for i in idx_preds]

            predictions += real_preds
            labels += [self.classes[i.item()] for i in label]

        accelerator.wait_for_everyone()

        predictions = torch.tensor([self.label_to_idx[p] for p in predictions]).to(
            self.device
        )
        labels = torch.tensor([self.label_to_idx[l] for l in labels]).to(self.device)

        predictions_outputs = accelerator.gather(predictions)
        labels_outputs = accelerator.gather(labels)

        if len(text_features) < len(self.classes):
            accuracy = torch.sum(predictions_outputs == labels_outputs) / len(
                predictions_outputs
            )
        else:
            # Get harmonic mean
            idx_seen = [self.label_to_idx[c] for c in self.seen_classes]
            seen_true = [i for i, c in enumerate(labels_outputs) if c in idx_seen]
            seen_preds = predictions_outputs[seen_true]
            seen_labs = labels_outputs[seen_true]
            seen_accuracy = torch.sum(seen_preds == seen_labs) / len(seen_true)

            idx_unseen = [self.label_to_idx[c] for c in self.unseen_classes]
            unseen_true = [i for i, c in enumerate(labels_outputs) if c in idx_unseen]
            unseen_preds = predictions_outputs[unseen_true]
            unseen_labs = labels_outputs[unseen_true]
            unseen_accuracy = torch.sum(unseen_preds == unseen_labs) / len(unseen_true)

            if only_seen:
                accuracy = seen_accuracy
                log.info(f"Validation SEEN accuracy after Epoch: {seen_accuracy}")

            else:
                accuracy = st.hmean([unseen_accuracy.cpu(), seen_accuracy.cpu()])
                log.info(f"Validation SEEN accuracy after Epoch: {seen_accuracy}")
                log.info(f"Validation UNSEEN accuracy after Epoch: {unseen_accuracy}")
                log.info(f"Validation HARMONIC accuracy after Epoch: {accuracy}")

        return accuracy

    def test_predictions(self, data, standard_zsl=False):
        """This function computes predictions on test data.
        :param data: Dataset object - test dataset
        """

        # Declare the data pre processing
        data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(
            data, batch_size=self.config.BATCH_SIZE
        )

        accelerator.wait_for_everyone()

        self.model, test_loader = accelerator.prepare(self.model, test_loader)

        # This is required for distributed training
        test_files = [f.split("/")[-1] for f in test_loader.dataset.filepaths]

        log.info(f"Start inference for test data")
        predictions = []
        images = []
        classes = self.unseen_classes if standard_zsl else self.classes
        for img, _, _, img_path in test_loader:
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
            idx_preds = torch.argmax(logits, dim=1)

            if standard_zsl:
                predictions += [self.unseen_classes[i] for i in idx_preds]
            else:
                predictions += [self.classes[i] for i in idx_preds]

            images += [i for i in img_path]

        predictions = torch.tensor([self.label_to_idx[p] for p in predictions]).to(
            self.device
        )
        images = torch.tensor([test_files.index(img) for img in images]).to(self.device)

        accelerator.wait_for_everyone()

        predictions_outputs = accelerator.gather(predictions)
        image_outputs = accelerator.gather(images)

        predictions_outputs = [self.classes[p] for p in predictions_outputs]
        image_outputs = [test_files[i] for i in image_outputs]

        df_predictions = pd.DataFrame(
            {"id": image_outputs, "class": predictions_outputs}
        )
        df_predictions.drop_duplicates(subset=["id", "class"], inplace=True)

        return df_predictions

    def load_model_eval(self):
        self.define_model()

    def evaluation(self, data):
        """This function computes predictions on test data.
        :param data: Dataset object - test dataset
        """
        # Define model 
        #self.define_model()

        # Declare the data pre processing
        data.transform = self.transform
        # Define the data loader
        test_loader = torch.utils.data.DataLoader(
            data, batch_size=self.config.BATCH_SIZE
        )

        log.info(f"Start inference for test data")
        predictions = []
        images = []
        prob_preds = []
        classes = self.classes
        for img, _, _, img_path in test_loader:
            img = img.to(self.device)
            with torch.no_grad():
                # Get text and image prompts using UPT
                text_features, image_features = self.model(img, classes)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # Calculate image prompts
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            idx_preds = torch.argmax(logits, dim=1)

            predictions += [self.classes[i] for i in idx_preds]
            prob_preds += [logits]

            images += [i for i in img_path]

        #predictions = torch.tensor([p for p in predictions])
        prob_preds = torch.cat(prob_preds, axis=0).detach().to('cpu')

        log.info(f"Number of images: {len(images)}")
        log.info(f"Number of images: {len(predictions)}")
        log.info(f"Number of probs: {prob_preds.size()}")

        return images, predictions, prob_preds