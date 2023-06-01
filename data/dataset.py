import logging
import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        # Adjust filepaths
        self.train = train
        if self.train:
            self.filepaths = [f"{root}/train/{f}" for f in filepaths]
        else:
            self.filepaths = [f"{root}/test/{f}" for f in filepaths]

        self.transform = transform
        if augmentations:
            self.aug1_transform = augmentations[0]
            self.aug2_transform = augmentations[1]
        else:
            self.aug1_transform = None
            self.aug2_transform = None
        self.labels = labels
        self.label_id = label_id
        self.label_map = label_map

    def __len__(self):
        # dataset size
        return len(self.filepaths)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, aug1, aug2, target) where target is index of the target class.
        """

        img = Image.open(self.filepaths[index]).convert("RGB")

        # Apply two transformations (strong and weak)
        if self.aug1_transform is not None:
            aug_1 = self.aug1_transform(img)
        else:
            img1 = self.transform(img)
            aug_1 = img1
        if self.aug2_transform is not None:
            aug_2 = self.aug2_transform(img)
        else:
            img2 = self.transform(img)
            aug_2 = img2

        if self.transform is not None:
            img = self.transform(img)

        # Get image label
        if self.labels is not None:
            if self.label_id:
                label = int(self.labels[index])
            else:
                label = int(self.label_map[self.labels[index]])
            return img, aug_1, aug_2, label, self.filepaths[index].split("/")[-1]
        else:
            return img, aug_1, aug_2, self.filepaths[index].split("/")[-1]



class EuroSAT(CustomDataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
        class_folder=False,
        original_filepaths=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        super().__init__(
            filepaths,
            root,
            transform,
            augmentations,
            train,
            labels,
            label_id,
            label_map,
        )
        # Adjust filepaths
        self.filepaths = [f"{root}/{f.split('_')[0]}/{f}" for f in filepaths]


class DTD(CustomDataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
        class_folder=False,
        original_filepaths=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        super().__init__(
            filepaths,
            root,
            transform,
            augmentations,
            train,
            labels,
            label_id,
            label_map,
        )
        # Adjust filepaths
        if class_folder:
            paths = []
            for f in filepaths:
                cl = f.split("_")[0]
                tr_files = os.listdir(f"{root}/train/{cl}")
                val_files = os.listdir(f"{root}/val/{cl}")
                if f in tr_files:
                    paths.append(f"{root}/train/{cl}/{f}")
                elif f in val_files:
                    paths.append(f"{root}/val/{cl}/{f}")

            self.filepaths = paths

        else:
            self.filepaths = [f"{root}/{f}" for f in filepaths]


class CUB(CustomDataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
        class_folder=False,
        original_filepaths=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        super().__init__(
            filepaths,
            root,
            transform,
            augmentations,
            train,
            labels,
            label_id,
            label_map,
        )
        # Adjust filepaths
        self.filepaths = [f"{root}/{f}" for f in filepaths]


class RESICS45(CustomDataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
        class_folder=False,
        original_filepaths=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        super().__init__(
            filepaths,
            root,
            transform,
            augmentations,
            train,
            labels,
            label_id,
            label_map,
        )
        # Adjust filepaths
        self.filepaths = []
        for f in filepaths:
            folder = "_".join(f.split("_")[:-1])
            self.filepaths.append(f"{root}/{folder}/{f}")


class FGVCAircraft(CustomDataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
        class_folder=False,
        original_filepaths=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        super().__init__(
            filepaths,
            root,
            transform,
            augmentations,
            train,
            labels,
            label_id,
            label_map,
        )
        if class_folder:
            filepaths = list(filepaths)
            new_paths = []
            for f in original_filepaths:
                img = f.split("/")[-1]
                if img in filepaths:
                    new_paths.append(f"{f}")

            self.filepaths = new_paths
        else:
            # Adjust filepaths
            self.filepaths = [f"{root}/{f}" for f in filepaths]


class MNIST(CustomDataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
        class_folder=False,
        original_filepaths=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        super().__init__(
            filepaths,
            root,
            transform,
            augmentations,
            train,
            labels,
            label_id,
            label_map,
        )
        if class_folder:
            filepaths = list(filepaths)
            new_paths = []
            for f in original_filepaths:
                img = f.split("/")[-1]
                if img in filepaths:
                    new_paths.append(f"{f}")

            self.filepaths = new_paths
        else:
            # Adjust filepaths
            self.filepaths = [f"{root}/{f}" for f in filepaths]


class Flowers102(CustomDataset):
    def __init__(
        self,
        filepaths,
        root,
        transform,
        augmentations=None,
        train=True,
        labels=None,
        label_id=False,
        label_map=None,
        class_folder=False,
        original_filepaths=None,
    ):
        """
        :param filepaths: list of images
        :param root: path to images
        :param transform: standard transform
        :param augmentations: None or tuple
        :param train: indicates in the data is in train or test folder
        :param labels: list of label
        :param label_id: true if labeles are passed as int
        :param label_map: dict mpping string labels to int
        """
        super().__init__(
            filepaths,
            root,
            transform,
            augmentations,
            train,
            labels,
            label_id,
            label_map,
        )
        # Adjust filepaths
        if class_folder:
            filepaths = list(filepaths)
            new_paths = []
            for f in original_filepaths:
                img = f.split("/")[-1]
                if img in filepaths:
                    new_paths.append(f"{f}")

            self.filepaths = new_paths

        else:
            self.filepaths = [f"{root}/{f}" for f in filepaths]
