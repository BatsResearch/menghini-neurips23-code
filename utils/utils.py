import logging
import random

import numpy as np
import torch


log = logging.getLogger(__name__)


def dataset_object(dataset_name):
    if dataset_name == "aPY":
        from data import aPY as DataObject
    elif dataset_name == "Animals_with_Attributes2":
        from data import AwA2 as DataObject
    elif dataset_name == "EuroSAT":
        from data import EuroSAT as DataObject
    elif dataset_name == "DTD":
        from data import DTD as DataObject
    elif dataset_name == "sun397":
        from data import SUN397 as DataObject
    elif dataset_name == "CUB":
        from data import CUB as DataObject
    elif dataset_name == "RESICS45":
        from data import RESICS45 as DataObject
    elif dataset_name == "FGVCAircraft":
        from data import FGVCAircraft as DataObject
    elif dataset_name == "MNIST":
        from data import MNIST as DataObject
    elif dataset_name == "Flowers102":
        from data import Flowers102 as DataObject

    return DataObject


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Config(object):
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)
