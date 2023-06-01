import json
import logging
import os
import random

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def get_class_names(dataset, dataset_dir, seed=500):
    """Returns the lists of the names of all classes, seen classes,
    and unseen classes.

    :param dataset: name of the dataset in use
    :param dataset_dir: path to get dataset dir (on CCV)
    """
    if dataset == "aPY":
        path = f"{dataset_dir}/{dataset}/proposed_split"
        seen_classes = []
        unseen_classes = []
        with open(f"{path}/trainvalclasses.txt", "r") as f:
            for l in f:
                seen_classes.append(l.strip())

        with open(f"{path}/testclasses.txt", "r") as f:
            for l in f:
                unseen_classes.append(l.strip())

        # Adjust class names
        correction_dict = {
            "diningtable": "dining table",
            "tvmonitor": "tv monitor",
            "jetski": "jet ski",
            "pottedplant": "potted plant",
        }
        for c in seen_classes:
            if c in correction_dict:
                seen_classes[seen_classes.index(c)] = correction_dict[c]
        for c in unseen_classes:
            if c in correction_dict:
                unseen_classes[unseen_classes.index(c)] = correction_dict[c]

        classes = seen_classes + unseen_classes

    elif dataset == "Animals_with_Attributes2":
        path = f"{dataset_dir}/{dataset}"

        seen_classes = []
        unseen_classes = []
        df = pd.read_csv(f"{path}/trainvalclasses.txt")
        with open(f"{path}/trainvalclasses.txt", "r") as f:
            for l in f:
                seen_classes.append(l.strip())

        with open(f"{path}/testclasses.txt", "r") as f:
            for l in f:
                unseen_classes.append(l.strip())

        # Adjust class names
        correction_dict = {
            "grizzly+bear": "grizzly bear",
            "killer+whale": "killer whale",
            "persian+cat": "persian cat",
            "german+shepherd": "german shepherd",
            "blue+whale": "blue whale",
            "siamese+cat": "siamese cat",
            "spider+monkey": "spider monkey",
            "humpback+whale": "humpback whale",
            "giant+panda": "giant panda",
            "polar+bear": "polar bear",
        }

        for c in seen_classes:
            if c in correction_dict:
                seen_classes[seen_classes.index(c)] = correction_dict[c]
        for c in unseen_classes:
            if c in correction_dict:
                unseen_classes[unseen_classes.index(c)] = correction_dict[c]

        classes = seen_classes + unseen_classes

    elif dataset == "EuroSAT":
        path = f"{dataset_dir}/{dataset}"

        classes = []
        with open(f"{path}/class_names.txt", "r") as f:
            for l in f:
                classes.append(l.strip())

        np.random.seed(seed)
        seen_indices = np.random.choice(
            range(len(classes)), size=int(len(classes) * 0.62), replace=False
        )
        unseen_indices = list(set(range(len(classes))).difference(set(seen_indices)))

        seen_classes = list(np.array(classes)[seen_indices])
        unseen_classes = list(np.array(classes)[unseen_indices])

    elif dataset == "DTD":
        path = f"{dataset_dir}/{dataset}"

        classes = []
        with open(f"{path}/class_names.txt", "r") as f:
            for l in f:
                classes.append(l.strip())

        np.random.seed(seed)
        seen_indices = np.random.choice(
            range(len(classes)), size=int(len(classes) * 0.62), replace=False
        )
        unseen_indices = list(set(range(len(classes))).difference(set(seen_indices)))

        seen_classes = list(np.array(classes)[seen_indices])
        unseen_classes = list(np.array(classes)[unseen_indices])

    elif dataset == "RESICS45":
        path = f"{dataset_dir}/{dataset}"

        classes = []
        with open(f"{path}/train.json", "r") as f:
            data = json.load(f)
            for d in data["categories"]:
                classes.append(d["name"].replace("_", " "))

        np.random.seed(seed)
        seen_indices = np.random.choice(
            range(len(classes)), size=int(len(classes) * 0.62), replace=False
        )
        unseen_indices = list(set(range(len(classes))).difference(set(seen_indices)))

        seen_classes = list(np.array(classes)[seen_indices])
        unseen_classes = list(np.array(classes)[unseen_indices])

    elif dataset == "FGVCAircraft":
        path = f"{dataset_dir}/{dataset}"

        classes = []
        with open(f"{path}/labels.txt", "r") as f:
            for l in f:
                classes.append(l.strip())

        np.random.seed(seed)
        seen_indices = np.random.choice(
            range(len(classes)), size=int(len(classes) * 0.62), replace=False
        )
        unseen_indices = list(set(range(len(classes))).difference(set(seen_indices)))

        seen_classes = list(np.array(classes)[seen_indices])
        unseen_classes = list(np.array(classes)[unseen_indices])

    elif dataset == "MNIST":
        path = f"{dataset_dir}/{dataset}"

        classes = []
        with open(f"{path}/labels.txt", "r") as f:
            for l in f:
                classes.append(l.strip())

        np.random.seed(seed)
        seen_indices = np.random.choice(
            range(len(classes)), size=int(len(classes) * 0.62), replace=False
        )
        unseen_indices = list(set(range(len(classes))).difference(set(seen_indices)))

        seen_classes = list(np.array(classes)[seen_indices])
        unseen_classes = list(np.array(classes)[unseen_indices])

    elif dataset == "Flowers102":
        path = f"{dataset_dir}/{dataset}"

        classes = []
        with open(f"{path}/class_names.txt", "r") as f:
            for l in f:
                classes.append(l.strip())

        np.random.seed(seed)
        seen_indices = np.random.choice(
            range(len(classes)), size=int(len(classes) * 0.62), replace=False
        )
        unseen_indices = list(set(range(len(classes))).difference(set(seen_indices)))

        seen_classes = list(np.array(classes)[seen_indices])
        unseen_classes = list(np.array(classes)[unseen_indices])

    elif dataset == "CUB":
        path = f"{dataset_dir}/{dataset}"

        seen_classes = []
        unseen_classes = []
        with open(f"{path}/trainvalclasses.txt", "r") as f:
            for l in f:
                seen_classes.append(
                    l.strip().split(".")[-1].strip().replace("_", " ").lower()
                )

        with open(f"{path}/testclasses.txt", "r") as f:
            for l in f:
                unseen_classes.append(
                    l.strip().split(".")[-1].strip().replace("_", " ").lower()
                )

        classes = seen_classes + unseen_classes

    return classes, seen_classes, unseen_classes


def get_labeled_and_unlabeled_data(
    dataset, data_folder, seen_classes, unseen_classes, classes=None
):
    """This function returns the list of
    - labeled_data: each item is (image name, class name)
    - unlabeled_data: each item is (image name, class name)
    - test_data: each item is (image name, class name)

    :param dataset: dataset name
    :param data_folder: path to folder of images
    :param seen_classes: list of seen classes' names
    :param unseen_classes: list of unseen classes' names
    """
    if dataset == "aPY":
        image_data = pd.read_csv(f"{data_folder}/image_data.csv", sep=",")

        list_images = []
        for i, row in image_data.iterrows():
            if (
                row["image_path"] == "yahoo_test_images/bag_227.jpg"
                or row["image_path"] == "yahoo_test_images/mug_308.jpg"
            ):
                list_images.append(f"broken")
            else:
                list_images.append(f"{i}.jpg")

        image_data["file_names"] = list_images
        correction_dict = {
            "diningtable": "dining table",
            "tvmonitor": "tv monitor",
            "jetski": "jet ski",
            "pottedplant": "potted plant",
        }
        image_data["label"] = image_data["label"].apply(
            lambda x: correction_dict[x] if x in correction_dict else x
        )
        image_data["seen"] = image_data["label"].apply(
            lambda x: 1 if x in seen_classes else 0
        )

        labeled_files = list(
            image_data[
                (image_data["seen"] == 1) & (image_data["file_names"] != "broken")
            ]["file_names"]
        )
        labels_files = list(
            image_data[
                (image_data["seen"] == 1) & (image_data["file_names"] != "broken")
            ]["label"]
        )

        unlabeled_lab_files = list(
            image_data[
                (image_data["seen"] == 0) & (image_data["file_names"] != "broken")
            ]["file_names"]
        )
        unlabeled_labs = list(
            image_data[
                (image_data["seen"] == 0) & (image_data["file_names"] != "broken")
            ]["label"]
        )

    elif dataset == "Animals_with_Attributes2":
        labeled_files = []
        labels_files = []
        for c in seen_classes:
            files = os.listdir(f"{data_folder}/JPEGImages/{c.replace(' ', '+')}")
            labeled_files += files
            labels_files += [c] * len(files)

        unlabeled_lab_files = []
        unlabeled_labs = []
        for c in unseen_classes:
            files = os.listdir(f"{data_folder}/JPEGImages/{c.replace(' ', '+')}")
            unlabeled_lab_files += files
            unlabeled_labs += [c] * len(files)

    elif dataset == "EuroSAT":
        correction_dict = {
            "annual crop land": "AnnualCrop",
            "brushland or shrubland": "HerbaceousVegetation",
            "highway or road": "Highway",
            "industrial buildings or commercial buildings": "Industrial",
            "pasture land": "Pasture",
            "permanent crop land": "PermanentCrop",
            "residential buildings or homes or apartments": "Residential",
            "lake or sea": "SeaLake",
            "river": "River",
            "forest": "Forest",
        }

        labeled_files = []
        labels_files = []
        for c in seen_classes:
            files = os.listdir(f"{data_folder}/{correction_dict[c]}")
            labeled_files += files
            labels_files += [c] * len(files)

        unlabeled_lab_files = []
        unlabeled_labs = []
        for c in unseen_classes:
            files = os.listdir(f"{data_folder}/{correction_dict[c]}")
            unlabeled_lab_files += files
            unlabeled_labs += [c] * len(files)

        labeled_data = list(zip(labeled_files, labels_files))
        unlabeled_data = list(zip(unlabeled_lab_files, unlabeled_labs))

        test_files = []
        test_labs = []
        with open(f"{data_folder}/test.txt", "r") as f:
            for l in f:
                line = l.split(" ")
                test_files.append(line[0].strip().split("@")[-1].split("/")[-1])
                test_labs.append(classes[int(line[1].strip())])
        test_data = list(zip(test_files, test_labs))

        return labeled_data, unlabeled_data, test_data

    elif dataset == "DTD":
        labeled_files = []
        labels_files = []

        unlabeled_lab_files = []
        unlabeled_labs = []

        for split in ["train", "val"]:
            with open(f"{data_folder}/{split}.txt", "r") as f:
                for l in f:
                    line = l.split(" ")
                    cl = classes[int(line[1].strip())]
                    if cl in seen_classes:
                        labeled_files.append(
                            f"{split}/{line[0].strip().split('@')[-1]}"
                        )
                        labels_files.append(cl)
                    elif cl in unseen_classes:
                        unlabeled_lab_files.append(
                            f"{split}/{line[0].strip().split('@')[-1]}"
                        )
                        unlabeled_labs.append(cl)
                    else:
                        raise Exception(
                            f"The extracted class is not among the seen or unseen classes."
                        )

        labeled_data = list(zip(labeled_files, labels_files))
        unlabeled_data = list(zip(unlabeled_lab_files, unlabeled_labs))

        test_files = []
        test_labs = []

        with open(f"{data_folder}/test.txt", "r") as f:
            for l in f:
                line = l.split(" ")
                test_files.append(f"test/{line[0].strip().split('@')[-1]}")
                test_labs.append(classes[int(line[1].strip())])
        test_data = list(zip(test_files, test_labs))

        return labeled_data, unlabeled_data, test_data

    elif dataset == "RESICS45":
        labeled_files = []
        labels_files = []

        unlabeled_lab_files = []
        unlabeled_labs = []

        for split in ["train", "val"]:
            with open(f"{data_folder}/{split}.json", "r") as f:
                data = json.load(f)
                for d in data["images"]:
                    file_name = d["file_name"].split("@")[-1]
                    cl = file_name.split("/")[0].replace("_", " ")
                    img = file_name.split("/")[-1]

                    if cl in seen_classes:
                        labeled_files.append(img)
                        labels_files.append(cl)
                    elif cl in unseen_classes:
                        unlabeled_lab_files.append(img)
                        unlabeled_labs.append(cl)
                    else:
                        raise Exception(
                            f"The extracted class is not among the seen or unseen classes."
                        )
        N1 = 500 # 4
        N2 = 50 # 50
        N3 = 10
        labeled_data = list(zip(labeled_files, labels_files))#[:N1]
        unlabeled_data = list(zip(unlabeled_lab_files, unlabeled_labs))#[:N2]

        test_files = []
        test_labs = []

        with open(f"{data_folder}/test.json", "r") as f:
            data = json.load(f)
            for d in data["images"]:
                file_name = d["file_name"].split("@")[-1]
                cl = file_name.split("/")[0].replace("_", " ")
                img = file_name.split("/")[-1]

                test_files.append(img)
                test_labs.append(cl)

        test_data = list(zip(test_files, test_labs))#[:N3]

        return labeled_data, unlabeled_data, test_data

    elif dataset == "FGVCAircraft":
        labeled_files = []
        labels_files = []

        unlabeled_lab_files = []
        unlabeled_labs = []

        for split in ["train", "val"]:
            with open(f"{data_folder}/{split}.txt", "r") as f:
                for l in f:
                    img = " ".join(l.split(" ")[:-1]).split("@")[-1].strip()
                    cl = img.split("/")[0].strip()

                    if cl in seen_classes:
                        labeled_files.append(f"{split}/{img}")
                        labels_files.append(cl)
                    elif cl in unseen_classes:
                        unlabeled_lab_files.append(f"{split}/{img}")
                        unlabeled_labs.append(cl)
                    else:
                        raise Exception(
                            f"The extracted class {cl} is not among the seen or unseen classes."
                        )

        labeled_data = list(zip(labeled_files, labels_files))
        unlabeled_data = list(zip(unlabeled_lab_files, unlabeled_labs))

        test_files = []
        test_labs = []

        with open(f"{data_folder}/test.txt", "r") as f:
            for l in f:
                img = " ".join(l.split(" ")[:-1]).split("@")[-1].strip()
                cl = img.split("/")[0].strip()

                test_files.append(f"test/{img}")
                test_labs.append(cl)

        test_data = list(zip(test_files, test_labs))

        return labeled_data, unlabeled_data, test_data

    elif dataset == "MNIST":
        labeled_files = []
        labels_files = []

        unlabeled_lab_files = []
        unlabeled_labs = []

        split = "train"
        with open(f"{data_folder}/{split}.txt", "r") as f:
            for l in f:
                img = l.split(" ")[0].split("@")[-1].strip()
                cl = img.split("/")[0].strip()

                if cl in seen_classes:
                    labeled_files.append(f"{split}/{img}")
                    labels_files.append(cl)
                elif cl in unseen_classes:
                    unlabeled_lab_files.append(f"{split}/{img}")
                    unlabeled_labs.append(cl)
                else:
                    raise Exception(
                        f"The extracted class {cl} is not among the seen or unseen classes."
                    )

        labeled_data = list(zip(labeled_files, labels_files))
        unlabeled_data = list(zip(unlabeled_lab_files, unlabeled_labs))

        test_files = []
        test_labs = []

        with open(f"{data_folder}/test.txt", "r") as f:
            for l in f:
                img = l.split(" ")[0].split("@")[-1].strip()
                cl = img.split("/")[0].strip()

                test_files.append(f"test/{img}")
                test_labs.append(cl)

        test_data = list(zip(test_files, test_labs))

        return labeled_data, unlabeled_data, test_data

    elif dataset == "Flowers102":
        labeled_files = []
        labels_files = []

        unlabeled_lab_files = []
        unlabeled_labs = []

        for split in ["train", "val"]:
            with open(f"{data_folder}/{split}.txt", "r") as f:
                for l in f:
                    line = l.split(" ")
                    img = line[0].split("@")[-1].strip()
                    cl = classes[int(line[1].strip())]

                    if cl in seen_classes:
                        labeled_files.append(f"{split}/{img}")
                        labels_files.append(cl)
                    elif cl in unseen_classes:
                        unlabeled_lab_files.append(f"{split}/{img}")
                        unlabeled_labs.append(cl)
                    else:
                        raise Exception(
                            f"The extracted class {cl} is not among the seen or unseen classes."
                        )

        labeled_data = list(zip(labeled_files, labels_files))
        unlabeled_data = list(zip(unlabeled_lab_files, unlabeled_labs))

        test_files = []
        test_labs = []

        with open(f"{data_folder}/test.txt", "r") as f:
            for l in f:
                line = l.split(" ")
                img = line[0].split("@")[-1].strip()
                cl = classes[int(line[1].strip())]

                test_files.append(f"test/{img}")
                test_labs.append(cl)

        test_data = list(zip(test_files, test_labs))

        return labeled_data, unlabeled_data, test_data

    elif dataset == "CUB":
        labeled_files = []
        labels_files = []

        unlabeled_lab_files = []
        unlabeled_labs = []

        with open(f"{data_folder}/train.txt", "r") as f:
            for l in f:
                line = l.strip()
                cl = line.split("/")[0].split(".")[-1].strip().replace("_", " ").lower()
                if cl in seen_classes:
                    labeled_files.append(f"CUB_200_2011/images/{line}")
                    labels_files.append(cl)
                elif cl in unseen_classes:
                    unlabeled_lab_files.append(f"CUB_200_2011/images/{line}")
                    unlabeled_labs.append(cl)
                else:
                    raise Exception(
                        f"The extracted class is not among the seen or unseen classes."
                    )

        labeled_data = list(zip(labeled_files, labels_files))
        unlabeled_data = list(zip(unlabeled_lab_files, unlabeled_labs))

        test_files = []
        test_labs = []

        with open(f"{data_folder}/test.txt", "r") as f:
            for l in f:
                line = l.strip()
                test_files.append(f"CUB_200_2011/images/{line}")
                test_labs.append(
                    line.split("/")[0].split(".")[-1].strip().replace("_", " ").lower()
                )
        test_data = list(zip(test_files, test_labs))

        return labeled_data, unlabeled_data, test_data

    # Split labeled and unlabeled data into test
    train_labeled_files, train_labeles, test_seen_files, test_seen_labs = split_data(
        0.8, labeled_files, labels_files
    )
    labeled_data = list(zip(train_labeled_files, train_labeles))

    (
        train_unlabeled_files,
        train_un_labeles,
        test_unseen_files,
        test_unseen_labs,
    ) = split_data(0.8, unlabeled_lab_files, unlabeled_labs)
    unlabeled_data = list(zip(train_unlabeled_files, train_un_labeles))

    test_seen = list(zip(test_seen_files, test_seen_labs))
    test_unseen = list(zip(test_unseen_files, test_unseen_labs))

    test_data = test_seen + test_unseen

    return labeled_data, unlabeled_data, test_data


def split_data(ratio, files, labels):
    np.random.seed(500)
    train_indices = np.random.choice(
        range(len(files)), size=int(len(files) * ratio), replace=False
    )
    val_indices = list(set(range(len(files))).difference(set(train_indices)))

    train_labeled_files = np.array(files)[train_indices]
    train_labeles = np.array(labels)[train_indices]

    val_labeled_files = np.array(files)[val_indices]
    val_labeles = np.array(labels)[val_indices]

    return train_labeled_files, train_labeles, val_labeled_files, val_labeles
