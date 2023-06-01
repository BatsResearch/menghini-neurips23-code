import logging
import os
import pickle

import clip
import torch
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)


def compute_pseudo_labels(
    k,
    template,
    dataset,
    classnames,
    transform,
    clip_model,
    label_to_idx,
    device,
    filename,
):
    prompts = [f"{template}{' '.join(i.split('_'))}" for i in classnames]
    text = clip.tokenize(prompts).to(device)

    if k == 10000000:
        log.info(f"Compute pseudo-labeles on all unlabeled data")
        new_labels = []
        new_imgs = []
        for i, image_path in enumerate(tqdm(dataset.filepaths)):
            img = Image.open(image_path).convert("RGB")
            img = transform(img).to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = clip_model(
                    torch.unsqueeze(img, 0).to(device), text
                )
                probs = logits_per_image.softmax(dim=-1)
                idx_preds = torch.argmax(probs, dim=1)
                pred_id = idx_preds.item()
                pred = label_to_idx[classnames[idx_preds.item()]]
            
            new_labels.append(pred)
            new_imgs.append(image_path)

    else:

        # to find the top k for each class, each class has it's own "leaderboard"
        top_k_leaderboard = {
            label_to_idx[classnames[i]]: [] for i in range(len(classnames))
        }  # maps class idx -> (confidence, image_path) tuple

        log.info(f"Compute {k} pseudo-labeles")
        # log.info(f"{label_to_idx}")
        for i, image_path in enumerate(tqdm(dataset.filepaths)):
            img = Image.open(image_path).convert("RGB")
            img = transform(img).to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = clip_model(
                    torch.unsqueeze(img, 0).to(device), text
                )
                probs = logits_per_image.softmax(dim=-1)
                idx_preds = torch.argmax(probs, dim=1)
                pred_id = idx_preds.item()
                pred = label_to_idx[classnames[idx_preds.item()]]
                # log.info(f"{classnames[idx_preds.item()]}")
                # log.info(f"{pred}")

            """if predicted class has empty leaderboard, or if the confidence is high
            enough for predicted class leaderboard, add the new example
            """
            prob_score = probs[0][pred_id]
            if len(top_k_leaderboard[pred]) < k:
                top_k_leaderboard[pred].append((prob_score, image_path))
            elif (
                top_k_leaderboard[pred][-1][0] < prob_score
            ):  # if the confidence in predicted class "qualifies" for top-k
                # default sorting of tuples is by first element
                top_k_leaderboard[pred] = sorted(
                    top_k_leaderboard[pred] + [(probs[0][pred_id], image_path)],
                    reverse=True,
                )[:k]
            else:
                # sort the other classes by confidence score
                order_of_classes = sorted(
                    [(probs[0][j], j) for j in range(len(classnames)) if j != pred_id],
                    reverse=True,
                )
                for score, index in order_of_classes:
                    index_dict = label_to_idx[classnames[index]]
                    # log.info(f"{classnames[index]}")
                    # log.info(f"{index_dict}")
                    if len(top_k_leaderboard[index_dict]) < k:
                        top_k_leaderboard[index_dict].append((probs[0][index], image_path))
                    elif top_k_leaderboard[index_dict][-1][0] < probs[0][index]:
                        # default sorting of tuples is by first element
                        top_k_leaderboard[index_dict] = sorted(
                            top_k_leaderboard[index_dict]
                            + [((probs[0][index], image_path))],
                            reverse=True,
                        )[:k]

        new_imgs = []
        new_labels = []
        # loop through, and rebuild the dataset
        for index, leaderboard in top_k_leaderboard.items():
            # print(len(dataset.imgs))
            new_imgs += [tup[1] for tup in leaderboard]
            new_labels += [index for _ in leaderboard]

    dataset.filepaths = new_imgs
    dataset.labels = new_labels

    with open(filename, "wb") as f:
        pickle.dump({"filepaths": new_imgs, "labels": new_labels}, f)

    return dataset


def pseudolabel_top_k(
    config,
    data_name,
    k,
    template,
    dataset,
    classnames,
    transform,
    clip_model,
    label_to_idx,
    device,
    vis_encoder,
    split_seed, 
):
    filename = f"pseudolabels/{data_name}_{vis_encoder.replace('/', '')}_{config.LEARNING_PARADIGM}_{config.MODEL}_{k}_pseudolabels_split_{split_seed}.pickle"
    if os.path.exists(filename):
        # print('Load pseudolabels')
        with open(filename, "rb") as f:
            pseudolabels = pickle.load(f)
            new_imgs = pseudolabels["filepaths"]
            new_labels = pseudolabels["labels"]

            dataset.filepaths = new_imgs
            dataset.labels = new_labels
    else:
        dataset = compute_pseudo_labels(
            k,
            template,
            dataset,
            classnames,
            transform,
            clip_model,
            label_to_idx,
            device,
            filename,
        )

    return dataset