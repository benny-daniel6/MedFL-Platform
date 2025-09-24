import os
import zipfile
import random
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split


def download_and_unzip_data():
    DATASET_NAME = "lgg-mri-segmentation"
    DATA_DIR = "data"
    if os.path.exists(os.path.join(DATA_DIR, DATASET_NAME)):
        print("Dataset already found.")
        return
    print("Please download the dataset manually and place it in the 'data' folder.")


class BrainMRIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # --- Synchronized Transformations ---
        # 1. Resize
        resize = transforms.Resize(size=(192, 192))
        image = resize(image)
        mask = resize(mask)

        # 2. Apply random augmentations only on the training set
        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # 3. Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # 4. Normalize the image (but not the mask)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)

        mask = (mask > 0.5).float()
        return image, mask


def load_and_prepare_data(data_dir="data/lgg-mri-segmentation"):
    all_files = glob(os.path.join(data_dir, "*", "*"))
    images = sorted([f for f in all_files if "mask" not in f])
    masks = sorted([f for f in all_files if "mask" in f])

    df = pd.DataFrame({"image": images, "mask": masks})
    df["patient_id"] = df["image"].apply(
        lambda x: os.path.normpath(x).split(os.sep)[-2]
    )

    patient_ids = df["patient_id"].unique()
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)

    train_df = df[df["patient_id"].isin(train_ids)]
    test_df = df[df["patient_id"].isin(test_ids)]

    return (
        train_df["image"].tolist(),
        train_df["mask"].tolist(),
        test_df["image"].tolist(),
        test_df["mask"].tolist(),
    )


def partition_data(train_images, train_masks, num_clients, is_iid=True):
    # This function remains the same as before
    partitions_img = [[] for _ in range(num_clients)]
    partitions_mask = [[] for _ in range(num_clients)]
    if is_iid:
        num_items_per_client = len(train_images) // num_clients
        for i in range(num_clients):
            start = i * num_items_per_client
            end = start + num_items_per_client
            partitions_img[i] = train_images[start:end]
            partitions_mask[i] = train_masks[start:end]
    else:
        mask_sums = [
            np.sum(np.array(Image.open(p)) > 0)
            for p in tqdm(train_masks, desc="Analyzing masks")
        ]
        sorted_indices = np.argsort(mask_sums)
        sorted_images = [train_images[i] for i in sorted_indices]
        sorted_masks = [train_masks[i] for i in sorted_indices]
        num_shards, shard_size = num_clients * 2, len(sorted_images) // (
            num_clients * 2
        )
        shards_img = [
            sorted_images[i : i + shard_size]
            for i in range(0, len(sorted_images), shard_size)
        ]
        shards_mask = [
            sorted_masks[i : i + shard_size]
            for i in range(0, len(sorted_masks), shard_size)
        ]
        client_idx = 0
        for i in range(0, len(shards_img), 2):
            if client_idx < num_clients:
                partitions_img[client_idx].extend(shards_img[i] + shards_img[i + 1])
                partitions_mask[client_idx].extend(shards_mask[i] + shards_mask[i + 1])
                client_idx += 1
    return partitions_img, partitions_mask


def get_dataloader(image_paths, mask_paths, batch_size=32, is_train=True):
    dataset = BrainMRIDataset(image_paths, mask_paths, is_train=is_train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=0 if os.name == "nt" else 2,
    )
