from __future__ import annotations

import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms

from .dataset import (
    TRAIN_METADATA_SUFFIX,
    find_image_members,
    find_metadata_member,
    read_csv_from_archive,
)
from .preprocessing import IMAGE_SIZE, resize_rgb

IMAGE_TRANSFORMS = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]),
    "validation": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]),
}


def load_training_metadata(archive: zipfile.ZipFile) -> pd.DataFrame:
    member = find_metadata_member(archive, TRAIN_METADATA_SUFFIX)
    frame = read_csv_from_archive(archive, member)
    required = {"image_name", "target", "patient_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training metadata is missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=["image_name", "target", "patient_id"]).copy()
    frame["image_name"] = frame["image_name"].astype(str)
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["target"] = frame["target"].astype(int)
    if not set(frame["target"].unique()).issubset({0, 1}):
        raise ValueError("The training target must use binary labels 0 and 1.")
    return frame.reset_index(drop=True)


def patient_train_validation_test_split(
    metadata: pd.DataFrame,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_seed)
    development_indices, test_indices = next(
        outer_splitter.split(metadata, metadata["target"], groups=metadata["patient_id"])
    )
    development = metadata.iloc[development_indices].reset_index(drop=True)
    test = metadata.iloc[test_indices].reset_index(drop=True)
    inner_splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_seed + 1)
    train_indices, validation_indices = next(
        inner_splitter.split(development, development["target"], groups=development["patient_id"])
    )
    train = development.iloc[train_indices].reset_index(drop=True)
    validation = development.iloc[validation_indices].reset_index(drop=True)
    patient_sets = [set(frame["patient_id"]) for frame in (train, validation, test)]
    if any(patient_sets[left] & patient_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("Patient leakage detected across train, validation, and test splits.")
    if any(frame["target"].nunique() < 2 for frame in (train, validation, test)):
        raise ValueError("Each patient-level split must contain both positive and negative labels.")
    return train, validation, test


def limit_partition(frame: pd.DataFrame, max_samples: int | None, random_seed: int) -> pd.DataFrame:
    if max_samples is None or len(frame) <= max_samples:
        return frame.reset_index(drop=True)
    selected = []
    for _, class_frame in frame.groupby("target"):
        count = max(1, round(max_samples * len(class_frame) / len(frame)))
        selected.append(class_frame.sample(n=min(count, len(class_frame)), random_state=random_seed))
    limited = pd.concat(selected).drop_duplicates("image_name")
    if len(limited) > max_samples:
        limited = limited.sample(n=max_samples, random_state=random_seed)
    return limited.sample(frac=1, random_state=random_seed).reset_index(drop=True)


class ZipImageDataset(Dataset):
    def __init__(
        self,
        archive_path: Path,
        metadata: pd.DataFrame,
        transform,
        folder: str = "train",
    ):
        self.archive_path = archive_path
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform
        self.folder = folder
        self._archive: zipfile.ZipFile | None = None
        with zipfile.ZipFile(archive_path) as archive:
            metadata_member = find_metadata_member(archive, TRAIN_METADATA_SUFFIX)
            root = metadata_member.rsplit("/", 1)[0]
            self.members = find_image_members(archive, f"{root}/{folder}")
        missing = sorted(set(self.metadata["image_name"]) - self.members.keys())
        if missing:
            raise FileNotFoundError(f"{len(missing)} training images are missing; examples: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.archive_path)
        row = self.metadata.iloc[index]
        with self._archive.open(self.members[row["image_name"]]) as stream:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(stream) as source:
                    source.load()
                    rgb = np.asarray(source.convert("RGB"))
            image = Image.fromarray(resize_rgb(rgb, IMAGE_SIZE))
            tensor = self.transform(image)
        return tensor, torch.tensor(int(row["target"]), dtype=torch.long)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archive"] = None
        return state

    def __del__(self):
        if self._archive is not None:
            self._archive.close()
