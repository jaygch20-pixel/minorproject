from __future__ import annotations

import io
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

TRAIN_METADATA_SUFFIX = "isic_2020_train_metadata.csv"
TEST_METADATA_SUFFIX = "isic_2020_test_metadata.csv"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def find_metadata_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.lower().endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {suffix} in archive; found {len(matches)}.")
    return matches[0]


def read_csv_from_archive(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(archive.read(member)))


def find_image_members(archive: zipfile.ZipFile, folder: str) -> dict[str, str]:
    prefix = folder.rstrip("/") + "/"
    result: dict[str, str] = {}
    for member in archive.namelist():
        if not member.lower().startswith(prefix.lower()) or not member.lower().endswith(IMAGE_SUFFIXES):
            continue
        image_id = Path(member).stem
        if image_id in result:
            raise ValueError(
                f"Duplicate image ID {image_id!r} in archive folder {folder!r}: "
                f"{result[image_id]!r} and {member!r}."
            )
        result[image_id] = member
    return result


def inspect_archive(archive_path: Path) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        uncompressed_bytes = sum(info.file_size for info in archive.infolist())
        train_csv = find_metadata_member(archive, TRAIN_METADATA_SUFFIX)
        test_csv = find_metadata_member(archive, TEST_METADATA_SUFFIX)
        train = read_csv_from_archive(archive, train_csv)
        test = read_csv_from_archive(archive, test_csv)
        train_image_col = "image_name" if "image_name" in train.columns else "image"
        test_image_col = "image" if "image" in test.columns else "image_name"
        target_col = "target" if "target" in train.columns else None
        train_root = train_csv.rsplit("/", 1)[0] + "/train"
        test_root = test_csv.rsplit("/", 1)[0] + "/test"
        train_images = find_image_members(archive, train_root)
        test_images = find_image_members(archive, test_root)
        train_extensions = Counter(Path(name).suffix.lower() for name in train_images.values())
        test_extensions = Counter(Path(name).suffix.lower() for name in test_images.values())
        mask_members = [
            name for name in members
            if any(token in name.lower() for token in ("mask", "segmentation", "ground_truth", "ground-truth"))
        ]
        train_ids = train[train_image_col].dropna().astype(str)
        test_ids = test[test_image_col].dropna().astype(str)
        target_counts = train[target_col].value_counts(dropna=False).to_dict() if target_col else {}
        return {
            "archive_name": archive_path.name,
            "archive_bytes": archive_path.stat().st_size,
            "archive_uncompressed_bytes": uncompressed_bytes,
            "free_disk_bytes": shutil.disk_usage(archive_path.parent).free,
            "archive_members": len(members),
            "train_metadata_member": train_csv,
            "train_columns": list(train.columns),
            "train_records": len(train),
            "train_image_column": train_image_col,
            "target_column": target_col,
            "target_counts": {str(key): int(value) for key, value in target_counts.items()},
            "label_encoding": {"0": "non_melanoma", "1": "melanoma"},
            "test_metadata_member": test_csv,
            "test_columns": list(test.columns),
            "test_records": len(test),
            "test_image_column": test_image_col,
            "train_images": len(train_images),
            "train_image_extensions": dict(train_extensions),
            "test_images": len(test_images),
            "test_image_extensions": dict(test_extensions),
            "train_ids_matched": int(train_ids.isin(train_images).sum()),
            "test_ids_matched": int(test_ids.isin(test_images).sum()),
            "train_missing_image_ids": sorted(set(train_ids) - train_images.keys())[:20],
            "test_missing_image_ids": sorted(set(test_ids) - test_images.keys())[:20],
            "train_image_examples": list(train_images.values())[:5],
            "test_image_examples": list(test_images.values())[:5],
            "patient_column": "patient_id" if "patient_id" in train.columns else None,
            "unique_train_patients": int(train["patient_id"].nunique()) if "patient_id" in train.columns else None,
            "mask_candidates": mask_members[:30],
            "verified_masks_available": False,
            "test_labels_available": "target" in test.columns,
        }
