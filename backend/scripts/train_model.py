from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.cnn import create_resnet18
from app.services.dataset import find_metadata_member
from app.services.preprocessing import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from app.services.training_data import (
    IMAGE_TRANSFORMS,
    ZipImageDataset,
    limit_partition,
    load_training_metadata,
    patient_train_validation_test_split,
)

ARCHIVE_PATH = BACKEND_DIR.parent / "archive.zip"
MODEL_DIR = BACKEND_DIR / "model"
CHECKPOINT_PATH = MODEL_DIR / "resnet18_isic2020.pt"
CONFIG_PATH = MODEL_DIR / "preprocessing.json"
RESUME_CHECKPOINT_PATH = MODEL_DIR / "resnet18_isic2020_training.pt"
RANDOM_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
        torch.save(value, temp_path)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def atomic_write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(value, temp_file, indent=2)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def data_fingerprint(*frames: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(
            pd.util.hash_pandas_object(
                frame[["image_name", "target", "patient_id"]], index=False
            ).values.tobytes()
        )
    return digest.hexdigest()


def capture_rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        np.asarray(numpy_state["state"], dtype=np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    loss_sum = 0.0
    examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * len(labels)
        examples += len(labels)
    return loss_sum / max(examples, 1)


def validate(model, loader, criterion, device) -> tuple[float, dict]:
    model.eval()
    loss_sum = 0.0
    examples = 0
    labels_all = []
    probabilities_all = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            device_labels = labels.to(device, non_blocking=device.type == "cuda")
            logits = model(images)
            loss_sum += float(criterion(logits, device_labels).item()) * len(labels)
            examples += len(labels)
            probabilities_all.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels_all.extend(labels.numpy())

    labels_array = np.asarray(labels_all, dtype=np.int64)
    probabilities_array = np.asarray(probabilities_all, dtype=np.float64)
    predictions = (probabilities_array >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels_array, predictions, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "precision": float(precision_score(labels_array, predictions, zero_division=0)),
        "melanoma_sensitivity_recall": float(recall_score(labels_array, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1_score": float(f1_score(labels_array, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(labels_array, probabilities_array))
            if len(np.unique(labels_array)) == 2 else None
        ),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "positive_examples": int((labels_array == 1).sum()),
        "negative_examples": int((labels_array == 0).sum()),
    }
    return loss_sum / max(examples, 1), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ISIC2020 ResNet18 without extracting archive.zip.")
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--smoke-test", action="store_true", help="Run one short patient-split training epoch.")
    parser.add_argument("--no-pretrained", action="store_true", help="Initialize ResNet18 randomly; no weights download.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest epoch checkpoint.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.cpu_threads < 1:
        parser.error("epochs, batch-size, and cpu-threads must be positive; num-workers cannot be negative.")
    if args.smoke_test:
        args.epochs = 1
        args.max_train_samples = args.max_train_samples or 64
        args.max_validation_samples = args.max_validation_samples or 32

    set_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    if not args.archive.exists():
        raise FileNotFoundError(f"Dataset archive not found: {args.archive}")

    with zipfile.ZipFile(args.archive) as archive:
        metadata_member = find_metadata_member(archive, "isic_2020_train_metadata.csv")
        metadata = load_training_metadata(archive)
        train_frame, validation_frame, test_frame = patient_train_validation_test_split(metadata, args.seed)
        train_frame = limit_partition(train_frame, args.max_train_samples, args.seed)
        validation_frame = limit_partition(validation_frame, args.max_validation_samples, args.seed)
        if train_frame["target"].nunique() < 2 or validation_frame["target"].nunique() < 2:
            raise ValueError("Both patient-level train and validation samples must contain positive and negative labels.")

    patient_overlap = (
        (set(train_frame["patient_id"]) & set(validation_frame["patient_id"]))
        | (set(train_frame["patient_id"]) & set(test_frame["patient_id"]))
        | (set(validation_frame["patient_id"]) & set(test_frame["patient_id"]))
    )
    if patient_overlap:
        raise RuntimeError("Patient leakage detected after sample limiting.")
    split_fingerprint = data_fingerprint(train_frame, validation_frame, test_frame)

    print(json.dumps({
        "archive": str(args.archive),
        "metadata_member": metadata_member,
        "training_records": len(train_frame),
        "validation_records": len(validation_frame),
        "held_out_test_records": len(test_frame),
        "training_patients": int(train_frame["patient_id"].nunique()),
        "validation_patients": int(validation_frame["patient_id"].nunique()),
        "held_out_test_patients": int(test_frame["patient_id"].nunique()),
        "training_label_counts": train_frame["target"].value_counts().sort_index().to_dict(),
        "validation_label_counts": validation_frame["target"].value_counts().sort_index().to_dict(),
        "patient_overlap": len(patient_overlap),
        "pretrained_weights": not args.no_pretrained,
    }, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    train_data = ZipImageDataset(args.archive, train_frame, IMAGE_TRANSFORMS["train"])
    validation_data = ZipImageDataset(args.archive, validation_frame, IMAGE_TRANSFORMS["validation"])
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    counts = train_frame["target"].value_counts().reindex([0, 1], fill_value=0).to_numpy(dtype=np.float32)
    class_weights = counts.sum() / np.maximum(counts * 2, 1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    model = create_resnet18(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    best_loss = float("inf")
    start_epoch = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    resume_path = (
        MODEL_DIR / "resnet18_isic2020_training_smoke.pt"
        if args.smoke_test else RESUME_CHECKPOINT_PATH
    )
    checkpoint_path = (
        MODEL_DIR / "resnet18_isic2020_smoke.pt" if args.smoke_test else CHECKPOINT_PATH
    )
    if args.resume:
        if not resume_path.exists():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {resume_path}")
        resume_state = torch.load(resume_path, map_location=device, weights_only=True)
        expected_training_config = {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_workers": args.num_workers,
            "pretrained_weights": not args.no_pretrained,
            "split_fingerprint": split_fingerprint,
        }
        if resume_state.get("training_config") != expected_training_config:
            raise ValueError("Resume checkpoint does not match the current data split or training settings.")
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        start_epoch = int(resume_state["epoch"])
        best_loss = float(resume_state["best_validation_loss"])
        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint already completed {start_epoch} epochs; --epochs cannot be smaller."
            )
        restore_rng_state(resume_state["rng_state"])

    training_config = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "pretrained_weights": not args.no_pretrained,
        "split_fingerprint": split_fingerprint,
    }
    output_config = (
        MODEL_DIR / "preprocessing.smoke.json" if args.smoke_test else CONFIG_PATH
    )
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, validation_loader, criterion, device)
        print(json.dumps({
            "epoch": epoch + 1,
            "epochs": args.epochs,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_metrics": val_metrics,
        }, indent=2))
        if val_loss < best_loss:
            best_loss = val_loss
            state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            atomic_torch_save({"model_state_dict": state_dict}, checkpoint_path)

        atomic_torch_save({
            "epoch": epoch + 1,
            "best_validation_loss": best_loss,
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": training_config,
            "rng_state": capture_rng_state(),
        }, resume_path)

    config = {
        "model_name": "resnet18_isic2020",
        "architecture": "torchvision.models.resnet18",
        "image_size": list(IMAGE_SIZE),
        "color_space": "RGB",
        "resize_mode": "aspect-preserving centered resize with ImageNet-mean padding",
        "normalization": {
            "mean": [float(value) for value in IMAGENET_MEAN],
            "std": [float(value) for value in IMAGENET_STD],
        },
        "hair_reduction": False,
        "label_encoding": {"0": "non_melanoma", "1": "melanoma"},
        "training_metadata": metadata_member,
        "validation_split": "Nested StratifiedGroupKFold; patient-disjoint train/validation/test folds",
        "validation_patients": sorted(validation_frame["patient_id"].unique().tolist()),
        "test_patients": sorted(test_frame["patient_id"].unique().tolist()),
        "pretrained_weights": not args.no_pretrained,
        "best_validation_loss": best_loss,
        "smoke_test": args.smoke_test,
        "seed": args.seed,
    }
    atomic_write_json(config, output_config)
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "resume_checkpoint": str(resume_path),
        "preprocessing_config": str(output_config),
        "best_validation_loss": best_loss,
        "status": "smoke_test_complete" if args.smoke_test else "training_complete",
    }, indent=2))


if __name__ == "__main__":
    main()
