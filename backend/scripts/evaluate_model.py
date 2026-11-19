from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.cnn import create_resnet18
from app.services.dataset import find_metadata_member
from app.services.training_data import (
    IMAGE_TRANSFORMS,
    ZipImageDataset,
    limit_partition,
    load_training_metadata,
    patient_train_validation_test_split,
)

ARCHIVE_PATH = BACKEND_DIR.parent / "archive.zip"
CHECKPOINT_PATH = BACKEND_DIR / "model" / "resnet18_isic2020.pt"
CONFIG_PATH = BACKEND_DIR / "model" / "preprocessing.json"
METRICS_PATH = BACKEND_DIR / "results" / "cnn_evaluation_metrics.json"


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities[:, 1] >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "melanoma_sensitivity_recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities[:, 1])) if len(np.unique(labels)) == 2 else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "positive_examples": int((labels == 1).sum()),
        "negative_examples": int((labels == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained CNN on its untouched patient-level test split.")
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path, default=METRICS_PATH)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch-size and cpu-threads must be positive.")
    if not CHECKPOINT_PATH.exists() or not CONFIG_PATH.exists():
        raise FileNotFoundError("Train the ResNet18 first; its checkpoint or preprocessing config is missing.")

    torch.set_num_threads(args.cpu_threads)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    with zipfile.ZipFile(args.archive) as archive:
        metadata_member = find_metadata_member(archive, "isic_2020_train_metadata.csv")
        metadata = load_training_metadata(archive)
        _, _, test_frame = patient_train_validation_test_split(metadata, int(config["seed"]))
    expected_patients = set(config["test_patients"])
    test_frame = test_frame[test_frame["patient_id"].isin(expected_patients)].copy()
    if set(test_frame["patient_id"]) != expected_patients:
        raise ValueError("The saved held-out test patient IDs do not match the current training metadata.")
    test_frame = limit_partition(test_frame, args.max_samples, int(config["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_resnet18(pretrained=False).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = ZipImageDataset(args.archive, test_frame, IMAGE_TRANSFORMS["validation"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    labels = []
    probabilities = []
    with torch.inference_mode():
        for images, batch_labels in loader:
            logits = model(images.to(device))
            probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
            labels.extend(batch_labels.numpy())
    metrics = compute_metrics(np.asarray(labels), np.asarray(probabilities))
    report = {
        "model_name": config["model_name"],
        "evaluation_source": "untouched patient-disjoint test fold from labeled training metadata",
        "metadata_member": metadata_member,
        "test_patient_count": len(expected_patients),
        "training_patients_excluded": True,
        "official_test_metadata_has_labels": False,
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
