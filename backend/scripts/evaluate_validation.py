from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR / "scripts"))
sys.path.insert(0, str(BACKEND_DIR))

from app.services.cnn import create_resnet18
from app.services.dataset import find_metadata_member
from app.services.training_data import (
    IMAGE_TRANSFORMS,
    ZipImageDataset,
    load_training_metadata,
    patient_train_validation_test_split,
)
from evaluate_model import compute_metrics
from train_model import data_fingerprint

ARCHIVE_PATH = BACKEND_DIR.parent / "archive.zip"
MODEL_DIR = BACKEND_DIR / "model"
CHECKPOINT_PATH = MODEL_DIR / "resnet18_isic2020.pt"
CONFIG_PATH = MODEL_DIR / "preprocessing.json"
RESUME_CHECKPOINT_PATH = MODEL_DIR / "resnet18_isic2020_training.pt"
OUTPUT_DIR = BACKEND_DIR / "results" / "validation_evaluation"
EXPECTED_LABEL_ENCODING = {"0": "non_melanoma", "1": "melanoma"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probability_distribution(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "minimum": float(values.min()),
        "percentile_10": float(np.percentile(values, 10)),
        "percentile_25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "percentile_75": float(np.percentile(values, 75)),
        "percentile_90": float(np.percentile(values, 90)),
        "maximum": float(values.max()),
        "at_or_above_0_50": int(np.count_nonzero(values >= 0.5)),
        "below_0_50": int(np.count_nonzero(values < 0.5)),
    }


def weighted_loss_for_batch_size(
    targets: np.ndarray,
    melanoma_scores: np.ndarray,
    class_weights: np.ndarray,
    batch_size: int,
) -> float:
    true_class_scores = np.where(targets == 1, melanoma_scores, 1.0 - melanoma_scores)
    sample_losses = -np.log(np.clip(true_class_scores, np.finfo(np.float64).tiny, 1.0))
    sample_weights = class_weights[targets]
    weighted_loss_sum = 0.0
    for start in range(0, targets.size, batch_size):
        batch_end = start + batch_size
        batch_weights = sample_weights[start:batch_end]
        batch_losses = sample_losses[start:batch_end]
        weighted_batch_mean = float(np.sum(batch_losses * batch_weights) / np.sum(batch_weights))
        weighted_loss_sum += weighted_batch_mean * len(batch_losses)
    return weighted_loss_sum / max(targets.size, 1)


def validation_threshold_analysis(
    targets: np.ndarray,
    melanoma_scores: np.ndarray,
    sensitivity_objectives: tuple[float, ...] = (0.90, 0.95),
) -> dict:
    analyses = {}
    for objective in sensitivity_objectives:
        eligible_thresholds = [
            float(threshold)
            for threshold in np.unique(melanoma_scores)
            if recall_score(
                targets,
                melanoma_scores >= threshold,
                zero_division=0,
            ) >= objective
        ]
        threshold = max(eligible_thresholds)
        predictions = melanoma_scores >= threshold
        tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
        analyses[f"{objective:.0%}_target_sensitivity"] = {
            "threshold": threshold,
            "sensitivity": float(recall_score(targets, predictions, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "precision": float(precision_score(targets, predictions, zero_division=0)),
            "f1_score": float(f1_score(targets, predictions, zero_division=0)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }
    return analyses


def fixed_threshold_analysis(
    targets: np.ndarray,
    melanoma_scores: np.ndarray,
    thresholds: tuple[float, ...] = (0.50, 0.30, 0.20, 0.10, 0.05, 0.03, 0.02, 0.01),
) -> list[dict]:
    analyses = []
    for threshold in thresholds:
        predictions = melanoma_scores >= threshold
        tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
        analyses.append({
            "threshold": threshold,
            "sensitivity": float(recall_score(targets, predictions, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "precision": float(precision_score(targets, predictions, zero_division=0)),
            "f1_score": float(f1_score(targets, predictions, zero_division=0)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        })
    return analyses


def validate_metadata_labels(metadata) -> dict:
    if "benign_malignant" not in metadata:
        raise ValueError("Training metadata has no benign_malignant column to verify target labels.")
    expected = metadata["benign_malignant"].str.lower().map({"benign": 0, "malignant": 1})
    missing = int(expected.isna().sum())
    mismatches = int((metadata["target"] != expected).sum())
    if missing or mismatches:
        raise ValueError(
            f"Metadata label consistency check failed: {missing} unknown labels and {mismatches} target conflicts."
        )
    conflicts = int(metadata.groupby("image_name")["target"].nunique().gt(1).sum())
    duplicates = int(metadata["image_name"].duplicated().sum())
    if conflicts or duplicates:
        raise ValueError(
            f"Training metadata contains {duplicates} duplicate image IDs and {conflicts} conflicting labels."
        )
    return {
        "target_matches_benign_malignant": True,
        "target_conflicts": mismatches,
        "duplicate_image_ids": duplicates,
        "conflicting_duplicate_labels": conflicts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the unchanged ResNet18 on its patient-disjoint labeled validation fold."
    )
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch-size and cpu-threads must be positive.")
    if not args.archive.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {args.archive}")
    if not CHECKPOINT_PATH.is_file() or not CONFIG_PATH.is_file():
        raise FileNotFoundError("The production checkpoint or its preprocessing configuration is missing.")

    torch.set_num_threads(args.cpu_threads)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("label_encoding") != EXPECTED_LABEL_ENCODING:
        raise ValueError("The production label encoding does not match ISIC2020 target indices.")
    if config.get("smoke_test") is not False:
        raise ValueError("The configured production model is marked as a smoke-test checkpoint.")

    with zipfile.ZipFile(args.archive) as archive:
        metadata_member = find_metadata_member(archive, "isic_2020_train_metadata.csv")
        metadata = load_training_metadata(archive)
        train_frame, validation_frame, test_frame = patient_train_validation_test_split(
            metadata,
            int(config["seed"]),
        )

    expected_validation_patients = set(config["validation_patients"])
    expected_test_patients = set(config["test_patients"])
    if set(validation_frame["patient_id"]) != expected_validation_patients:
        raise ValueError("Saved validation patients do not match the current patient-level data split.")
    if set(test_frame["patient_id"]) != expected_test_patients:
        raise ValueError("Saved held-out test patients do not match the current patient-level data split.")
    if (
        set(train_frame["patient_id"]) & expected_validation_patients
        or set(train_frame["patient_id"]) & expected_test_patients
        or expected_validation_patients & expected_test_patients
    ):
        raise ValueError("Patient overlap detected between training, validation, and held-out test folds.")
    label_integrity = validate_metadata_labels(metadata)
    split_fingerprint = data_fingerprint(train_frame, validation_frame, test_frame)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_resnet18(pretrained=False).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_counts = train_frame["target"].value_counts().reindex([0, 1], fill_value=0)
    class_weights = train_counts.sum() / (2 * train_counts.to_numpy(dtype=np.float32))
    weighted_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
    )
    unweighted_criterion = nn.CrossEntropyLoss(reduction="sum")
    dataset = ZipImageDataset(
        args.archive,
        validation_frame,
        IMAGE_TRANSFORMS["validation"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    targets: list[int] = []
    melanoma_scores: list[float] = []
    weighted_loss_sum = 0.0
    unweighted_loss_sum = 0.0
    examples = 0
    with torch.inference_mode():
        for batch_index, (images, batch_targets) in enumerate(loader, start=1):
            logits = model(images.to(device))
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(f"Expected two ResNet18 output logits; received shape {tuple(logits.shape)}.")
            weighted_loss_sum += float(weighted_criterion(logits, batch_targets.to(device)).item()) * len(batch_targets)
            unweighted_loss_sum += float(unweighted_criterion(logits, batch_targets.to(device)).item())
            batch_probabilities = torch.softmax(logits, dim=1)
            melanoma_scores.extend(batch_probabilities[:, 1].cpu().tolist())
            targets.extend(batch_targets.tolist())
            examples += len(batch_targets)
            if batch_index % 100 == 0:
                print(f"Evaluated {examples}/{len(dataset)} validation images.", flush=True)

    target_array = np.asarray(targets, dtype=np.int64)
    score_array = np.asarray(melanoma_scores, dtype=np.float64)
    probability_matrix = np.column_stack((1.0 - score_array, score_array))
    metrics = compute_metrics(target_array, probability_matrix)
    distributions = {
        "non_melanoma": probability_distribution(score_array[target_array == 0]),
        "melanoma": probability_distribution(score_array[target_array == 1]),
    }
    predictions_path = args.output_dir / "validation_predictions.csv"
    metrics_path = args.output_dir / "validation_metrics.json"

    resume_provenance = {
        "available": RESUME_CHECKPOINT_PATH.is_file(),
        "training_config_matches_current_split": False,
        "production_weights_match_latest_resume_checkpoint": False,
        "latest_resume_epoch": None,
    }
    if RESUME_CHECKPOINT_PATH.is_file():
        resume_state = torch.load(RESUME_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        resume_config = resume_state.get("training_config", {})
        resume_provenance.update({
            "training_config_matches_current_split": (
                resume_config.get("split_fingerprint") == split_fingerprint
                and resume_config.get("seed") == config["seed"]
                and resume_config.get("pretrained_weights") == config.get("pretrained_weights")
            ),
            "production_weights_match_latest_resume_checkpoint": all(
                torch.equal(
                    checkpoint["model_state_dict"][name].cpu(),
                    resume_state["model_state_dict"][name].cpu(),
                )
                for name in checkpoint["model_state_dict"]
            ),
            "latest_resume_epoch": int(resume_state["epoch"]),
            "best_validation_loss": float(resume_state["best_validation_loss"]),
            "training_config": {
                key: value
                for key, value in resume_config.items()
                if key != "split_fingerprint"
            },
        })

    training_batch_size = int(
        resume_provenance["training_config"].get("batch_size", args.batch_size)
    )
    reproduced_weighted_loss = weighted_loss_for_batch_size(
        target_array,
        score_array,
        class_weights,
        training_batch_size,
    )
    recorded_best_validation_loss = config.get("best_validation_loss")

    report = {
        "model_name": config["model_name"],
        "evaluation_source": "patient-disjoint validation fold from labeled ISIC2020 training metadata",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "filename": CHECKPOINT_PATH.name,
            "sha256": file_sha256(CHECKPOINT_PATH),
            "architecture": config["architecture"],
            "pretrained_weights": config["pretrained_weights"],
            "smoke_test": config["smoke_test"],
            "production_best_validation_loss": config.get("best_validation_loss"),
            **resume_provenance,
        },
        "split": {
            "metadata_member": metadata_member,
            "seed": config["seed"],
            "training_examples": len(train_frame),
            "validation_examples": len(validation_frame),
            "validation_patients": len(expected_validation_patients),
            "training_patients_excluded": True,
            "held_out_test_patients_excluded": True,
            "patient_overlap": False,
            "training_class_counts": {
                "non_melanoma": int(train_counts.loc[0]),
                "melanoma": int(train_counts.loc[1]),
            },
            "validation_class_counts": {
                "non_melanoma": int(np.count_nonzero(target_array == 0)),
                "melanoma": int(np.count_nonzero(target_array == 1)),
            },
            "label_integrity": label_integrity,
        },
        "label_encoding": config["label_encoding"],
        "inference": {
            "output_type": "two logits converted to class probabilities using softmax",
            "image_size": config["image_size"],
            "color_space": config["color_space"],
            "resize_mode": config["resize_mode"],
            "normalization": config["normalization"],
            "hair_reduction": config["hair_reduction"],
            "threshold": 0.5,
        },
        "validation_loss": {
            "class_weighted_cross_entropy": reproduced_weighted_loss,
            "class_weighted_cross_entropy_at_evaluation_batch_size": weighted_loss_sum / max(examples, 1),
            "unweighted_cross_entropy": unweighted_loss_sum / max(examples, 1),
            "training_batch_size": training_batch_size,
            "evaluation_batch_size": args.batch_size,
            "matches_recorded_best_validation_loss": (
                bool(np.isclose(
                    reproduced_weighted_loss,
                    recorded_best_validation_loss,
                    rtol=1e-4,
                    atol=1e-5,
                ))
                if recorded_best_validation_loss is not None else None
            ),
            "training_loss_history_available": False,
            "note": (
                "The class-weighted loss is reproduced at the recorded training batch size; "
                "the production run saved its best validation loss but not per-epoch training-loss history. "
                "Weighted mean loss can vary with batch composition."
            ),
        },
        "melanoma_probability_by_true_class": distributions,
        "fixed_threshold_analysis": {
            "thresholds": fixed_threshold_analysis(target_array, score_array),
            "note": (
                "These cutoffs are descriptive validation results. Only the separately "
                "reported sensitivity-objective threshold is selected on validation; none "
                "is applied to production predictions."
            ),
        },
        "validation_only_threshold_analysis": {
            "sensitivity_objectives": [0.90, 0.95],
            "thresholds": validation_threshold_analysis(target_array, score_array),
            "note": (
                "Exploratory thresholds are selected only on this validation fold to illustrate "
                "sensitivity/specificity tradeoffs. They are not applied to production predictions "
                "and are not clinical recommendations."
            ),
        },
        "pr_auc_average_precision": float(average_precision_score(target_array, score_array)),
        "pr_auc_definition": "Average precision (non-interpolated area under the precision-recall curve)",
        "predictions_file": predictions_path.name,
        **metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "image_id",
                "actual_class",
                "predicted_class",
                "p_non_melanoma",
                "p_melanoma",
                "threshold",
            ),
        )
        writer.writeheader()
        for image_id, actual, melanoma_probability in zip(
            validation_frame["image_name"],
            target_array,
            score_array,
            strict=True,
        ):
            actual_label = config["label_encoding"][str(int(actual))]
            predicted_index = int(melanoma_probability >= 0.5)
            predicted_label = config["label_encoding"][str(predicted_index)]
            writer.writerow({
                "image_id": image_id,
                "actual_class": actual_label,
                "predicted_class": predicted_label,
                "p_non_melanoma": f"{1.0 - melanoma_probability:.10f}",
                "p_melanoma": f"{melanoma_probability:.10f}",
                "threshold": "0.5",
            })
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Validation predictions: {predictions_path}")
    print(f"Validation report: {metrics_path}")


if __name__ == "__main__":
    main()
