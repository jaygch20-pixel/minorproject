from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np

from .feature_extraction import extract_features
from .preprocessing import IMAGE_SIZE, image_data_url, mask_data_url, preprocess_image
from .segmentation import segment_lesion

APP_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = APP_DIR.parent
MODEL_DIR = BACKEND_DIR / "model"
CHECKPOINT_PATH = MODEL_DIR / "resnet18_isic2020.pt"
PREPROCESSING_CONFIG_PATH = MODEL_DIR / "preprocessing.json"
METRICS_PATH = (
    BACKEND_DIR
    / "results"
    / "held_out_evaluation_20261028"
    / "cnn_evaluation_metrics.json"
)
VALIDATION_METRICS_PATH = BACKEND_DIR / "results" / "validation_evaluation" / "validation_metrics.json"
LOGGER = logging.getLogger(__name__)


def map_class_probabilities(
    probabilities: np.ndarray,
    label_encoding: Mapping[str, str],
) -> dict[str, float]:
    if probabilities.shape != (2,) or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite probabilities for exactly two model output classes.")
    if (
        np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(float(probabilities.sum()), 1.0, atol=1e-5)
    ):
        raise ValueError("Model class probabilities must be non-negative and sum to one.")
    if (
        set(label_encoding) != {"0", "1"}
        or set(label_encoding.values()) != {"non_melanoma", "melanoma"}
    ):
        raise ValueError("Model label encoding must map indices 0 and 1 to the ISIC2020 classes.")
    return {
        label_encoding[str(index)]: float(probabilities[index])
        for index in range(2)
    }


def model_status() -> dict:
    if not CHECKPOINT_PATH.exists() or not PREPROCESSING_CONFIG_PATH.exists():
        return {"available": False, "model_name": "resnet18_isic2020", "status": "unavailable"}
    config = json.loads(PREPROCESSING_CONFIG_PATH.read_text(encoding="utf-8"))
    return {"available": True, "model_name": config["model_name"], "status": "trained"}


def evaluation_metrics() -> dict | None:
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        LOGGER.warning("Evaluation metrics file contains invalid JSON: %s", METRICS_PATH)
        return None

    if not isinstance(metrics, dict):
        LOGGER.warning("Evaluation metrics file must contain a JSON object: %s", METRICS_PATH)
        return None
    metrics["validation_evaluation"] = validation_evaluation_metrics()
    return metrics


def validation_evaluation_metrics() -> dict | None:
    try:
        metrics = json.loads(VALIDATION_METRICS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        LOGGER.warning("Validation metrics file contains invalid JSON: %s", VALIDATION_METRICS_PATH)
        return None

    if not isinstance(metrics, dict):
        LOGGER.warning("Validation metrics file must contain a JSON object: %s", VALIDATION_METRICS_PATH)
        return None
    return metrics


@lru_cache(maxsize=1)
def _load_classifier():
    import torch

    from .cnn import create_resnet18

    config = json.loads(PREPROCESSING_CONFIG_PATH.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_resnet18(pretrained=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device, config


def analyze_skin_image(image_bytes: bytes) -> dict:
    preprocessing = preprocess_image(image_bytes, IMAGE_SIZE, hair_reduction=False)
    rgb = preprocessing["preprocessed_rgb"]
    segmentation = segment_lesion(rgb)
    features = extract_features(rgb, segmentation)

    result = {
        "mode": "cnn" if CHECKPOINT_PATH.exists() and PREPROCESSING_CONFIG_PATH.exists() else "unavailable",
        "status": "model_unavailable",
        "model_available": False,
        "prediction": None,
        "confidence": None,
        "probabilities": {"melanoma": None, "non_melanoma": None},
        "preprocessing": {
            "status": preprocessing["status"],
            "input_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "normalization": "ImageNet mean/std RGB",
            "hair_reduction": preprocessing["hair_reduction"],
            "cnn_input_uses_mask": False,
            "original_size": [
                int(preprocessing["resize_layout"]["original_width"]),
                int(preprocessing["resize_layout"]["original_height"]),
            ],
            "resize_layout": preprocessing["resize_layout"],
            "image_data": image_data_url(rgb),
        },
        "segmentation": {
            "status": segmentation["status"],
            "method": segmentation["method"],
            "is_usable": segmentation["is_usable"],
            "area_fraction": segmentation["area_fraction"],
            "foreground_fraction": segmentation["foreground_fraction"],
            "foreground_pixels": segmentation["foreground_pixels"],
            "mask_min": segmentation["mask_min"],
            "mask_max": segmentation["mask_max"],
            "contour_count": segmentation["contour_count"],
            "contour_points": segmentation["contour_points"],
            "message": segmentation["message"],
            "mask_size": [
                int(segmentation["mask"].shape[1]),
                int(segmentation["mask"].shape[0]),
            ],
            "mask_image_data": (
                mask_data_url(segmentation["mask"])
                if segmentation["is_usable"] else None
            ),
            "segmented_image_data": (
                image_data_url(segmentation["segmented_rgb"])
                if segmentation["segmented_rgb"] is not None else None
            ),
            "boundary_overlay_image_data": (
                image_data_url(segmentation["boundary_overlay_rgb"])
                if segmentation["boundary_overlay_rgb"] is not None else None
            ),
        },
        "features": features,
        "evaluation_metrics": evaluation_metrics(),
        "warning": "Research prototype only. This output is not a medical diagnosis or a clinical decision.",
    }

    if not CHECKPOINT_PATH.exists() or not PREPROCESSING_CONFIG_PATH.exists():
        result["warning"] = (
            "No trained ISIC2020 CNN is available. No prediction was generated. "
            "This research prototype is not a medical diagnostic system."
        )
        return result

    import torch

    model, device, config = _load_classifier()
    expected_size = tuple(config["image_size"])
    if tuple(preprocessing["preprocessed_rgb"].shape[1::-1]) != expected_size:
        raise ValueError(
            f"Preprocessing produced {preprocessing['preprocessed_rgb'].shape[1::-1]}, "
            f"but the trained model expects {expected_size}."
        )
    tensor = torch.from_numpy(np.asarray(preprocessing["model_input"])).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    label_map = config["label_encoding"]
    probabilities_by_label = map_class_probabilities(probabilities, label_map)
    predicted_label = label_map[str(predicted_index)]
    result.update({
        "mode": "cnn",
        "status": "success",
        "model_available": True,
        "prediction": predicted_label,
        "confidence": probabilities_by_label[predicted_label],
        "probabilities": probabilities_by_label,
        "model_name": config["model_name"],
    })
    return result
