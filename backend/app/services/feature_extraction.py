from __future__ import annotations

import cv2
import numpy as np


def extract_features(image_rgb: np.ndarray, segmentation: dict) -> dict:
    if not segmentation.get("is_usable"):
        return {
            "status": "unavailable",
            "reason": "Lesion-specific measurements were skipped because segmentation was not usable.",
            "traditional_features": None,
        }

    mask = segmentation["mask"]
    area = float(np.count_nonzero(mask))
    if area == 0:
        return {
            "status": "unavailable",
            "reason": "The segmentation mask contains no lesion pixels.",
            "traditional_features": None,
        }

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "status": "unavailable",
            "reason": "No lesion contour could be measured.",
            "traditional_features": None,
        }

    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, True))
    contour_area = float(cv2.contourArea(contour))
    x, y, width, height = cv2.boundingRect(contour)
    circularity = float((4.0 * np.pi * contour_area) / (perimeter * perimeter)) if perimeter else None
    roi = image_rgb[mask > 0].astype(np.float32)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    texture = _texture_features(gray, mask)

    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        covariance = np.asarray([
            [moments["mu20"] / moments["m00"], moments["mu11"] / moments["m00"]],
            [moments["mu11"] / moments["m00"], moments["mu02"] / moments["m00"]],
        ])
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
        eccentricity = float(np.sqrt(1 - eigenvalues[0] / eigenvalues[1])) if eigenvalues[1] else 0.0
    else:
        eccentricity = None

    return {
        "status": "available_approximate",
        "reason": "Measurements are approximate and depend on an unverified OpenCV mask.",
        "traditional_features": {
            "lesion_area_px": area,
            "lesion_area_fraction": float(area / (mask.shape[0] * mask.shape[1])),
            "lesion_perimeter_px": perimeter,
            "circularity": circularity,
            "bounding_box_aspect_ratio": float(width / height) if height else None,
            "extent": float(area / (width * height)) if width and height else None,
            "eccentricity": eccentricity,
            "mean_rgb": [float(value) for value in roi.mean(axis=0)],
            "rgb_standard_deviation": [float(value) for value in roi.std(axis=0)],
            **texture,
        },
    }


def _texture_features(gray: np.ndarray, mask: np.ndarray) -> dict:
    quantized = (gray.astype(np.uint16) // 16).astype(np.uint8)
    valid = (mask[:, :-1] > 0) & (mask[:, 1:] > 0)
    left = quantized[:, :-1][valid].astype(np.int16)
    right = quantized[:, 1:][valid].astype(np.int16)
    if left.size == 0:
        return {"texture_contrast": None, "texture_homogeneity": None}

    difference = np.abs(left - right).astype(np.float32)
    return {
        "texture_contrast": float(np.mean(difference**2)),
        "texture_homogeneity": float(np.mean(1.0 / (1.0 + difference**2))),
    }
