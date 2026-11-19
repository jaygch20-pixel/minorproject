from __future__ import annotations

import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
MIN_FOREGROUND_FRACTION = 0.002
MAX_FOREGROUND_FRACTION = 0.90


def segment_lesion(image_rgb: np.ndarray) -> dict:
    """Create an approximate OpenCV mask; this is not a medical annotation."""
    height, width = image_rgb.shape[:2]
    if (
        image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.dtype != np.uint8
        or height == 0
        or width == 0
    ):
        result = _failed_result(
            width,
            height,
            "Expected a three-channel uint8 RGB image.",
            "input_validation",
        )
        _log_result(image_rgb, result)
        return result

    x0, y0, x1, y1 = _content_bounds(image_rgb)
    content_rgb = image_rgb[y0:y1, x0:x1]
    content_height, content_width = content_rgb.shape[:2]
    if min(content_height, content_width) < 16:
        result = _failed_result(
            width,
            height,
            "Image is too small for the segmentation baseline.",
            "input_validation",
        )
        _log_result(image_rgb, result)
        return result
    if float(np.std(content_rgb.astype(np.float32), axis=(0, 1)).max()) < 2.0:
        result = _failed_result(
            width,
            height,
            "Image has insufficient colour contrast for approximate segmentation.",
            "input_validation",
        )
        _log_result(image_rgb, result)
        return result

    image_bgr = cv2.cvtColor(content_rgb, cv2.COLOR_RGB2BGR)
    margin_x = max(1, int(content_width * 0.05))
    margin_y = max(1, int(content_height * 0.05))
    rect = (margin_x, margin_y, content_width - 2 * margin_x, content_height - 2 * margin_y)
    failures: list[str] = []
    method = "opencv_grabcut"

    grabcut_mask = np.zeros((content_height, content_width), dtype=np.uint8)
    try:
        cv2.grabCut(image_bgr, grabcut_mask, rect, None, None, 3, cv2.GC_INIT_WITH_RECT)
        candidate = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
        candidate = _clean_mask(candidate, (5, 5))
        issue = _candidate_issue(candidate)
        _log_candidate(method, candidate, issue)
        if issue is None:
            return _success_result(image_rgb, candidate, (x0, y0, x1, y1), method)
        failures.append(f"GrabCut rejected: {issue}")
    except cv2.error as error:
        failures.append("GrabCut failed with an OpenCV error.")
        LOGGER.warning("OpenCV GrabCut failed for segmentation input: %s", error)

    method = "opencv_lab_border_contrast"
    try:
        candidate = _threshold_lab_contrast(content_rgb)
        candidate = _clean_mask(candidate, (3, 3))
        issue = _candidate_issue(candidate)
        _log_candidate(method, candidate, issue)
        if issue is None:
            return _success_result(image_rgb, candidate, (x0, y0, x1, y1), method)
        failures.append(f"Lab border-colour threshold rejected: {issue}")
    except cv2.error as error:
        failures.append("Lab border-colour threshold failed with an OpenCV error.")
        LOGGER.warning("OpenCV Lab threshold failed for segmentation input: %s", error)

    reason = " ".join(failures) or "No segmentation candidate was produced."
    result = _failed_result(
        width,
        height,
        reason,
        "opencv_grabcut+opencv_lab_border_contrast",
    )
    _log_result(image_rgb, result)
    return result


def _success_result(
    image_rgb: np.ndarray,
    content_mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    method: str,
) -> dict:
    height, width = image_rgb.shape[:2]
    x0, y0, x1, y1 = bounds
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = content_mask
    segmented_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=full_mask)
    overlay_rgb = image_rgb.copy()
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_rgb, contours, -1, (20, 220, 120), 2)
    contour_points = [
        [[int(point[0][0]), int(point[0][1])] for point in contour]
        for contour in contours
    ]

    foreground_pixels = int(np.count_nonzero(full_mask))
    foreground_fraction = foreground_pixels / float(width * height)
    result = {
        "status": "approximate",
        "method": method,
        "is_usable": True,
        "mask": full_mask,
        "segmented_rgb": segmented_rgb,
        "boundary_overlay_rgb": overlay_rgb,
        "area_fraction": foreground_fraction,
        "foreground_fraction": foreground_fraction,
        "foreground_pixels": foreground_pixels,
        "mask_min": int(full_mask.min()),
        "mask_max": int(full_mask.max()),
        "contour_count": len(contours),
        "contour_points": contour_points,
        "message": "Approximate computer-vision mask only; not a medically verified lesion boundary.",
    }
    _log_result(image_rgb, result)
    return result


def _content_bounds(image_rgb: np.ndarray) -> tuple[int, int, int, int]:
    padding_rgb = np.asarray((124, 116, 104), dtype=np.uint8)
    content = np.any(image_rgb != padding_rgb, axis=2)
    ys, xs = np.where(content)
    if not len(xs):
        height, width = image_rgb.shape[:2]
        return 0, 0, width, height
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _threshold_lab_contrast(image_rgb: np.ndarray) -> np.ndarray:
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    height, width = image_lab.shape[:2]
    ring = max(2, int(min(height, width) * 0.08))
    border = np.concatenate(
        (
            image_lab[:ring, :, :].reshape(-1, 3),
            image_lab[-ring:, :, :].reshape(-1, 3),
            image_lab[:, :ring, :].reshape(-1, 3),
            image_lab[:, -ring:, :].reshape(-1, 3),
        ),
        axis=0,
    )
    border_colour = np.median(border, axis=0).astype(np.float32)
    colour_distance = np.linalg.norm(image_lab.astype(np.float32) - border_colour, axis=2)
    distance_u8 = np.clip(colour_distance, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(distance_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask


def _clean_mask(mask: np.ndarray, kernel_size: tuple[int, int]) -> np.ndarray:
    mask = _largest_center_component(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)


def _largest_center_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask

    center = np.asarray((mask.shape[1] / 2, mask.shape[0] / 2))
    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        distance = float(np.linalg.norm(centroids[label] - center))
        candidates.append((area / (1 + distance / max(mask.shape)), label))
    chosen = max(candidates)[1]
    return np.where(labels == chosen, 255, 0).astype(np.uint8)


def _candidate_issue(mask: np.ndarray) -> str | None:
    foreground_pixels = int(np.count_nonzero(mask))
    fraction = foreground_pixels / float(mask.size)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if foreground_pixels == 0 or fraction < MIN_FOREGROUND_FRACTION:
        return f"foreground area is too small ({fraction:.3%})."
    if fraction >= MAX_FOREGROUND_FRACTION:
        return f"foreground area is nearly the entire content region ({fraction:.3%})."
    if not contours:
        return "no external contour was detected."
    if _touches_border(mask):
        return "foreground mask touches the content boundary."
    return None


def _touches_border(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def _log_candidate(method: str, mask: np.ndarray, issue: str | None) -> None:
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    foreground_pixels = int(np.count_nonzero(mask))
    LOGGER.debug(
        "Segmentation candidate: method=%s mask_min=%d mask_max=%d "
        "foreground_pixels=%d foreground_percent=%.3f contours=%d accepted=%s reason=%s",
        method,
        int(mask.min()),
        int(mask.max()),
        foreground_pixels,
        foreground_pixels / float(mask.size) * 100,
        len(contours),
        issue is None,
        issue or "none",
    )


def _log_result(image_rgb: np.ndarray, result: dict) -> None:
    mask = result["mask"]
    foreground_pixels = int(np.count_nonzero(mask))
    mask_min = int(mask.min()) if mask.size else 0
    mask_max = int(mask.max()) if mask.size else 0
    foreground_percent = foreground_pixels / float(mask.size) * 100 if mask.size else 0.0
    image_range = (
        (int(image_rgb.min()), int(image_rgb.max()))
        if image_rgb.size
        else (0, 0)
    )
    log_method = LOGGER.info if result["is_usable"] else LOGGER.warning
    log_method(
        "Segmentation result: method=%s status=%s image_shape=%s image_dtype=%s "
        "image_range=[%d,%d] input_channels=RGB opencv_input_channels=BGR "
        "mask_shape=%s mask_dtype=%s mask_min=%d mask_max=%d foreground_pixels=%d "
        "foreground_percent=%.3f contours=%d reason=%s",
        result["method"],
        result["status"],
        image_rgb.shape,
        image_rgb.dtype,
        image_range[0],
        image_range[1],
        mask.shape,
        mask.dtype,
        mask_min,
        mask_max,
        foreground_pixels,
        foreground_percent,
        result["contour_count"],
        "none" if result["is_usable"] else result["message"],
    )


def _failed_result(width: int, height: int, message: str, method: str) -> dict:
    mask = np.zeros((height, width), dtype=np.uint8)
    return {
        "status": "failed",
        "method": method,
        "is_usable": False,
        "mask": mask,
        "segmented_rgb": None,
        "boundary_overlay_rgb": None,
        "area_fraction": None,
        "foreground_fraction": 0.0,
        "foreground_pixels": 0,
        "mask_min": 0,
        "mask_max": 0,
        "contour_count": 0,
        "contour_points": [],
        "message": message,
    }
