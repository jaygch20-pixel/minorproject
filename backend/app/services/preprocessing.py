from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def decode_image(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("RGB"))


def reduce_hair_artifacts(image_rgb: np.ndarray) -> np.ndarray:
    """Apply a conservative black-hat hair mask and inpaint only detected dark strands."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return cv2.cvtColor(cv2.inpaint(image_bgr, hair_mask, 3, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)


def normalize_rgb(image_rgb: np.ndarray) -> np.ndarray:
    values = image_rgb.astype(np.float32) / 255.0
    return (values - IMAGENET_MEAN) / IMAGENET_STD


def resize_rgb(image_rgb: np.ndarray, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    target_width, target_height = image_size
    height, width = image_rgb.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = min(target_width, max(1, round(width * scale)))
    resized_height = min(target_height, max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=interpolation)

    horizontal_padding = target_width - resized_width
    vertical_padding = target_height - resized_height
    left = horizontal_padding // 2
    right = horizontal_padding - left
    top = vertical_padding // 2
    bottom = vertical_padding - top
    if horizontal_padding or vertical_padding:
        resized = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(124, 116, 104),
        )
    return resized


def resize_layout(
    image_rgb: np.ndarray,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> dict[str, int]:
    target_width, target_height = image_size
    height, width = image_rgb.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = min(target_width, max(1, round(width * scale)))
    resized_height = min(target_height, max(1, round(height * scale)))
    horizontal_padding = target_width - resized_width
    vertical_padding = target_height - resized_height
    return {
        "original_width": int(width),
        "original_height": int(height),
        "target_width": int(target_width),
        "target_height": int(target_height),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "left": int(horizontal_padding // 2),
        "top": int(vertical_padding // 2),
        "right": int(horizontal_padding - horizontal_padding // 2),
        "bottom": int(vertical_padding - vertical_padding // 2),
    }


def preprocess_image(
    image_bytes: bytes,
    image_size: tuple[int, int] = IMAGE_SIZE,
    hair_reduction: bool = False,
) -> dict[str, np.ndarray | str]:
    original_rgb = decode_image(image_bytes)
    source_rgb = reduce_hair_artifacts(original_rgb) if hair_reduction else original_rgb.copy()
    resized_rgb = resize_rgb(source_rgb, image_size)
    normalized = normalize_rgb(resized_rgb)
    tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False)
    return {
        "original_rgb": original_rgb,
        "preprocessed_rgb": resized_rgb,
        "model_input": tensor,
        "status": "ok",
        "hair_reduction": "applied" if hair_reduction else "disabled",
        "resize_layout": resize_layout(original_rgb, image_size),
    }


def image_data_url(image_rgb: np.ndarray, image_format: str = "PNG") -> str:
    image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime_type = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime_type};base64,{encoded}"


def mask_data_url(mask: np.ndarray) -> str:
    image = Image.fromarray(mask.astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"
