from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.dataset import find_image_members, find_metadata_member, read_csv_from_archive
from app.services.feature_extraction import extract_features
from app.services.preprocessing import IMAGE_SIZE, preprocess_image
from app.services.segmentation import segment_lesion

ARCHIVE_PATH = BACKEND_DIR.parent / "archive.zip"
OUTPUT_DIR = BACKEND_DIR / "results" / "cv_pipeline_verification"
FEATURE_FIELDS = (
    "lesion_area_px",
    "lesion_area_fraction",
    "lesion_perimeter_px",
    "circularity",
    "bounding_box_aspect_ratio",
    "extent",
    "eccentricity",
    "mean_rgb",
    "rgb_standard_deviation",
    "texture_contrast",
    "texture_homogeneity",
)
CELL_SIZE = 256
GAP = 12


def make_panel(title: str, image: np.ndarray | None) -> Image.Image:
    panel = Image.new("RGB", (CELL_SIZE, CELL_SIZE + 30), "#f1f7f7")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), title, fill="#13253f")
    if image is None:
        draw.rectangle((0, 30, CELL_SIZE, CELL_SIZE + 30), fill="#dfe8e8")
        draw.text((12, 140), "Not available", fill="#596a70")
        return panel

    thumbnail = Image.fromarray(image.astype("uint8")).convert("RGB")
    thumbnail.thumbnail((CELL_SIZE, CELL_SIZE), Image.Resampling.LANCZOS)
    left = (CELL_SIZE - thumbnail.width) // 2
    top = 30 + (CELL_SIZE - thumbnail.height) // 2
    panel.paste(thumbnail, (left, top))
    return panel


def save_comparison(
    output_path: Path,
    image_id: str,
    original: np.ndarray,
    preprocessed: np.ndarray,
    segmentation: dict,
) -> None:
    mask_panel = segmentation["mask"] if segmentation["is_usable"] else None
    segmented_panel = segmentation["segmented_rgb"] if segmentation["is_usable"] else None
    boundary_panel = segmentation["boundary_overlay_rgb"] if segmentation["is_usable"] else None
    panels = [
        make_panel("Original image", original),
        make_panel("Preprocessed RGB (224 x 224)", preprocessed),
        make_panel("Approximate binary mask", mask_panel),
        make_panel("Segmented image", segmented_panel),
        make_panel("Approximate lesion boundary", boundary_panel),
    ]
    rows = (len(panels) + 1) // 2
    sheet = Image.new(
        "RGB",
        (2 * CELL_SIZE + 3 * GAP, rows * (CELL_SIZE + 30) + (rows + 1) * GAP),
        "#dfeef0",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((GAP, 2), f"{image_id} | {segmentation['status']} | {segmentation['method']}", fill="#13253f")
    for index, panel in enumerate(panels):
        x = GAP + (index % 2) * (CELL_SIZE + GAP)
        y = GAP + 18 + (index // 2) * (CELL_SIZE + 30 + GAP)
        sheet.paste(panel, (x, y))
    sheet.save(output_path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create small CV comparisons and feature measurements from images read directly from the ZIP."
    )
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.count <= 10:
        parser.error("count must be between 1 and 10 to keep generated demonstration files small.")

    args.output.mkdir(parents=True, exist_ok=True)
    feature_rows = []
    status_counts: dict[str, int] = {}
    with zipfile.ZipFile(args.archive) as archive:
        metadata_member = find_metadata_member(archive, "isic_2020_train_metadata.csv")
        metadata = read_csv_from_archive(archive, metadata_member)
        id_column = "image_name" if "image_name" in metadata.columns else "image"
        root = metadata_member.rsplit("/", 1)[0]
        image_members = find_image_members(archive, f"{root}/train")
        selected_ids = metadata[id_column].dropna().astype(str).tolist()[:args.count]
        missing = [image_id for image_id in selected_ids if image_id not in image_members]
        if missing:
            raise FileNotFoundError(f"Training images are missing for demonstration IDs: {missing}")

        for image_id in selected_ids:
            member = image_members[image_id]
            with archive.open(member) as stream:
                image_bytes = stream.read()
            preprocessing = preprocess_image(image_bytes, IMAGE_SIZE, hair_reduction=False)
            original = preprocessing["original_rgb"]
            preprocessed = preprocessing["preprocessed_rgb"]
            segmentation = segment_lesion(preprocessed)
            features = extract_features(preprocessed, segmentation)
            status_counts[segmentation["status"]] = status_counts.get(segmentation["status"], 0) + 1
            save_comparison(
                args.output / f"{image_id}_comparison.png",
                image_id,
                original,
                preprocessed,
                segmentation,
            )
            row = {
                "image_id": image_id,
                "segmentation_status": segmentation["status"],
                "segmentation_method": segmentation["method"],
                "segmentation_area_fraction": segmentation["area_fraction"],
                "feature_status": features["status"],
                "feature_reason": features.get("reason"),
            }
            values = features.get("traditional_features") or {}
            row.update({
                key: json.dumps(values.get(key), separators=(",", ":"))
                if isinstance(values.get(key), (list, dict)) else values.get(key)
                for key in FEATURE_FIELDS
            })
            feature_rows.append(row)
            print(
                f"{image_id}: segmentation={segmentation['status']} "
                f"method={segmentation['method']} features={features['status']}"
            )

    csv_path = args.output / "sample_features.csv"
    columns = [
        "image_id",
        "segmentation_status",
        "segmentation_method",
        "segmentation_area_fraction",
        "feature_status",
        "feature_reason",
        *FEATURE_FIELDS,
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(feature_rows)

    print(json.dumps({
        "images_processed": len(feature_rows),
        "segmentation_status_counts": status_counts,
        "comparison_directory": str(args.output),
        "feature_csv": str(csv_path),
        "classifier_required": False,
        "masks_are_ground_truth": False,
    }, indent=2))
    print("Segmentation is approximate OpenCV output; no mask-based accuracy is calculated.")


if __name__ == "__main__":
    main()
