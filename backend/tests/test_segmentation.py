from __future__ import annotations

import base64
import unittest
from io import BytesIO
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from app.services.preprocessing import image_data_url, mask_data_url
from app.services.segmentation import segment_lesion


def make_test_image() -> np.ndarray:
    image = np.full((192, 192, 3), (170, 150, 130), dtype=np.uint8)
    cv2.ellipse(image, (96, 96), (38, 27), 20, 0, 360, (240, 80, 205), -1)
    return image


class SegmentationTests(unittest.TestCase):
    def test_success_outputs_share_a_valid_binary_mask_and_encoding(self) -> None:
        image = make_test_image()
        with self.assertLogs("app.services.segmentation", level="INFO") as diagnostics:
            result = segment_lesion(image)

        self.assertEqual(result["status"], "approximate")
        self.assertTrue(result["is_usable"])
        self.assertEqual(result["mask"].shape, image.shape[:2])
        self.assertEqual(result["mask"].dtype, np.uint8)
        self.assertGreater(result["foreground_pixels"], 0)
        self.assertEqual(result["mask_min"], 0)
        self.assertEqual(result["mask_max"], 255)
        self.assertGreater(result["contour_count"], 0)
        for field in ("method=", "mask_min=", "mask_max=", "foreground_pixels=", "foreground_percent=", "contours="):
            self.assertIn(field, diagnostics.output[0])
        np.testing.assert_array_equal(
            result["segmented_rgb"],
            cv2.bitwise_and(image, image, mask=result["mask"]),
        )

        encoded_mask = mask_data_url(result["mask"])
        self.assertTrue(encoded_mask.startswith("data:image/png;base64,"))
        with Image.open(BytesIO(base64.b64decode(encoded_mask.split(",", 1)[1]))) as decoded:
            self.assertEqual(decoded.mode, "L")
            self.assertEqual(decoded.size, (image.shape[1], image.shape[0]))
            np.testing.assert_array_equal(np.asarray(decoded), result["mask"])

        for encoded_image in (
            image_data_url(result["segmented_rgb"]),
            image_data_url(result["boundary_overlay_rgb"]),
        ):
            self.assertTrue(encoded_image.startswith("data:image/png;base64,"))
            with Image.open(BytesIO(base64.b64decode(encoded_image.split(",", 1)[1]))) as decoded:
                self.assertEqual(decoded.mode, "RGB")
                self.assertEqual(decoded.size, (image.shape[1], image.shape[0]))

    def test_lab_fallback_handles_grabcut_empty_output(self) -> None:
        image = make_test_image()

        def empty_grabcut(_image, mask, *_args) -> None:
            mask.fill(cv2.GC_BGD)

        with patch("app.services.segmentation.cv2.grabCut", side_effect=empty_grabcut):
            result = segment_lesion(image)

        self.assertEqual(result["status"], "approximate")
        self.assertEqual(result["method"], "opencv_lab_border_contrast")
        self.assertGreater(result["foreground_pixels"], 0)
        self.assertIsNotNone(result["segmented_rgb"])
        self.assertIsNotNone(result["boundary_overlay_rgb"])

    def test_low_contrast_image_reports_failure_without_a_mask(self) -> None:
        with self.assertLogs("app.services.segmentation", level="WARNING") as diagnostics:
            result = segment_lesion(np.full((192, 192, 3), (150, 150, 150), dtype=np.uint8))

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["is_usable"])
        self.assertEqual(result["mask_max"], 0)
        self.assertEqual(result["foreground_pixels"], 0)
        self.assertIn("insufficient colour contrast", result["message"])
        self.assertIsNone(result["segmented_rgb"])
        self.assertIsNone(result["boundary_overlay_rgb"])
        self.assertIn("status=failed", diagnostics.output[0])
        self.assertIn("reason=", diagnostics.output[0])


if __name__ == "__main__":
    unittest.main()
