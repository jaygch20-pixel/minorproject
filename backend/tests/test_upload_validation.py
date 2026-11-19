from __future__ import annotations

import io
import unittest

import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageFilter

from app.main import validate_upload_image


def encode_image(image: Image.Image, image_format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


class UploadValidationTests(unittest.TestCase):
    def test_accepts_detailed_rgb_and_rgba_png_images(self) -> None:
        pixels = np.indices((128, 128)).sum(axis=0).astype(np.uint8) * 2
        rgb = np.stack((pixels, np.flipud(pixels), np.fliplr(pixels)), axis=2)
        png = encode_image(Image.fromarray(rgb, mode="RGB"))
        rgba = encode_image(Image.fromarray(np.dstack((rgb, np.full_like(pixels, 255))), mode="RGBA"))

        self.assertEqual(validate_upload_image(png), "PNG")
        self.assertEqual(validate_upload_image(rgba), "PNG")

    def test_rejects_tiny_nearly_uniform_and_severely_blurred_images(self) -> None:
        tiny = encode_image(Image.new("RGB", (32, 32), (120, 80, 60)))
        uniform = encode_image(Image.new("RGB", (128, 128), (120, 80, 60)))
        noisy = np.random.default_rng(42).integers(0, 256, (256, 256), dtype=np.uint8)
        blurred = Image.fromarray(noisy, mode="L").filter(ImageFilter.GaussianBlur(radius=24))

        for image_bytes, message in (
            (tiny, "at least"),
            (uniform, "too little visible detail"),
            (encode_image(blurred), "out of focus"),
        ):
            with self.subTest(message=message):
                with self.assertRaises(HTTPException) as error:
                    validate_upload_image(image_bytes)
                self.assertEqual(error.exception.status_code, 400)
                self.assertIn(message, error.exception.detail)

    def test_rejects_unsupported_and_corrupt_files(self) -> None:
        with self.assertRaises(HTTPException) as unsupported:
            validate_upload_image(encode_image(Image.new("RGB", (128, 128)), "BMP"))
        self.assertEqual(unsupported.exception.status_code, 400)

        with self.assertRaises(HTTPException) as corrupt:
            validate_upload_image(b"not an image")
        self.assertEqual(corrupt.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
