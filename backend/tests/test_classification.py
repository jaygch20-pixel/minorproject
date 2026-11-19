from __future__ import annotations

import unittest

import numpy as np

from app.services.skin_analysis import map_class_probabilities


class ClassificationMappingTests(unittest.TestCase):
    def test_isic2020_output_indices_map_to_expected_labels(self) -> None:
        mapped = map_class_probabilities(
            np.asarray([0.936, 0.064], dtype=np.float32),
            {"0": "non_melanoma", "1": "melanoma"},
        )

        self.assertAlmostEqual(mapped["non_melanoma"], 0.936, places=6)
        self.assertAlmostEqual(mapped["melanoma"], 0.064, places=6)
        self.assertAlmostEqual(sum(mapped.values()), 1.0, places=6)

    def test_mapping_follows_declared_output_indices(self) -> None:
        mapped = map_class_probabilities(
            np.asarray([0.8, 0.2], dtype=np.float32),
            {"0": "melanoma", "1": "non_melanoma"},
        )

        self.assertAlmostEqual(mapped["melanoma"], 0.8, places=6)
        self.assertAlmostEqual(mapped["non_melanoma"], 0.2, places=6)

    def test_rejects_invalid_probability_vector_or_label_encoding(self) -> None:
        with self.assertRaises(ValueError):
            map_class_probabilities(
                np.asarray([0.8, 0.3], dtype=np.float32),
                {"0": "non_melanoma", "1": "melanoma"},
            )
        with self.assertRaises(ValueError):
            map_class_probabilities(
                np.asarray([0.8, 0.2], dtype=np.float32),
                {"0": "melanoma", "1": "melanoma"},
            )


if __name__ == "__main__":
    unittest.main()
