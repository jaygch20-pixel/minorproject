from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_validation import (
    fixed_threshold_analysis,
    validation_threshold_analysis,
    weighted_loss_for_batch_size,
)


class ValidationEvaluationTests(unittest.TestCase):
    def test_weighted_loss_reproduces_weighted_batch_mean(self) -> None:
        loss = weighted_loss_for_batch_size(
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([0.1, 0.2], dtype=np.float64),
            np.asarray([1.0, 2.0], dtype=np.float64),
            batch_size=2,
        )

        expected = (-np.log(0.9) + 2.0 * -np.log(0.2)) / 3.0
        self.assertAlmostEqual(loss, expected, places=12)

    def test_validation_threshold_meets_only_its_declared_sensitivity_goal(self) -> None:
        result = validation_threshold_analysis(
            np.asarray([0, 0, 1, 1], dtype=np.int64),
            np.asarray([0.01, 0.35, 0.30, 0.40], dtype=np.float64),
            sensitivity_objectives=(0.90,),
        )["90%_target_sensitivity"]

        self.assertEqual(result["threshold"], 0.3)
        self.assertEqual(result["sensitivity"], 1.0)
        self.assertEqual(result["specificity"], 0.5)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])

    def test_fixed_threshold_report_preserves_the_requested_cutoffs(self) -> None:
        rows = fixed_threshold_analysis(
            np.asarray([0, 0, 1, 1], dtype=np.int64),
            np.asarray([0.01, 0.35, 0.30, 0.40], dtype=np.float64),
            thresholds=(0.5, 0.3),
        )

        self.assertEqual([row["threshold"] for row in rows], [0.5, 0.3])
        self.assertEqual(rows[0]["sensitivity"], 0.0)
        self.assertEqual(rows[1]["sensitivity"], 1.0)


if __name__ == "__main__":
    unittest.main()
