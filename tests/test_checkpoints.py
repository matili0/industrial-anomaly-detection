"""Regression checks for the two checkpoint formats used by this project."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.checkpoints import load_student_weights


class CheckpointTests(unittest.TestCase):
    def test_plain_state_dictionary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "student.pth"
            state = {"layer.weight": torch.arange(6).reshape(2, 3)}
            torch.save(state, path)
            loaded = load_student_weights(path)
            torch.testing.assert_close(loaded["layer.weight"], state["layer.weight"])

    def test_original_bundle_with_numpy_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "student.pth"
            state = {"layer.weight": torch.ones(2, 3)}
            torch.save({"student": state, "threshold": np.float64(0.2),
                        "score_mean": torch.tensor(0.1), "score_std": torch.tensor(0.4)}, path)
            loaded = load_student_weights(path)
            self.assertEqual(set(loaded), {"layer.weight"})
            torch.testing.assert_close(loaded["layer.weight"], state["layer.weight"])

    def test_invalid_payload_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.pth"
            torch.save({"student": {"layer.weight": "not a tensor"}}, path)
            with self.assertRaisesRegex(ValueError, "named tensors"):
                load_student_weights(path)


if __name__ == "__main__":
    unittest.main()
