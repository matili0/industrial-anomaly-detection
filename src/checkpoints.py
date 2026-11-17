"""Read pure state dictionaries and the original teacher-student checkpoints."""

from pathlib import Path

import numpy as np
import torch


def load_student_weights(path: str | Path, device="cpu"):
    """Return tensor weights without enabling unrestricted pickle loading.

    Historical checkpoints include a ``student`` state dictionary, normalization
    tensors and a NumPy scalar threshold. New training runs save a state dictionary
    directly. The app uses raw scores, so calibration metadata is not applied here.
    """
    allowed = [
        np.dtype,
        np._core.multiarray.scalar,
        np.dtypes.Float32DType,
        np.dtypes.Float64DType,
    ]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a student checkpoint or state dictionary")
    state = checkpoint["student"] if "student" in checkpoint else checkpoint
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()
    ):
        raise ValueError("The student state dictionary must contain named tensors")
    return state
