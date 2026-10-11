"""Opt-in checkpoint compatibility and device execution check."""

import numpy as np


def check_lifter(checkpoint, requested):
    import torch

    from tennis3d.artifacts import file_sha256
    from tennis3d.inference.predict import predict_view
    from tennis3d.lifting.checkpoint import load_lifter
    from tennis3d.lifting.runtime import environment, resolve_device

    device = resolve_device(requested)
    torch.set_num_threads(2)
    model, state = load_lifter(checkpoint, device=device)
    xyz, covered, _ = predict_view(
        model,
        {
            "t_s": np.array([0.0, 0.04]),
            "image_size": np.array([640, 360]),
            "ball_uv": np.array([[320.0, 180.0], [np.nan, np.nan]]),
            "ball_observed": np.array([True, False]),
            "camera_valid": np.ones(2, dtype=bool),
            "court_uv": np.full((2, 14, 2), 100.0),
        },
        device=device,
        max_frames=2,
        stride=1,
    )
    if not covered.all() or not np.isfinite(xyz).all():
        raise RuntimeError("Lifter probe did not produce finite coverage")
    return {
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_step": state.get("step"),
        "weights": "EMA",
        "runtime": environment(device),
        "finite_forward": True,
        "accuracy_claim": "none; two-row execution probe",
    }
