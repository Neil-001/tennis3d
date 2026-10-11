"""Raw dense window predictions shared by synthetic evaluation and video lifting."""

import numpy as np
import torch

from tennis3d.inference.windows import WindowMerger, make_windows
from tennis3d.lifting.features import WORLD_SCALE_M, build_features


@torch.inference_mode()
def predict_view(
    model,
    arrays: dict,
    *,
    device="cpu",
    max_frames=250,
    stride=125,
    require_complete=False,
    progress=None,
):
    """Return (XYZ metres, coverage, RMS window disagreement).

    All-missing windows are skipped for production. Evaluation uses
    require_complete=True and rejects any such window before model execution.
    Invalid camera rows remain unpredicted. No gap/plausibility filtering occurs.
    """
    windows = make_windows(arrays["camera_valid"], max_frames=max_frames, stride=stride)
    supported = []
    for w in windows:
        if np.asarray(arrays["ball_observed"])[w.start : w.stop].any():
            supported.append(w)
        elif require_complete:
            raise ValueError(
                f"evaluation window {w.start}:{w.stop} has no ball evidence"
            )
    merger = WindowMerger(len(arrays["t_s"]))
    was_training = model.training
    model.eval()
    try:
        from tennis3d.inference.progress import report

        for index, w in enumerate(supported):
            report(progress, "lift", index, len(supported), "windows")
            sl = slice(w.start, w.stop)
            f = build_features(
                **{
                    name: arrays[name][sl]
                    for name in (
                        "t_s",
                        "ball_uv",
                        "ball_observed",
                        "court_uv",
                        "camera_valid",
                    )
                },
                image_size=arrays["image_size"],
                pad_to=max_frames,
            )
            xyz = model(f.to(device))[0, : w.length].cpu().numpy() * WORLD_SCALE_M
            merger.add(w, xyz)
    finally:
        model.train(was_training)
    report(progress, "lift", len(supported), len(supported), "windows")
    result = merger.finish()
    if require_complete:
        expected = np.asarray(arrays["camera_valid"]) & np.asarray(
            arrays["label_valid"]
        )
        if not result[1][expected].all():
            raise ValueError("evaluation does not cover every supported labelled frame")
    return result
