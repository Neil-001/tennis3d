"""Regenerate a small analytical viewer fixture, not MuJoCo training data."""

import json
from pathlib import Path

import numpy as np

from tennis3d.artifacts import json_sha256
from tennis3d.court.model import court_landmarks
from tennis3d.geometry.camera import Camera
from tennis3d.geometry.image import ImageSize
from tennis3d.geometry.projection import project
from tennis3d.schemas import Result


def main():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "tests/fixtures/trajectory.json").read_text())
    data["source"].update(name="analytical-ballistic-example", frame_count=9)
    times = [0.0, 0.2, 0.55, 0.9, 1.05, 1.35, 1.6, 1.8, 2.0]
    xyz = [[0.5 + t, -9 + 8 * t, 1.5 + 9.81 * t - 4.905 * t * t] for t in times]
    center = np.array([0.0, -28.0, 18.0])
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.array([right, down, forward])
    camera = Camera(
        np.array([[480.0, 0, 320], [0, 480, 180], [0, 0, 1]]),
        rotation,
        -rotation @ center,
        ImageSize(640, 360),
    )
    court, _ = project(camera, court_landmarks())
    ball, _ = project(camera, np.array(xyz))
    frames, projections = [], []
    for i, t in enumerate(times):
        missing, gap = i in (2, 3, 5, 6), i in (5, 6)
        frames.append(
            {
                "frame_index": i,
                "t_s": t,
                "xyz_m": None if gap else xyz[i],
                "observation": "missing" if missing else "detected",
                "reconstruction": "unavailable"
                if gap
                else "inferred"
                if missing
                else "estimated",
                "camera_valid": True,
                "reason_codes": ["long_gap"] if gap else [],
                "duplicate_of": None,
            }
        )
        projections.append(
            {
                "frame_index": i,
                "t_s": t,
                "ball_clean_uv": ball[i].tolist(),
                "ball_observed_uv": None if missing else (ball[i] + [2, -1]).tolist(),
                "court_uv": court.tolist(),
            }
        )
    data["frames"] = frames
    data["summary"] = {"valid_frame_fraction": 7 / 9, "detected_frame_fraction": 5 / 9}
    data["provenance"]["calibration_assumptions"] = [
        "Analytical gravity-only flight, not MuJoCo or measured tennis.",
        "Artificial missing observations and unavailable interval for viewer checks.",
    ]
    data["synthetic_diagnostics"] = {
        "kind": "synthetic",
        "frames": projections,
        "events": [
            {"kind": "hit", "t_s": 0.0, "xyz_m": xyz[0]},
        ],
    }
    # The fixture's checksum identifies its deterministic source samples.
    data["source"]["sha256"] = json_sha256({"times": times, "xyz": xyz})
    result = Result.model_validate(data)
    (root / "viewer/example.json").write_text(result.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
