"""Controlled geometry and label joins, not fabricated real-clip ground truth."""

import numpy as np
import pytest

from tennis3d.evaluation.annotations import (
    Annotation,
    AnnotationManifest,
    read_tracknet_csv,
)
from tennis3d.evaluation.landing import bounce_reference, landing_metrics
from tennis3d.evaluation.real import pixel_metrics
from tennis3d.evaluation.runs import run_report
from tennis3d.geometry.camera import Camera, ImageSize
from tennis3d.geometry.projection import project


def metadata():
    return {
        "source_sha256": "a" * 64,
        "clip_id": "controlled",
        "recording_group": "fixture",
        "split": "development",
        "width": 640,
        "height": 480,
        "source_frame_count": 4,
        "time_basis": "source_pts",
        "timing_note": "Controlled source timestamps",
        "pixel_convention": "ball centre",
        "rights_and_split_note": "Generated test fixture",
    }


def test_csv_mapping_and_pixel_denominators(tmp_path):
    csv = tmp_path / "Label.csv"
    csv.write_text(
        "file name,visibility,x-coordinate,y-coordinate,status\n"
        "0002.jpg,0,,,\n0000.jpg,1,10,20,0\n"
        "0003.jpg,3,30,40,2\n0001.jpg,2,20,30,2\n"
    )
    times = np.array([0.0, 0.04, 0.09, 0.13])
    mapping = {f"{i:04}.jpg": (i, float(t)) for i, t in enumerate(times)}
    labels = read_tracknet_csv(csv, image_mapping=mapping, metadata=metadata())
    assert [row.frame_index for row in labels.annotations] == [0, 1, 2, 3]
    assert sum(row.event == "bounce" for row in labels.annotations) == 2
    uv = np.array([[13, 24], [50, 30], [10, 10], [np.nan, np.nan]])
    report = pixel_metrics(
        labels,
        uv,
        np.array([True, True, True, False]),
        source_sha256="a" * 64,
        image_size=(640, 480),
        tolerances_px=[5.0],
        t_s=times,
    )
    score = report["all"]["scores"][0]
    assert (
        score["true_positive"],
        score["false_positive"],
        score["false_negative"],
    ) == (1, 2, 2)
    assert score["precision"] == score["recall"] == 1 / 3
    assert report["all"]["localization_error_px"]["mean"] == 17.5
    assert report["by_visibility"]["3"]["scores"][0]["false_negative"] == 1
    with pytest.raises(ValueError, match="timestamps"):
        pixel_metrics(
            labels,
            uv,
            np.array([True, True, True, False]),
            source_sha256="a" * 64,
            image_size=(640, 480),
            tolerances_px=[5.0],
            t_s=times + 0.01,
        )
    with pytest.raises(ValueError, match="unmapped"):
        read_tracknet_csv(csv, image_mapping={}, metadata=metadata())


def test_bounce_plane_convention_and_missing_coverage():
    camera = Camera(
        K=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
        R=np.diag([1, -1, -1]),
        t=np.array([0, 0, 10]),
        image_size=ImageSize(640, 480),
    )
    xyz = np.array([[1.0, 2.0, 0.0335]] * 4)
    uv, _ = project(camera, xyz)
    reference = bounce_reference(camera, uv[0], convention="ball_center")
    np.testing.assert_allclose(reference, xyz[0], atol=1e-12)
    ground = bounce_reference(camera, uv[0], convention="ground")
    assert ground[2] == 0 and ground[0] > reference[0]
    times = np.array([0.0, 0.04, 0.08, 0.12])
    rows = [
        Annotation(
            frame_index=i,
            image_name=f"{i}.jpg",
            t_s=float(times[i]),
            visibility=1,
            uv=uv[i].tolist(),
            event="bounce",
        )
        for i in range(4)
    ]
    labels = AnnotationManifest(**metadata(), labels_sha256="b" * 64, annotations=rows)
    xyz[0, 0] += 0.3
    xyz[0, 1] += 0.4
    report = landing_metrics(
        labels,
        times,
        xyz,
        np.array([True, False, True, True]),
        [camera, camera, None, camera],
        source_sha256="a" * 64,
        camera_source="controlled exact pinhole",
        convention="ball_center",
    )
    assert report["annotated_bounce_rows"] == 4
    assert report["coverage"] == 0.5
    assert report["landing_xy_error_m"]["mean"] == pytest.approx(0.25)
    assert report["records"][1]["reason"] == "unavailable_prediction"
    assert report["records"][2]["reason"] == "invalid_camera"
    nominal = labels.model_copy(update={"time_basis": "declared_nominal"})
    with pytest.raises(ValueError, match="source PTS"):
        landing_metrics(
            nominal,
            times,
            xyz,
            np.ones(4, dtype=bool),
            [camera] * 4,
            source_sha256="a" * 64,
            camera_source="fixture",
            convention="ground",
        )


def test_run_seconds_keep_duplicate_ambiguous_and_endpoint():
    report = run_report(
        [0, 0.04, 0.11, 0.2, 0.24],
        ["detected", "duplicate", "ambiguous", "detected", "missing"],
    )
    assert report["observation_counts"]["duplicate"] == 1
    assert report["observation_counts"]["ambiguous"] == 1
    assert report["empirical_corruption"]["missing_run_lengths_s"] == [0.16]
    assert report["runs"][-1]["duration_s"] == 0
    assert sum(row["duration_s"] for row in report["runs"]) == pytest.approx(0.24)
    explicit = run_report([0.0], ["missing"], end_time_s=0.04)
    assert explicit["empirical_corruption"]["missing_run_lengths_s"] == [0.04]
    assert not explicit["empirical_ready"]
