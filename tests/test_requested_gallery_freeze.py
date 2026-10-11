"""Requested source-second ranges map to exact decoded frames without adjustment."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")

from scripts.freeze_requested_gallery import freeze, map_ranges, prepare
from scripts.prepare_real_gallery import prepare as reproduce
from tennis3d.artifacts import file_sha256, read_json
from tennis3d.evaluation.inventory import read_inventory, verify_clip

# Nonzero first PTS and an irregular gap; times from the first frame, in seconds:
# 0, 0.25, 0.5, 1.0, 1.25, 2.0, 2.25
PTS = [512, 768, 1024, 1536, 1792, 2560, 2816]


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "sources" / "video.mp4"
    path.parent.mkdir()
    basis = Fraction(1, 1024)
    with av.open(
        str(path),
        "w",
        options={"movie_timescale": "1024", "avoid_negative_ts": "disabled"},
    ) as output:
        stream = output.add_stream("libx264rgb", rate=4)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "rgb24"
        stream.time_base = stream.codec_context.time_base = basis
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for i, pts in enumerate(PTS):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 64, 3), i * 30, dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = pts, basis
            output.mux(stream.encode(frame))
        output.mux(stream.encode())
    return path


def request(*ranges):
    return {
        "expected_final_clips": len(ranges),
        "provided_clips": [
            {"id": f"clip-{i}", "video_id": "video", "start_s": start, "end_s": end}
            for i, (start, end) in enumerate(ranges)
        ],
    }


def test_start_is_inclusive_end_is_exclusive_from_the_first_frame(source):
    rows, origin, size = map_ranges(source, {"a": (0, 1), "b": (1, 2), "c": (5, 6)})
    with av.open(str(source)) as container:
        first = next(container.decode(video=0)).pts
    assert origin == first and size == [64, 32]
    assert [row["original_frame_index"] for row in rows["a"]] == [0, 1, 2]
    # The frame at exactly 1 s opens the second range; the one at 2 s is outside.
    assert [row["original_frame_index"] for row in rows["b"]] == [3, 4]
    assert [row["source_time_s"] for row in rows["b"]] == ["1/1", "5/4"]
    assert origin > 0
    assert [
        Fraction((row["pts"] - origin) * row["time_base"][0], row["time_base"][1])
        for row in rows["b"]
    ] == [1, Fraction(5, 4)]
    assert rows["c"] == []


def test_freeze_keeps_every_requested_clip_and_reproduces(source, tmp_path):
    wanted = request((1, 2), (5, 6))
    records = prepare(wanted, source.parent, {"video": source.name}, tmp_path / "out")
    assert [record["frame_count"] for record in records] == [2, 0]
    assert records[1]["barrier"] == "requested range holds no decoded source frame"
    recording = {
        "recording_group": "fixture-recording",
        "recording_evidence": "fixture",
        "distinct_recording_verified": True,
        "local_use": "permitted",
        "redistribution": "unknown",
        "rights_evidence": "fixture",
    }
    review = {
        "recordings": {"video": recording},
        "clips": {
            "clip-0": {"categories": ["static"], "category_evidence": {"static": "x"}},
            "clip-1": {"categories": [], "category_evidence": {}},
        },
        "timing_note": "fixture",
        "selection_rule": "fixture",
        "excluded_development_recording_groups": [],
        "excluded_development_source_sha256": [],
        "exclusion_evidence": "fixture",
        "limitations": [],
    }
    prepared = {"request_sha256": "0" * 64, "records": records}
    provenance = {
        "sources": [{"youtube_id": "video", "downloaded_filename": "video.mp4"}]
    }
    output = tmp_path / "config"
    summary = freeze(wanted, prepared, review, provenance, "fixture-gallery", output)
    assert summary["preselected_records"] == 2 and summary["eligible_clips"] == 1
    private = read_inventory(output / "inventory.private.json")
    kept, empty = private.clips
    timeline = verify_clip(kept)
    assert [row["original_frame_index"] for row in timeline["frames"]] == [3, 4]
    assert timeline["requested_source_seconds"] == [1, 2]
    assert not empty.eligible and empty.barriers == [records[1]["barrier"]]
    portable = read_inventory(output / "inventory.json").clips[0]
    assert portable.source_path == "data/fixture-gallery/prepared/clip-0.mp4"
    assert portable.source_sha256 == kept.source_sha256
    mapping = read_json(output / "provenance.json")["sources"][0]
    assert mapping["downloaded_sha256"] == file_sha256(source)
    assert mapping["requested_clips"][0]["original_source_frame_start"] == 3
    # The existing reproduction tool regenerates the same bytes from the source.
    receipt = reproduce(kept, source, tmp_path / "reproduced")
    assert receipt["prepared_sha256"] == kept.source_sha256

    with pytest.raises(ValueError, match="every requested clip"):
        review["clips"].pop("clip-1")
        freeze(wanted, prepared, review, provenance, "fixture-gallery", tmp_path / "x")


def test_committed_gallery_keeps_every_requested_range_exactly():
    root = Path("configs/stage4-real-gallery-v2")
    selection = read_json(root / "source-selection.json")
    inventory = read_inventory(root / "inventory.json")
    wanted = {clip["id"]: clip for clip in selection["clips"]}
    assert len(wanted) == selection["expected_final_clips"] == 20
    assert [clip.clip_id for clip in inventory.clips] == list(wanted)
    manifest = read_json(root / "gallery-input.json")
    assert [clip["id"] for clip in manifest["clips"]] == list(wanted)
    assert manifest["inventory_sha256"] == file_sha256(root / "inventory.json")
    lowres = read_json(root / "matched-lowres-descriptor.json")
    assert lowres["parent_inventory_sha256"] == manifest["inventory_sha256"]
    for clip, entry, derived in zip(
        inventory.clips, manifest["clips"], lowres["clips"]
    ):
        assert clip.eligible and clip.label_status == "unavailable"
        assert clip.timeline_path == f"{root}/timelines/{clip.clip_id}.json"
        assert file_sha256(clip.timeline_path) == clip.timeline_sha256
        assert entry["source_sha256"] == clip.source_sha256
        assert entry["timeline_sha256"] == clip.timeline_sha256
        assert "result_path" not in entry
        assert derived["parent_source_sha256"] == clip.source_sha256
        timeline = read_json(clip.timeline_path)
        rows, origin = timeline["frames"], timeline["source_time_origin_pts"]
        start, end = wanted[clip.clip_id]["start_s"], wanted[clip.clip_id]["end_s"]
        assert timeline["requested_source_seconds"] == [start, end]
        assert len(rows) == clip.frame_count
        times = [
            Fraction((row["pts"] - origin) * row["time_base"][0], row["time_base"][1])
            for row in rows
        ]
        assert all(start <= time < end for time in times)
        assert times == sorted(set(times))
        assert [
            str(time.numerator) + "/" + str(time.denominator) for time in times
        ] == [row["source_time_s"] for row in rows]
        # The neighbouring source frames lie outside the range: none was trimmed.
        step = times[1] - times[0]
        assert times[0] - step < start and times[-1] + step >= end
        assert [row["original_frame_index"] for row in rows] == list(
            range(
                clip.original_source_frame_start,
                clip.original_source_frame_end_exclusive,
            )
        )
