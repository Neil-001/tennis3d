"""Source-only streaming reproduction of nonzero, irregular downloaded PTS."""

import hashlib
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

av = pytest.importorskip("av")

from scripts import prepare_real_gallery
from scripts.prepare_real_gallery import encode, prepare, source_frames
from tennis3d.artifacts import file_sha256, read_json, write_json
from tennis3d.evaluation.inventory import Clip


@pytest.fixture
def gallery_source(tmp_path):
    original = tmp_path / "section.mp4"
    basis = Fraction(1, 12800)
    with av.open(
        str(original),
        "w",
        options={"movie_timescale": "12800", "avoid_negative_ts": "disabled"},
    ) as output:
        stream = output.add_stream("libx264rgb", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "rgb24"
        stream.time_base = stream.codec_context.time_base = basis
        stream.codec_context.thread_count = 1
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for i, pts in enumerate([1024, 1536, 2560, 4096, 4608]):
            pixels = np.full((32, 64, 3), i * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = pts, basis
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    rows = []
    with av.open(str(original)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if 1 <= i < 4:
                rows.append(
                    {
                        "frame_index": len(rows),
                        "original_frame_index": i,
                        "pts": frame.pts,
                        "time_base": [
                            frame.time_base.numerator,
                            frame.time_base.denominator,
                        ],
                        "decoded_rgb_sha256": hashlib.sha256(
                            frame.to_ndarray(format="rgb24").tobytes()
                        ).hexdigest(),
                    }
                )
    timeline = {"frames": rows, "image_size": [64, 32]}
    values = read_json("configs/stage4-real-gallery-v2/inventory.json")["clips"][0]
    values.update(
        clip_id="fixture",
        eligible=False,
        barriers=["fixture before encoding freeze"],
        width=64,
        height=32,
        frame_count=3,
        original_source_sha256=file_sha256(original),
        original_source_bytes=original.stat().st_size,
        original_source_frame_start=1,
        original_source_frame_end_exclusive=4,
        source_frame_start=0,
        categories=[],
        category_evidence={},
    )
    candidate = tmp_path / "frozen.mp4"
    clip = Clip.model_validate(values)
    encode(clip, original, candidate, timeline)
    timeline["source_sha256"] = file_sha256(candidate)
    timeline_path = tmp_path / "timeline.json"
    write_json(timeline_path, timeline)
    values.update(
        eligible=True,
        barriers=[],
        source_path=str(candidate),
        source_sha256=file_sha256(candidate),
        source_bytes=candidate.stat().st_size,
        timeline_path=str(timeline_path),
        timeline_sha256=file_sha256(timeline_path),
    )
    return Clip.model_validate(values), original, timeline


def test_reproduction_preserves_nonzero_irregular_pts_and_pixels(
    gallery_source, tmp_path
):
    clip, original, timeline = gallery_source
    receipt = prepare(clip, original, tmp_path / "prepared")
    assert receipt["every_rgb_and_integer_pts_verified"]
    assert receipt["prepared_sha256"] == clip.source_sha256
    assert timeline["frames"][0]["pts"] > 0
    pts = [row["pts"] for row in timeline["frames"]]
    assert pts[1] - pts[0] != pts[2] - pts[1]
    # The existing-output path must independently check its retained source too.
    original.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source identity"):
        prepare(clip, original, tmp_path / "prepared")


def test_wrong_original_frame_join_fails(gallery_source):
    clip, original, timeline = gallery_source
    timeline["frames"][0]["original_frame_index"] += 1
    with pytest.raises(ValueError, match="source frame pixels or exact timestamp"):
        list(source_frames(clip, original, timeline))


def test_changed_prepared_bytes_are_rejected(gallery_source, tmp_path):
    clip, original, _ = gallery_source
    prepare(clip, original, tmp_path / "prepared")
    target = Path(tmp_path / "prepared" / "fixture.mp4")
    target.write_bytes(target.read_bytes() + b"changed prepared bytes")
    with pytest.raises(ValueError, match="source video identity changed"):
        prepare(clip, original, tmp_path / "prepared")


@pytest.mark.parametrize("missing_filename", [None, ""])
def test_cli_rejects_missing_provenance_before_any_preparation(
    monkeypatch, tmp_path, capsys, missing_filename
):
    clips = [
        SimpleNamespace(eligible=True, clip_id="valid", original_source_sha256="a"),
        SimpleNamespace(eligible=True, clip_id="absent", original_source_sha256="b"),
        SimpleNamespace(eligible=True, clip_id="unnamed", original_source_sha256="c"),
    ]
    monkeypatch.setattr(
        prepare_real_gallery, "read_inventory", lambda _: SimpleNamespace(clips=clips)
    )
    provenance = tmp_path / "provenance.json"
    write_json(
        provenance,
        {
            "sources": [
                {"downloaded_sha256": "a", "downloaded_filename": "section.mp4"},
                {"downloaded_sha256": "c", "downloaded_filename": missing_filename},
            ]
        },
    )
    output, receipt = tmp_path / "prepared", tmp_path / "receipt.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_real_gallery",
            "--inventory",
            "unused.json",
            "--provenance",
            str(provenance),
            "--sources",
            str(tmp_path),
            "--output",
            str(output),
            "--receipt",
            str(receipt),
        ],
    )
    monkeypatch.setattr(
        prepare_real_gallery,
        "prepare",
        lambda *_: pytest.fail("preparation started before provenance validation"),
    )
    with pytest.raises(SystemExit) as error:
        prepare_real_gallery.main()
    assert error.value.code == 2
    assert (
        "provenance lacks downloaded source for: absent, unnamed"
        in capsys.readouterr().err
    )
    assert not output.exists()
    assert not receipt.exists()
