"""Paired low-resolution derivation keeps every parent frame, index and PTS."""

import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")

from scripts.derive_lowres_gallery import (
    area_resize,
    binding,
    derive,
    reproduce,
    rgb_sha256,
)
from tennis3d.artifacts import file_sha256, read_json, write_json
from tennis3d.evaluation.inventory import read_inventory, verify_clip

# Nonzero first PTS, an irregular gap and a repeated image (frames 2 and 3).
PTS = [1024, 1536, 2560, 4096, 4608]
SIZE = (30, 15)  # 7.5 source pixels per target pixel on both axes, as 1080p to 144p


def parent_inventory(root: Path, eligible=True):
    root.mkdir()
    source, basis = root / "parent.mp4", Fraction(1, 12800)
    rng = np.random.default_rng(7)
    images = [rng.integers(0, 256, (SIZE[1], SIZE[0], 3), dtype=np.uint8) for _ in PTS]
    images[3] = images[2]
    with av.open(
        str(source),
        "w",
        options={"movie_timescale": "12800", "avoid_negative_ts": "disabled"},
    ) as output:
        stream = output.add_stream("libx264rgb", rate=25)
        stream.width, stream.height, stream.pix_fmt = *SIZE, "rgb24"
        stream.time_base = stream.codec_context.time_base = basis
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for pts, image in zip(PTS, images):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = pts, basis
            output.mux(stream.encode(frame))
        output.mux(stream.encode())
    rows = [
        {
            "frame_index": i,
            "original_frame_index": 40 + i,
            "pts": pts,
            "time_base": [1, 12800],
            "source_time_s": f"{i}/25",
            "decoded_rgb_sha256": rgb_sha256(image),
        }
        for i, (pts, image) in enumerate(zip(PTS, images))
    ]
    timeline = root / "parent.json"
    write_json(
        timeline,
        {
            "source_sha256": file_sha256(source),
            "image_size": list(SIZE),
            "frames": rows,
        },
    )
    values = read_json("configs/stage4-real-gallery-v2/inventory.json")
    clip = dict(values["clips"][0])
    clip.update(
        clip_id="parent",
        source_path=str(source),
        source_sha256=file_sha256(source),
        source_bytes=source.stat().st_size,
        timeline_path=str(timeline),
        timeline_sha256=file_sha256(timeline),
        frame_count=len(PTS),
        width=SIZE[0],
        height=SIZE[1],
        original_source_frame_start=40,
        original_source_frame_end_exclusive=45,
        eligible=eligible,
        barriers=[] if eligible else ["fixture barrier"],
    )
    path = root / "inventory.json"
    write_json(path, {**values, "clips": [clip]})
    return path, images


def reference(image, width, height):
    """Area average from exact rational pixel overlaps."""
    rows, columns = image.shape[:2]
    out = np.zeros((height, width, 3), dtype=np.uint8)

    def overlaps(index, source, target):
        low, high = (
            Fraction(index * source, target),
            Fraction((index + 1) * source, target),
        )
        return [
            (k, min(high, k + 1) - max(low, k))
            for k in range(int(low), min(source, int(high) + 1))
            if min(high, k + 1) > max(low, k)
        ]

    for y in range(height):
        for x in range(width):
            for channel in range(3):
                total = sum(
                    int(image[r, c, channel]) * a * b
                    for r, a in overlaps(y, rows, height)
                    for c, b in overlaps(x, columns, width)
                )
                mean = total / (Fraction(rows, height) * Fraction(columns, width))
                out[y, x, channel] = int(mean + Fraction(1, 2))  # round half up
    return out


def test_area_resize_is_the_exact_rounded_area_average():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (15, 30, 3), dtype=np.uint8)
    assert np.array_equal(area_resize(image, 4, 2), reference(image, 4, 2))
    assert np.array_equal(area_resize(image, 10, 5), reference(image, 10, 5))
    assert np.array_equal(area_resize(image, 30, 15), image)
    flat = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    assert np.array_equal(area_resize(flat, 256, 144), flat[:144, :256])
    # A half sum rounds up: pixels 0, 0, 1, 1 average to 1, and 0, 0, 0, 1 to 0.
    block = np.zeros((2, 2, 3), dtype=np.uint8)
    block[1] = 1
    assert area_resize(block, 1, 1).tolist() == [[[1, 1, 1]]]
    block[1, 0] = 0
    assert area_resize(block, 1, 1).tolist() == [[[0, 0, 0]]]
    with pytest.raises(ValueError, match="aspect ratio"):
        area_resize(image, 4, 3)
    with pytest.raises(ValueError, match="exceed"):
        area_resize(image, 60, 30)


def test_derivation_pairs_every_frame_and_reproduces(tmp_path):
    parents, images = parent_inventory(tmp_path / "parent")
    output = tmp_path / "lowres"
    summary = derive(parents, output, (4, 2), "fixture-lowres", "-lowres")
    assert summary["eligible_clips"] == 1
    assert summary["intended_role"] == "resolution_stress"
    parent = read_inventory(parents).clips[0]
    clip = read_inventory(output / "inventory.private.json").clips[0]
    timeline = verify_clip(clip)
    assert clip.clip_id == "parent-lowres" and [clip.width, clip.height] == [4, 2]
    assert clip.source_revision == binding(parent)
    assert timeline["parent_source_sha256"] == parent.source_sha256
    assert timeline["parent_timeline_sha256"] == parent.timeline_sha256
    rows = timeline["frames"]
    assert [row["pts"] for row in rows] == PTS
    assert [row["original_frame_index"] for row in rows] == [40, 41, 42, 43, 44]
    assert [row["parent_frame_index"] for row in rows] == [0, 1, 2, 3, 4]
    assert all(row["time_base"] == [1, 12800] for row in rows)
    assert [row["source_time_s"] for row in rows] == [f"{i}/25" for i in range(5)]
    # The repeated parent image stays a separate row with its own timestamp.
    assert rows[2]["decoded_rgb_sha256"] == rows[3]["decoded_rgb_sha256"]
    with av.open(clip.source_path) as container:
        decoded = [
            (frame.pts, frame.to_ndarray(format="rgb24"))
            for frame in container.decode(video=0)
        ]
    assert [pts for pts, _ in decoded] == PTS
    for row, image, (_, small) in zip(rows, images, decoded):
        assert np.array_equal(small, reference(image, 4, 2))
        assert row["decoded_rgb_sha256"] == rgb_sha256(small)
        assert row["parent_decoded_rgb_sha256"] == rgb_sha256(image)
        assert row["decoded_rgb_sha256"] != row["parent_decoded_rgb_sha256"]
    pairing = read_json(output / "pairing.json")
    assert pairing["clips"][0]["parent_source_sha256"] == parent.source_sha256
    assert pairing["frames"] == 5
    portable = read_inventory(output / "inventory.json").clips[0]
    assert portable.source_path == "data/fixture-lowres/prepared/parent-lowres.mp4"

    records = reproduce(output / "inventory.private.json", parents, tmp_path / "again")
    assert records[0]["reproduced_sha256"] == clip.source_sha256
    assert pairing["tool_sha256"] == file_sha256("scripts/derive_lowres_gallery.py")
    assert any(name.startswith("libx264") for name in pairing["bundled_library_sha256"])
    # A derived record that disagrees with its parent or timeline gets no receipt.
    frozen = read_json(output / "inventory.private.json")
    for field, value in (
        ("frame_count", 1),
        ("original_source_sha256", "0" * 64),
        ("original_source_frame_start", 41),
        ("width", 8),
    ):
        changed = {**frozen, "clips": [{**frozen["clips"][0], field: value}]}
        write_json(tmp_path / f"{field}.json", changed)
        with pytest.raises(ValueError, match="disagrees with its parent"):
            reproduce(tmp_path / f"{field}.json", parents, tmp_path / f"out-{field}")
        assert not list((tmp_path / f"out-{field}").iterdir())
    # A frozen timeline that no longer matches the parent stops reproduction.
    edited = read_json(clip.timeline_path)
    edited["frames"][0]["pts"] += 1
    write_json(clip.timeline_path, edited, overwrite=True)
    with pytest.raises(ValueError, match="derived timeline changed"):
        reproduce(output / "inventory.private.json", parents, tmp_path / "third")


def test_wrong_or_incomplete_parents_write_no_inventory(tmp_path):
    parents, _ = parent_inventory(tmp_path / "parent")
    source = Path(read_inventory(parents).clips[0].source_path)
    intact = tmp_path / "intact.mp4"
    shutil.copy(source, intact)

    source.write_bytes(source.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="source video identity changed"):
        derive(parents, tmp_path / "corrupt", (4, 2), "fixture", "-lowres")
    assert not (tmp_path / "corrupt" / "inventory.json").exists()
    shutil.copy(intact, source)

    # The right file with a timeline row naming other pixels is refused too.
    values = read_json(parents)
    timeline = read_json(values["clips"][0]["timeline_path"])
    timeline["frames"][4]["decoded_rgb_sha256"] = "0" * 64
    write_json(values["clips"][0]["timeline_path"], timeline, overwrite=True)
    values["clips"][0]["timeline_sha256"] = file_sha256(
        values["clips"][0]["timeline_path"]
    )
    write_json(parents, values, overwrite=True)
    with pytest.raises(ValueError, match="parent frame pixels"):
        derive(parents, tmp_path / "rows", (4, 2), "fixture", "-lowres")
    assert not (tmp_path / "rows" / "inventory.json").exists()
    assert not list((tmp_path / "rows" / "prepared").iterdir())

    with pytest.raises(ValueError, match="aspect ratio"):
        derive(
            parent_inventory(tmp_path / "other")[0],
            tmp_path / "aspect",
            (4, 3),
            "fixture",
            "-lowres",
        )
    blocked, _ = parent_inventory(tmp_path / "blocked", eligible=False)
    with pytest.raises(ValueError, match="none may be dropped"):
        derive(blocked, tmp_path / "dropped", (4, 2), "fixture", "-lowres")
    assert not (tmp_path / "dropped").exists()


def test_committed_lowres_cohort_pairs_every_parent_row():
    root = Path("configs/stage4-real-gallery-v2-lowres")
    parents = read_inventory("configs/stage4-real-gallery-v2/inventory.json")
    derived = read_inventory(root / "inventory.json")
    pairing = read_json(root / "pairing.json")
    descriptor = read_json(
        "configs/stage4-real-gallery-v2/matched-lowres-descriptor.json"
    )
    assert derived.intended_role == "resolution_stress"
    assert descriptor["inventory_sha256"] == file_sha256(root / "inventory.json")
    assert descriptor["pairing_sha256"] == file_sha256(root / "pairing.json")
    assert len(derived.clips) == len(parents.clips) == len(pairing["clips"]) == 20
    frames = 0
    for parent, clip, pair, bound in zip(
        parents.clips, derived.clips, pairing["clips"], descriptor["clips"]
    ):
        assert clip.clip_id == parent.clip_id + "-lowres" == pair["clip_id"]
        assert clip.eligible and [clip.width, clip.height] == [256, 144]
        assert clip.source_revision == binding(parent)
        assert clip.source_sha256 == pair["source_sha256"] == bound["source_sha256"]
        assert clip.source_sha256 != parent.source_sha256
        assert bound["parent_timeline_sha256"] == parent.timeline_sha256
        assert file_sha256(clip.timeline_path) == clip.timeline_sha256
        timeline, original = (
            read_json(clip.timeline_path),
            read_json(parent.timeline_path),
        )
        assert timeline["source_sha256"] == clip.source_sha256
        assert timeline["parent_source_sha256"] == parent.source_sha256
        assert timeline["parent_timeline_sha256"] == parent.timeline_sha256
        assert timeline["image_size"] == [256, 144]
        assert len(timeline["frames"]) == len(original["frames"]) == clip.frame_count
        for row, source in zip(timeline["frames"], original["frames"]):
            assert (
                row["frame_index"] == row["parent_frame_index"] == source["frame_index"]
            )
            for key in ("original_frame_index", "pts", "time_base", "source_time_s"):
                assert row[key] == source[key]
            assert row["parent_decoded_rgb_sha256"] == source["decoded_rgb_sha256"]
            assert row["decoded_rgb_sha256"] != source["decoded_rgb_sha256"]
        frames += clip.frame_count
    assert frames == pairing["frames"] == 10749
