"""Source timeline endpoints cannot silently attach timing from another clip."""

import pytest

from tennis3d.app.gallery import Gallery
from tennis3d.artifacts import file_sha256, write_json


def test_gallery_timeline_requires_both_file_and_source_bindings(tmp_path):
    source_hash = "a" * 64
    timeline = tmp_path / "timeline.json"
    write_json(timeline, {"source_sha256": source_hash, "frames": []})
    manifest = tmp_path / "gallery.json"
    clip = {
        "id": "source-1",
        "source_sha256": source_hash,
        "timeline_path": timeline.name,
        "timeline_sha256": file_sha256(timeline),
    }
    write_json(manifest, {"schema_version": 1, "clips": [clip]})
    gallery = Gallery(manifest, tmp_path / "cache")
    assert gallery.describe()[0]["has_timeline"]
    assert gallery.timeline("source-1") == timeline
    write_json(timeline, {"source_sha256": "b" * 64}, overwrite=True)
    with pytest.raises(ValueError, match="SHA-256"):
        gallery.timeline("source-1")
    gallery.clips["source-1"]["timeline_sha256"] = file_sha256(timeline)
    with pytest.raises(ValueError, match="different source"):
        gallery.timeline("source-1")
    gallery.clips["source-1"].pop("timeline_path")
    with pytest.raises(KeyError):
        gallery.timeline("source-1")
