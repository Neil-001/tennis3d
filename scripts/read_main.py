"""Read one view through a verified immutable manifest and explicit split."""

import argparse
import json
from pathlib import Path

from tennis3d.datasets.schema import Artifact
from tennis3d.datasets.source import SourceSpec, V2Source

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("pins", type=Path)
p.add_argument("cache", type=Path)
p.add_argument("--split", choices=("train", "val", "test"), default="train")
a = p.parse_args()
pins = json.loads(a.pins.read_text())
if pins["status"] != "verified-release":
    p.error("bind verified release receipts before reading")
main = pins["main"]
spec = SourceSpec(
    manifest=Artifact.model_validate(main["manifest_artifact"]),
    split=a.split,
    repo_id=pins["repo_id"],
    revision=main["commit"],
    cache_dir=str(a.cache.resolve() / a.split),
    cache_bytes=main["reader_cache_bytes"],
    max_group_bytes=main["max_group_bytes"],
)
source = V2Source(a.pins.resolve(), spec.model_dump())
try:
    arrays = source.read(0)
    print(json.dumps({"split": a.split, "revision": main["commit"], "view_read": True}))
finally:
    source.close()
