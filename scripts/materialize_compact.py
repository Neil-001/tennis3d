"""Materialize only compact artifacts, then verify them offline before timing."""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

from tennis3d.datasets.schema import Artifact
from tennis3d.datasets.source import SourceSpec, VerifiedCache
from tennis3d.datasets.subset import (
    materialize_subset,
    read_descriptor,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("pins", type=Path)
p.add_argument("output", type=Path)
p.add_argument("cache", type=Path)
a = p.parse_args()
pins = json.loads(a.pins.read_text())
if pins["status"] != "verified-release":
    p.error("bind verified release receipts before materializing")
compact = pins["compact"]
artifact = Artifact.model_validate(compact["descriptor_artifact"])
canonical_sha = compact["descriptor_sha256"]
spec = SourceSpec(
    manifest=artifact,
    split="train",
    repo_id=pins["repo_id"],
    revision=compact["publication_commit"],
    cache_dir=str(a.cache.resolve()),
    cache_bytes=compact["reader_cache_bytes"],
    max_group_bytes=compact["max_group_bytes"],
)
cache = VerifiedCache(a.cache.resolve(), spec)


@contextmanager
def resolver(item):
    path = cache.get(item)
    cache.pinned = path
    try:
        yield path
    finally:
        cache.pinned = None


try:
    descriptor = read_descriptor(cache.get(artifact), expected_sha256=canonical_sha)
    if (descriptor.main_repo_id, descriptor.main_commit) != (
        pins["repo_id"],
        pins["main"]["commit"],
    ):
        raise ValueError("compact descriptor binds to a different main release")
    # This API verifies copied numeric artifacts offline before returning.
    materialize_subset(
        descriptor,
        a.output.resolve(),
        resolver=resolver,
        expected_descriptor_sha256=canonical_sha,
        max_group_bytes=compact["max_group_bytes"],
    )
finally:
    cache.close()
print(
    json.dumps(
        {
            "materialized": True,
            "verified_offline": True,
            "output": str(a.output.resolve()),
        }
    )
)
