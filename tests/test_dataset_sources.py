"""Model-free cache admission, retry, ownership and reader memory boundaries."""

import hashlib

import pytest

pytest.importorskip("pyarrow")
from test_dataset_parquet import sample, write_sample  # noqa: F401

from tennis3d.datasets.io import ShardReader
from tennis3d.datasets.schema import Artifact
from tennis3d.datasets.source import SourceSpec, VerifiedCache


def spec(**kwargs):
    return SourceSpec(
        manifest=artifact(b"manifest"),
        split="train",
        repo_id="test/data",
        revision="a" * 40,
        cache_dir="cache",
        **kwargs,
    )


def artifact(data, path="file"):
    return Artifact(
        path=path, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)
    )


def test_cache_accounts_partials_and_protects_pinned_entry(tmp_path, monkeypatch):
    from tennis3d.datasets import source

    cache = VerifiedCache(tmp_path / "cache", spec(cache_bytes=12))

    def chunks(_spec, value):
        yield value.path.encode()

    monkeypatch.setattr(source, "hub_chunks", chunks)
    try:
        a = artifact(b"abcdef", "abcdef")
        first = cache.get(a)
        cache.pinned = first
        # Interrupted bytes count, and can be evicted before admitting a fetch.
        (cache.root / ("f" * 64 + ".partial")).write_bytes(b"leftovers")
        b = cache.get(artifact(b"ghijkl", "ghijkl"))
        assert first.exists() and b.exists() and cache.usage() == 12
        with pytest.raises(ValueError, match="pinned"):
            cache.get(artifact(b"1234567", "1234567"))
        assert first.exists()
        with pytest.raises(ValueError, match="already in use"):
            VerifiedCache(cache.root, spec(cache_bytes=12))
    finally:
        cache.close()


@pytest.mark.parametrize("bad", [b"too-long", b"bad", b"xy"])
def test_cache_rejects_bad_transfer_without_residue(tmp_path, monkeypatch, bad):
    from tennis3d.datasets import source

    def chunks(*args):
        yield bad

    monkeypatch.setattr(source, "hub_chunks", chunks)
    cache = VerifiedCache(tmp_path / "cache", spec(cache_bytes=8))
    try:
        with pytest.raises(ValueError, match="size|checksum"):
            cache.get(artifact(b"good"))
        assert cache.usage() == 0
        assert [p.name for p in cache.root.iterdir()] == ["lock"]
    finally:
        cache.close()


def test_cache_stops_after_bounded_retries(tmp_path, monkeypatch):
    from tennis3d.datasets import source

    calls = []

    def chunks(*args):
        calls.append(1)
        yield b"x"
        raise OSError("interrupted")

    monkeypatch.setattr(source, "hub_chunks", chunks)
    monkeypatch.setattr(source.time, "sleep", lambda _: None)
    cache = VerifiedCache(tmp_path / "cache", spec(cache_bytes=8, retries=2))
    try:
        with pytest.raises(OSError):
            cache.get(artifact(b"good"))
        assert len(calls) == 3 and cache.usage() == 0
    finally:
        cache.close()


def test_reader_rejects_group_before_decode(tmp_path, sample, monkeypatch):  # noqa: F811
    path, shard = write_sample(tmp_path, sample)
    with ShardReader(path, shard, provenance=sample[1], max_group_bytes=1) as reader:

        def forbidden(*args, **kwargs):
            pytest.fail("oversize group reached decode")

        monkeypatch.setattr(reader._parquet, "read_row_group", forbidden)
        with pytest.raises(ValueError, match="admission bound"):
            reader.read(0)


def test_public_reader_never_resolves_cached_publishing_token(monkeypatch):
    import httpx
    from huggingface_hub import utils
    from huggingface_hub.utils import _headers

    from tennis3d.datasets.source import hub_chunks

    def forbidden():
        pytest.fail("public read resolved the cached publishing credential")

    monkeypatch.setattr(_headers, "get_token", forbidden)
    # Test the explicit path even if implicit-token disabling is absent.
    monkeypatch.setattr(_headers.constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)

    def respond(request):
        assert request.method == "GET" and "authorization" not in request.headers
        return httpx.Response(200, content=b"public")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(utils, "get_session", lambda: client)
        assert list(hub_chunks(spec(), artifact(b"public"))) == [b"public"]


@pytest.mark.parametrize("elapsed, succeeds", [(36, True), (61, False)])
def test_transfer_budget_is_separate_from_request_timeout(
    monkeypatch, elapsed, succeeds
):
    import httpx
    from huggingface_hub import utils

    from tennis3d.datasets import source

    clock = [0.0]
    monkeypatch.setattr(source.time, "monotonic", lambda: clock[0])

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            clock[0] = elapsed
            yield b"public"

    def respond(request):
        assert request.extensions["timeout"]["read"] == 30
        return httpx.Response(200, stream=Stream())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(utils, "get_session", lambda: client)
        chunks = source.hub_chunks(spec(transfer_seconds=60), artifact(b"public"))
        if succeeds:
            assert list(chunks) == [b"public"]
        else:
            with pytest.raises(OSError, match="pinned Hub download failed"):
                list(chunks)


@pytest.mark.parametrize("budget", [0, -1, 1801, float("inf")])
def test_transfer_budget_rejects_unbounded_values(budget):
    with pytest.raises(ValueError):
        spec(transfer_seconds=budget)


@pytest.mark.parametrize("encoding", [None, "gzip"])
def test_transfer_deadline_precedes_httpx_buffering_and_decoding(monkeypatch, encoding):
    import gzip

    import httpx
    from huggingface_hub import utils

    from tennis3d.datasets import source

    clock = [0.0]
    closed = []
    payload = b"public bytes" * 20
    wire = gzip.compress(payload) if encoding else payload
    monkeypatch.setattr(source.time, "monotonic", lambda: clock[0])

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for byte in wire:
                clock[0] += 10
                yield bytes([byte])

        def close(self):
            closed.append(True)

    def respond(request):
        assert request.extensions["timeout"]["read"] == 30
        assert "authorization" not in request.headers
        headers = {"content-encoding": encoding} if encoding else {}
        return httpx.Response(200, headers=headers, stream=Stream())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(utils, "get_session", lambda: client)
        with pytest.raises(OSError, match="pinned Hub download failed"):
            list(source.hub_chunks(spec(transfer_seconds=60), artifact(payload)))
    assert clock[0] == 70 <= 60 + 30
    assert closed == [True]


def test_httpx_transfer_preserves_decompressed_artifact_bytes(tmp_path, monkeypatch):
    import gzip

    import httpx
    from huggingface_hub import utils

    payload = b"public artifact" * 100
    wire = gzip.compress(payload)

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for byte in wire:
                yield bytes([byte])

    def respond(request):
        return httpx.Response(
            200, headers={"content-encoding": "gzip"}, stream=Stream()
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(utils, "get_session", lambda: client)
        with_cache = VerifiedCache(tmp_path / "cache", spec(cache_bytes=len(payload)))
        try:
            path = with_cache.get(artifact(payload))
            assert path.read_bytes() == payload
            assert with_cache.usage() == len(payload)
        finally:
            with_cache.close()


def test_manifest_validation_rechecks_mutable_pool_provenance(tmp_path, sample):  # noqa: F811
    from tennis3d.artifacts import json_sha256
    from tennis3d.datasets.schema import Manifest, count_records

    _, shard = write_sample(tmp_path, sample)
    manifest = Manifest(
        state="complete",
        provenance=sample[1],
        shards=[shard],
        expected=count_records([shard]),
    )
    raw = manifest.model_dump()
    assert json_sha256(Manifest.model_validate(raw).model_dump()) == json_sha256(raw)
    # A previous successful validation must not leave a stale cached pool lookup.
    pool = manifest.provenance.pools[0]
    manifest.provenance.pools[0] = pool.model_copy(update={"sha256": "f" * 64})
    with pytest.raises(ValueError, match="rally family/pool provenance mismatch"):
        Manifest.model_validate(manifest.model_dump())
    manifest.provenance.pools[0] = pool.model_copy(update={"split": "val"})
    with pytest.raises(ValueError, match="rally family/pool provenance mismatch"):
        Manifest.model_validate(manifest.model_dump())
    manifest.provenance.pools.clear()
    with pytest.raises(ValueError, match="rally family/pool provenance mismatch"):
        Manifest.model_validate(manifest.model_dump())
