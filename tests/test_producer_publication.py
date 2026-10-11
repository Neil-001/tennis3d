"""Producer readback uses fake immutable Hub snapshots and HTTP transports."""

import gzip
import hashlib
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub import utils
from huggingface_hub.utils import _headers

from tennis3d.datasets import source
from tennis3d.datasets.publication import RemoteMissing, StagingFull, VerificationError
from tennis3d.datasets.schema import Artifact
from tennis3d.synthetic import producer_publication
from tennis3d.synthetic.producer_publication import Publication


def artifact(content=b"verified payload", path="dataset/shards/000.parquet"):
    return Artifact(
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


class FakeHub:
    def __init__(self):
        self.current = "0" * 40
        self.snapshots = {self.current: {}}
        self.uploads = 0
        self.reads = []
        self.corrupt = False

    def head(self, repo_id, revision):
        return self.current

    def upload(self, repo_id, revision, parent, path, local):
        assert parent == self.current
        self.uploads += 1
        snapshot = {**self.snapshots[parent], path: local.read_bytes()}
        self.current = f"{self.uploads:040x}"
        self.snapshots[self.current] = snapshot
        return self.current

    def fingerprint(self, repo_id, commit, path):
        self.reads.append((commit, path))
        try:
            content = self.snapshots[commit][path]
        except KeyError:
            raise RemoteMissing(path) from None
        if self.corrupt:
            content += b"corrupt"
        return hashlib.sha256(content).hexdigest(), len(content)


@pytest.fixture
def publication(tmp_path):
    return Publication(
        tmp_path,
        SimpleNamespace(
            staging_bytes=65536,
            metadata_reserve_bytes=8192,
            repo_id="owner/dataset",
            revision="release",
            evict=True,
        ),
        FakeHub(),
    )


def publish_evicted(publication, content=b"verified payload"):
    spec = artifact(content)
    publication.staging.stage(spec, [content])
    commit = publication.publish(spec, evict=True)
    assert not publication.staging.local_path(spec).exists()
    return spec, commit


def forbid_credentials(monkeypatch):
    def forbidden():
        pytest.fail("anonymous producer read resolved a publishing credential")

    monkeypatch.setattr(_headers, "get_token", forbidden)
    monkeypatch.setattr(_headers.constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)


def test_resolve_reuses_explicit_bounded_source_descriptor(publication, monkeypatch):
    spec, commit = publish_evicted(publication)
    observed = []

    def chunks(descriptor, item):
        observed.append(descriptor)
        assert item == spec
        yield b"verified payload"

    monkeypatch.setattr(producer_publication, "hub_chunks", chunks)
    with publication.resolve(spec) as path:
        assert path.read_bytes() == b"verified payload"
    assert len(observed) == 1
    descriptor = observed[0]
    assert isinstance(descriptor, source.SourceSpec)
    assert descriptor.repo_id == "owner/dataset"
    assert descriptor.revision == commit
    assert descriptor.timeout_seconds == 30
    assert descriptor.transfer_seconds == 600
    # Transport policy still passes the merged validator's hard ceiling.
    with pytest.raises(ValueError):
        source.SourceSpec.model_validate(
            {**descriptor.model_dump(), "transfer_seconds": 1801}
        )


@pytest.mark.parametrize("compressed", [False, True])
def test_anonymous_readback_is_pinned_and_eviction_rechecks_receipt(
    publication, monkeypatch, compressed
):
    payload = b"verified payload" * 100
    spec, commit = publish_evicted(publication, payload)
    backend = publication.publisher.backend
    backend.current = "f" * 40
    backend.snapshots[backend.current] = {spec.path: b"branch moved"}
    backend.reads.clear()
    forbid_credentials(monkeypatch)
    closed, requests = [], []
    wire = gzip.compress(payload) if compressed else payload

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for offset in range(0, len(wire), 7):
                yield wire[offset : offset + 7]

        def close(self):
            closed.append(True)

    def respond(request):
        requests.append(request)
        assert request.method == "GET"
        assert (
            request.url.path == f"/datasets/owner/dataset/resolve/{commit}/{spec.path}"
        )
        assert "authorization" not in request.headers
        assert set(request.extensions["timeout"].values()) == {30}
        headers = {"content-encoding": "gzip"} if compressed else {}
        return httpx.Response(200, headers=headers, stream=Stream())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(utils, "get_session", lambda: client)
        with publication.resolve(spec) as path:
            assert path.read_bytes() == payload
            assert backend.reads == []
        assert not path.exists()
    assert len(requests) == 1
    assert closed == [True]
    assert backend.reads == [(commit, spec.path)]
    assert backend.uploads == 1
    assert publication.receipt(spec)["commit"] == commit


@pytest.mark.parametrize("compressed", [False, True])
def test_raw_fragment_deadline_precedes_buffering_and_decompression(
    publication, monkeypatch, compressed
):
    spec, _ = publish_evicted(publication)
    clock, closed, consumed = [0.0], [], []
    monkeypatch.setattr(source.time, "monotonic", lambda: clock[0])
    payload = b"verified payload"
    wire = gzip.compress(payload) if compressed else payload

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for byte in wire:
                clock[0] += 100
                consumed.append(byte)
                yield bytes([byte])

        def close(self):
            closed.append(True)

    def respond(request):
        assert request.extensions["timeout"]["read"] == 30
        headers = {"content-encoding": "gzip"} if compressed else {}
        return httpx.Response(200, headers=headers, stream=Stream())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(utils, "get_session", lambda: client)
        with (
            pytest.raises(OSError, match="^pinned Hub download failed$"),
            publication.resolve(spec),
        ):
            pytest.fail("expired transfer reached the reader")
    assert clock[0] == 700
    assert len(consumed) == 7 < len(wire)
    assert closed == [True]
    assert not publication.staging.local_path(spec).exists()
    assert not list(publication.staging.files.rglob(".partial-*"))


@pytest.mark.parametrize("elapsed,succeeds", [(36, True), (599, True), (601, False)])
def test_transfer_budget_is_separate_from_request_timeout(
    publication, monkeypatch, elapsed, succeeds
):
    spec, _ = publish_evicted(publication)
    clock = [0.0]
    monkeypatch.setattr(source.time, "monotonic", lambda: clock[0])

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            clock[0] = elapsed
            yield b"verified payload"

    def respond(request):
        assert request.extensions["timeout"]["read"] == 30
        return httpx.Response(200, stream=Stream())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(utils, "get_session", lambda: client)
        if succeeds:
            with publication.resolve(spec) as path:
                assert path.read_bytes() == b"verified payload"
        else:
            with (
                pytest.raises(OSError, match="pinned Hub download failed"),
                publication.resolve(spec),
            ):
                pytest.fail("expired transfer reached the reader")


@pytest.mark.parametrize(
    "payload",
    [b"wrong hash bytes", b"short", b"oversized payload" * 5000],
    ids=["wrong-hash", "short", "oversized"],
)
def test_readback_hash_and_size_failures_close_transport_and_remove_partial(
    publication, monkeypatch, payload
):
    spec, _ = publish_evicted(publication)
    closed = []

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield payload
            pytest.fail("staging read beyond the first oversized fragment")

        def close(self):
            closed.append(True)

    def respond(request):
        # Oversize the first decoded chunk so staging stops with the generator
        # suspended inside hub_chunks, before the HTTP response's normal close.
        if len(payload) > spec.size_bytes:
            return httpx.Response(200, stream=Stream())
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(utils, "get_session", lambda: client)
        with pytest.raises(VerificationError), publication.resolve(spec):
            pytest.fail("unverified readback reached the reader")
    assert not publication.staging.local_path(spec).exists()
    assert not list(publication.staging.files.rglob(".partial-*"))
    if len(payload) > spec.size_bytes:
        assert closed == [True]


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", "uploaded"),
        ("sha256", "a" * 64),
        ("size_bytes", 1),
        ("path", "other.parquet"),
        ("commit", "release"),
    ],
)
def test_receipt_must_match_exact_artifact_before_network(
    publication, monkeypatch, field, value
):
    spec, _ = publish_evicted(publication)
    ledger = publication.publisher._load()
    ledger["entries"][spec.path][field] = value
    publication.publisher._save(ledger)

    def forbidden():
        pytest.fail("invalid receipt opened a network session")

    monkeypatch.setattr(utils, "get_session", forbidden)
    with pytest.raises(ValueError), publication.resolve(spec):
        pytest.fail("invalid receipt reached the reader")


def test_staging_admission_precedes_transport(publication, monkeypatch):
    spec, _ = publish_evicted(publication)
    publication.staging.max_bytes = publication.staging.metadata_reserve_bytes

    def forbidden():
        pytest.fail("full staging opened a network session")

    monkeypatch.setattr(utils, "get_session", forbidden)
    with pytest.raises(StagingFull), publication.resolve(spec):
        pytest.fail("full staging reached the reader")
    assert not list(publication.staging.files.rglob(".partial-*"))


def test_missing_receipt_blocks_remote_fetch(publication, monkeypatch):
    def forbidden():
        pytest.fail("missing receipt opened a network session")

    monkeypatch.setattr(utils, "get_session", forbidden)
    with (
        pytest.raises(ValueError, match="missing exact verified publication receipt"),
        publication.resolve(artifact()),
    ):
        pytest.fail("missing receipt reached the reader")


@pytest.mark.parametrize("failure", ["timeout", "http-status"])
def test_transport_failure_is_sanitized_and_keeps_receipt(
    publication, monkeypatch, failure
):
    spec, commit = publish_evicted(publication)

    def respond(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("signed URL contains secret", request=request)
        return httpx.Response(403, content=b"signed URL contains secret")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(utils, "get_session", lambda: client)
        with (
            pytest.raises(OSError, match="^pinned Hub download failed$"),
            publication.resolve(spec),
        ):
            pytest.fail("failed transport reached the reader")
    assert not publication.staging.local_path(spec).exists()
    assert not list(publication.staging.files.rglob(".partial-*"))
    assert publication.receipt(spec)["commit"] == commit


def test_disabled_eviction_retains_verified_download(publication, monkeypatch):
    spec, _ = publish_evicted(publication)
    publication.execution.evict = False
    backend = publication.publisher.backend
    backend.reads.clear()
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"verified payload")
        )
    ) as client:
        monkeypatch.setattr(utils, "get_session", lambda: client)
        with publication.resolve(spec) as path:
            assert path.read_bytes() == b"verified payload"
    assert path.read_bytes() == b"verified payload"
    assert backend.reads == []


def test_existing_local_artifact_is_verified_without_fetch_or_eviction(
    publication, monkeypatch
):
    spec = artifact()
    publication.staging.stage(spec, [b"verified payload"])

    def forbidden(*args):
        pytest.fail("existing local artifact attempted remote resolution")

    monkeypatch.setattr(producer_publication, "hub_chunks", forbidden)
    with publication.resolve(spec) as path:
        assert path.read_bytes() == b"verified payload"
    assert path.exists()
    path.write_bytes(b"changed payload!")
    with (
        pytest.raises(ValueError, match="owned artifact differs"),
        publication.resolve(spec),
    ):
        pytest.fail("changed local artifact reached the reader")


def test_failed_eviction_readback_retains_downloaded_artifact(publication, monkeypatch):
    spec, commit = publish_evicted(publication)
    backend = publication.publisher.backend
    backend.reads.clear()
    backend.corrupt = True
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"verified payload")
        )
    ) as client:
        monkeypatch.setattr(utils, "get_session", lambda: client)
        with (
            pytest.raises(VerificationError, match="Remote content mismatch"),
            publication.resolve(spec) as path,
        ):
            assert path.read_bytes() == b"verified payload"
    assert path.read_bytes() == b"verified payload"
    assert backend.reads == [(commit, spec.path)]


def test_consumer_failure_still_uses_verified_eviction(publication, monkeypatch):
    spec, commit = publish_evicted(publication)
    backend = publication.publisher.backend
    backend.reads.clear()
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"verified payload")
        )
    ) as client:
        monkeypatch.setattr(utils, "get_session", lambda: client)
        with (
            pytest.raises(RuntimeError, match="reader failed"),
            publication.resolve(spec) as path,
        ):
            raise RuntimeError("reader failed")
    assert not path.exists()
    assert backend.reads == [(commit, spec.path)]


@pytest.mark.parametrize("damage", ["entries-list", "entry-list"])
def test_malformed_receipt_shape_is_refused_without_network(publication, damage):
    spec, _ = publish_evicted(publication)
    publisher = publication.publisher
    assert publisher is not None
    ledger = publisher._load()
    if damage == "entries-list":
        ledger["entries"] = []
    else:
        ledger["entries"][spec.path] = ["verified"]
    publisher._save(ledger)
    reads = list(publisher.backend.reads)
    with pytest.raises(ValueError, match="publication receipt"):
        publication.receipt(spec)
    assert publisher.backend.reads == reads
