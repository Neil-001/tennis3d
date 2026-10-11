"""Publication failure tests use immutable in-memory Hub snapshots."""

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from tennis3d.datasets.publication import (
    BoundedStaging,
    HuggingFaceBackend,
    Publisher,
    RemoteMissing,
    StagingFull,
    VerificationError,
)


@dataclass
class Artifact:
    path: str
    sha256: str
    size_bytes: int


def artifact(path="shards/000.jsonl", content=b"sample\n"):
    return Artifact(path, hashlib.sha256(content).hexdigest(), len(content))


class FakeHub:
    def __init__(self):
        self.current = "0" * 40
        self.snapshots = {self.current: {}}
        self.uploads = 0
        self.fail = None
        self.corrupt = False
        self.reads = []

    def head(self, repo_id, revision):
        return self.current

    def upload(self, repo_id, revision, parent, path, local):
        assert parent == self.current
        if self.fail == "before":
            raise ConnectionError("offline")
        self.uploads += 1
        snapshot = dict(self.snapshots[parent])
        snapshot[path] = local.read_bytes()
        self.current = f"{self.uploads:040x}"
        self.snapshots[self.current] = snapshot
        if self.fail == "after":
            raise ConnectionError("response lost after successful commit")
        return self.current

    def fingerprint(self, repo_id, commit, path):
        self.reads.append((commit, path))
        try:
            data = self.snapshots[commit][path]
        except KeyError:
            raise RemoteMissing(path) from None
        if self.corrupt:
            data += b"truncated or corrupt transport"
        digest = hashlib.sha256()
        for offset in range(0, len(data), 2):
            digest.update(data[offset : offset + 2])
        return digest.hexdigest(), len(data)


@pytest.fixture
def setup(tmp_path):
    staging = BoundedStaging(tmp_path, 65536, metadata_reserve_bytes=8192)
    hub = FakeHub()
    return staging, hub, Publisher(staging, hub, "owner/dataset")


def test_full_budget_does_not_advance_producer(tmp_path):
    staging = BoundedStaging(tmp_path, 1028, metadata_reserve_bytes=1024)
    consumed = []

    def producer():
        consumed.append(True)
        yield b"sample\n"

    with pytest.raises(StagingFull):
        staging.stage(artifact(), producer())
    assert not consumed
    assert not list(staging.files.rglob(".partial-*"))


@pytest.mark.parametrize("content", [b"too large", b"bad", b""])
def test_bad_producer_retains_no_partial(tmp_path, content):
    staging = BoundedStaging(tmp_path, 2048, metadata_reserve_bytes=1024)
    with pytest.raises(VerificationError):
        staging.stage(artifact(content=b"good"), [content])
    assert not list(staging.files.rglob(".partial-*"))
    assert not staging.local_path(artifact()).exists()


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a//b", "a/./b", "a\\b"])
def test_staging_rejects_unsafe_path(setup, path):
    staging, _, _ = setup
    with pytest.raises(ValueError):
        staging.stage(artifact(path), [b"sample\n"])


def test_symlink_cannot_escape_staging(setup, tmp_path):
    staging, _, _ = setup
    (staging.files / "shards").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        staging.stage(artifact(), [b"sample\n"])


def test_staging_accepts_symlink_above_root_but_rejects_one_inside(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "scratch"
    alias.symlink_to(actual, target_is_directory=True)
    staging = BoundedStaging(alias / "stage", 65536, metadata_reserve_bytes=8192)
    item = artifact()
    staged = staging.stage(item, [b"sample\n"])
    hub = FakeHub()
    Publisher(staging, hub, "owner/dataset").publish(item, evict=True)
    assert not staged.exists()
    (staging.files / "escape").symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        staging.local_path(artifact("escape/payload"))


def test_success_receipt_is_durable_before_eviction(setup):
    staging, hub, publisher = setup
    item = artifact()
    local = staging.stage(item, [b"sample\n"])
    commit = publisher.publish(item, evict=True)
    assert not local.exists()
    envelope = json.loads(publisher.ledger_path.read_text())
    entry = envelope["ledger"]["entries"][item.path]
    assert entry["state"] == "verified"
    assert entry["commit"] == commit
    reopened = Publisher(staging, hub, "owner/dataset")
    assert reopened.publish(item, evict=True) == commit
    assert hub.uploads == 1


@pytest.mark.parametrize("failure", ["before", "after"])
def test_restart_resolves_ambiguous_commit_without_duplicate_upload(setup, failure):
    staging, hub, publisher = setup
    item = artifact()
    local = staging.stage(item, [b"sample\n"])
    hub.fail = failure
    with pytest.raises(ConnectionError):
        publisher.publish(item, evict=True)
    assert local.exists()
    assert (
        json.loads(publisher.ledger_path.read_text())["ledger"]["entries"][item.path][
            "state"
        ]
        == "pending"
    )
    hub.fail = None
    reopened = Publisher(staging, hub, "owner/dataset")
    reopened.publish(item, evict=True)
    assert hub.uploads == 1
    assert not local.exists()


def test_failed_remote_readback_never_evicts(setup):
    staging, hub, publisher = setup
    item = artifact()
    local = staging.stage(item, [b"sample\n"])
    hub.corrupt = True
    with pytest.raises(VerificationError):
        publisher.publish(item, evict=True)
    assert local.read_bytes() == b"sample\n"
    hub.corrupt = False
    publisher.publish(item, evict=True)
    assert hub.uploads == 1


def test_size_is_verified_independently_of_hash(setup, monkeypatch):
    staging, hub, publisher = setup
    item = artifact()
    local = staging.stage(item, [b"sample\n"])
    original = hub.fingerprint

    def wrong_size(*args):
        digest, size = original(*args)
        return digest, size + 1

    monkeypatch.setattr(hub, "fingerprint", wrong_size)
    with pytest.raises(VerificationError):
        publisher.publish(item, evict=True)
    assert local.exists()


def test_upload_requires_immutable_commit_even_if_file_exists(setup, monkeypatch):
    staging, hub, publisher = setup
    item = artifact()
    local = staging.stage(item, [b"sample\n"])
    upload = hub.upload

    def unpinned(*args):
        upload(*args)
        return "main"

    monkeypatch.setattr(hub, "upload", unpinned)
    with pytest.raises(VerificationError, match="immutable commit"):
        publisher.publish(item, evict=True)
    assert local.exists()
    entry = json.loads(publisher.ledger_path.read_text())["ledger"]["entries"][
        item.path
    ]
    assert entry["state"] == "pending"


def test_readback_uses_upload_commit_when_branch_moves(setup, monkeypatch):
    staging, hub, publisher = setup
    item = artifact()
    staging.stage(item, [b"sample\n"])
    upload = hub.upload

    def moving_branch(*args):
        commit = upload(*args)
        hub.current = "f" * 40
        hub.snapshots[hub.current] = {item.path: b"different bytes"}
        return commit

    monkeypatch.setattr(hub, "upload", moving_branch)
    commit = publisher.publish(item, evict=True)
    assert commit != hub.current
    assert hub.reads[-1] == (commit, item.path)


def test_corrupt_ledger_blocks_eviction(setup):
    staging, _, publisher = setup
    item = artifact()
    local = staging.stage(item, [b"sample\n"])
    publisher.publish(item)
    envelope = json.loads(publisher.ledger_path.read_text())
    envelope["ledger"]["entries"][item.path]["commit"] = "f" * 40
    publisher.ledger_path.write_text(json.dumps(envelope))
    with pytest.raises(VerificationError, match="ledger"):
        publisher.publish(item, evict=True)
    assert local.exists()


def test_receipt_save_failure_keeps_local_copy(setup, monkeypatch):
    staging, _, publisher = setup
    item = artifact()
    local = staging.stage(item, [b"sample\n"])
    original = publisher._save

    def fail_verified(ledger):
        if ledger["entries"][item.path]["state"] == "verified":
            raise OSError("disk failed")
        original(ledger)

    monkeypatch.setattr(publisher, "_save", fail_verified)
    with pytest.raises(OSError):
        publisher.publish(item, evict=True)
    assert local.exists()


def test_ledger_growth_is_bounded(tmp_path):
    staging = BoundedStaging(tmp_path, 4096, metadata_reserve_bytes=1024)
    hub = FakeHub()
    publisher = Publisher(staging, hub, "owner/dataset")
    item = artifact("a" * 400)
    # Long valid remote path exceeds the receipt budget before any transfer.
    item.path = "/".join(["a" * 100] * 4)
    local = staging.stage(item, [b"sample\n"])
    with pytest.raises(StagingFull, match="ledger"):
        publisher.publish(item, evict=True)
    assert hub.uploads == 0
    assert local.exists()


def test_accounting_charges_stale_partials_and_sdk_files(tmp_path):
    staging = BoundedStaging(tmp_path, 1032, metadata_reserve_bytes=1024)
    (staging.files / ".partial-stale").write_bytes(b"1234")
    with pytest.raises(StagingFull):
        staging.stage(artifact(), [b"sample\n"])


def prepare_release(staging, publisher):
    data = artifact()
    audit = artifact("audit.json", b'{"passed":true}')
    manifest = artifact("manifest.json", b'{"complete":true}')
    for item, content in [
        (data, b"sample\n"),
        (audit, b'{"passed":true}'),
        (manifest, b'{"complete":true}'),
    ]:
        staging.stage(item, [content])
    publisher.publish(data, evict=True)
    publisher.publish(audit)
    return data, audit, manifest


def test_release_commits_manifest_last_after_validation(setup):
    staging, hub, publisher = setup
    data, audit, manifest = prepare_release(staging, publisher)
    card = artifact("README.md", b"Dataset card and usage restrictions")
    license_file = artifact("LICENSE", b"CC BY 4.0")
    for item, content in [
        (card, b"Dataset card and usage restrictions"),
        (license_file, b"CC BY 4.0"),
    ]:
        staging.stage(item, [content])
        publisher.publish(item)
    validated = []

    def validate(manifest_path, audit_path):
        assert json.loads(manifest_path.read_text())["complete"]
        assert json.loads(audit_path.read_text())["passed"]
        assert manifest.path not in hub.snapshots[hub.current]
        validated.append(True)

    parent = hub.current
    hub.reads.clear()
    commit = publisher.publish_release(
        manifest, [data, audit, card, license_file], audit, validate
    )
    assert validated == [True]
    assert hub.uploads == 5
    assert hub.reads == [
        (parent, data.path),
        (parent, audit.path),
        (parent, card.path),
        (parent, license_file.path),
        (parent, manifest.path),
        (commit, manifest.path),
    ]
    assert (
        json.loads(publisher.ledger_path.read_text())["ledger"]["complete"]["commit"]
        == commit
    )


def test_release_rejects_missing_expected_artifact_or_duplicate(setup):
    staging, _, publisher = setup
    data, audit, manifest = prepare_release(staging, publisher)
    with pytest.raises(VerificationError, match="sets differ"):
        publisher.publish_release(manifest, [audit], audit, lambda *_: None)
    with pytest.raises(VerificationError, match="unique"):
        publisher.publish_release(manifest, [data, data, audit], audit, lambda *_: None)


def test_release_failed_audit_does_not_publish_manifest(setup):
    staging, hub, publisher = setup
    data, audit, manifest = prepare_release(staging, publisher)

    def rejected(*args):
        raise VerificationError("Full audit failed")

    with pytest.raises(VerificationError, match="audit failed"):
        publisher.publish_release(manifest, [data, audit], audit, rejected)
    assert manifest.path not in hub.snapshots[hub.current]
    assert "complete" not in json.loads(publisher.ledger_path.read_text())["ledger"]


def test_release_checks_remote_set_before_manifest(setup):
    staging, hub, publisher = setup
    data, audit, manifest = prepare_release(staging, publisher)
    hub.snapshots[hub.current].pop(data.path)
    with pytest.raises(RemoteMissing):
        publisher.publish_release(manifest, [data, audit], audit, lambda *_: None)
    assert manifest.path not in hub.snapshots[hub.current]


def test_release_ambiguous_upload_restarts_without_duplicate_commit(setup):
    staging, hub, publisher = setup
    data, audit, manifest = prepare_release(staging, publisher)
    hub.fail = "after"
    with pytest.raises(ConnectionError):
        publisher.publish_release(manifest, [data, audit], audit, lambda *_: None)
    hub.fail = None
    Publisher(staging, hub, "owner/dataset").publish_release(
        manifest, [data, audit], audit, lambda *_: None
    )
    assert hub.uploads == 3


def test_sdk_worker_isolates_caches_and_credentials(setup, monkeypatch):
    staging, _, _ = setup
    observed = []

    def run(command, **kwargs):
        assert "secret" not in str(command)
        env = kwargs["env"]
        for key in (
            "HF_HOME",
            "HF_HUB_CACHE",
            "HF_XET_CACHE",
            "HF_ASSETS_CACHE",
            "TMPDIR",
        ):
            assert Path(env[key]).is_relative_to(staging.control)
        assert env["HF_HUB_DISABLE_XET"] == "1"
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        code = command[-1]
        assert code.index("setrlimit") < code.index("from huggingface_hub")
        assert "RLIMIT_FSIZE, (0, 0)" in code
        assert "parent_commit=request" in code
        assert ".stream(" in code
        request = json.loads(kwargs["input"])
        assert request["token"] == "secret"
        observed.append(request)
        return SimpleNamespace(stdout=json.dumps({"result": "1" * 40}).encode())

    monkeypatch.setattr(subprocess, "run", run)
    backend = HuggingFaceBackend(staging, token="secret")
    assert backend.head("owner/dataset", "main") == "1" * 40
    assert observed[0]["action"] == "head"


def test_sdk_failure_does_not_expose_credentials(setup, monkeypatch):
    staging, _, _ = setup

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "worker", stderr=b"secret")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(RuntimeError) as exc:
        HuggingFaceBackend(staging, token="secret").head("owner/data", "main")
    assert "secret" not in str(exc.value)


def test_sdk_resolves_cached_authentication_in_parent(setup, monkeypatch):
    staging, _, _ = setup
    calls = []

    def cached_token():
        calls.append("SDK credential lookup")
        return "cached-secret"

    def run(command, **kwargs):
        assert "cached-secret" not in str(command)
        assert json.loads(kwargs["input"])["token"] == "cached-secret"
        return SimpleNamespace(stdout=json.dumps({"result": "a" * 40}).encode())

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(get_token=cached_token)
    )
    monkeypatch.setattr(subprocess, "run", run)
    backend = HuggingFaceBackend(staging)
    assert not calls
    assert backend.head("owner/data", "main") == "a" * 40
    assert calls == ["SDK credential lookup"]


def test_sdk_missing_auth_fails_before_worker(setup, monkeypatch):
    staging, _, _ = setup
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(get_token=lambda: None)
    )
    with pytest.raises(RuntimeError, match="authentication is required"):
        HuggingFaceBackend(staging).head("owner/data", "main")


def test_sdk_worker_executes_supported_calls_and_enforces_zero_disk_writes(
    setup, tmp_path, monkeypatch
):
    staging, _, _ = setup
    package_root = tmp_path / "fake_sdk"
    package = package_root / "huggingface_hub"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("""
import os, resource
from pathlib import Path
from types import SimpleNamespace
assert os.environ["HF_HUB_DISABLE_XET"] == "1"
assert resource.getrlimit(resource.RLIMIT_FSIZE) == (0, 0)
try:
    with open(Path(os.environ["HF_HOME"]) / "cache_attempt", "wb", buffering=0) as f:
        f.write(b"unbounded cache")
except OSError:
    pass
else:
    raise AssertionError("SDK cache write was not blocked")
class CommitOperationAdd:
    def __init__(self, path_in_repo, path_or_fileobj):
        assert path_in_repo == "data.jsonl"
        assert Path(path_or_fileobj).read_bytes() == b"payload"
class HfApi:
    def __init__(self, token):
        assert token == "explicit-token"
    def repo_info(self, **kwargs):
        assert kwargs == {"repo_id": "owner/data", "repo_type": "dataset", "revision": "release"}
        return SimpleNamespace(sha="a" * 40)
    def create_commit(self, **kwargs):
        assert kwargs["parent_commit"] == "a" * 40
        assert kwargs["revision"] == "release"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["num_threads"] == 1
        assert len(kwargs["operations"]) == 1
        return SimpleNamespace(oid="b" * 40)
def hf_hub_url(**kwargs):
    assert kwargs["revision"] == "b" * 40
    assert kwargs["repo_type"] == "dataset"
    return "https://mock.invalid/data"
""")
    (package / "utils.py").write_text("""
def build_hf_headers(token):
    return {"Authorization": token}
class Response:
    status_code = 200
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def raise_for_status(self):
        pass
    def iter_bytes(self, chunk_size):
        assert chunk_size == 1024 * 1024
        yield b"pay"
        yield b"load"
class Session:
    def stream(self, method, url, headers, follow_redirects):
        assert method == "GET"
        assert headers["Authorization"] == "explicit-token"
        assert follow_redirects
        return Response()
def get_session():
    return Session()
""")
    monkeypatch.setenv("PYTHONPATH", str(package_root))
    backend = HuggingFaceBackend(staging, token="explicit-token")
    item = artifact("data.jsonl", b"payload")
    local = staging.stage(item, [b"payload"])
    assert backend.head("owner/data", "release") == "a" * 40
    assert (
        backend.upload("owner/data", "release", "a" * 40, item.path, local) == "b" * 40
    )
    assert backend.fingerprint("owner/data", "b" * 40, item.path) == (
        item.sha256,
        item.size_bytes,
    )
    assert (staging.control / "sdk" / "cache_attempt").stat().st_size == 0
