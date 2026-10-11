import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest

from tennis3d.synthetic.producer_recovery import (
    BundleSizeError,
    RecoveryBundleError,
    create_bundle,
    restore_bundle,
)


def test_bundle_round_trip_into_fresh_directory(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    (source / "plan.json").write_text('{"seed":17}\n')
    (source / "state").mkdir()
    (source / "state" / "ledger.json").write_bytes(b"ledger\x00state")
    bundle = tmp_path / "recovery.zip"

    artifact = create_bundle(
        source, ["state/ledger.json", "plan.json"], bundle, max_bytes=4096
    )
    restored = tmp_path / "restored"
    report = restore_bundle(bundle, restored, max_bytes=4096)

    assert artifact.path == bundle
    assert artifact.size_bytes == bundle.stat().st_size
    assert artifact.sha256 == hashlib.sha256(bundle.read_bytes()).hexdigest()
    assert [item.path for item in artifact.files] == ["plan.json", "state/ledger.json"]
    assert report.manifest_sha256 == artifact.manifest_sha256
    assert report.bundle_sha256 == artifact.sha256
    assert (restored / "plan.json").read_bytes() == (source / "plan.json").read_bytes()
    assert (restored / "state" / "ledger.json").read_bytes() == b"ledger\x00state"


def test_bundle_cap_includes_zip_overhead_and_leaves_no_artifact(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    (source / "state.json").write_bytes(b"x")
    bundle = tmp_path / "recovery.zip"

    with pytest.raises(BundleSizeError, match="exceeds max_bytes"):
        create_bundle(source, ["state.json"], bundle, max_bytes=32)

    assert not bundle.exists()
    assert not list(tmp_path.glob(".recovery.zip.*.tmp"))


def test_restore_rejects_truncated_and_tampered_archives_without_partial_output(
    tmp_path,
):
    source = tmp_path / "run"
    source.mkdir()
    (source / "ledger.json").write_bytes(b"durable state")
    original = tmp_path / "original.zip"
    create_bundle(source, ["ledger.json"], original, max_bytes=4096)

    damaged_archives = []
    truncated = tmp_path / "truncated.zip"
    truncated.write_bytes(original.read_bytes()[:-12])
    damaged_archives.append(truncated)
    tampered = tmp_path / "tampered.zip"
    data = bytearray(original.read_bytes())
    offset = data.find(b"durable state")
    assert offset >= 0
    data[offset] ^= 1
    tampered.write_bytes(data)
    damaged_archives.append(tampered)

    for index, damaged in enumerate(damaged_archives):
        destination = tmp_path / f"restore-{index}"
        with pytest.raises(RecoveryBundleError):
            restore_bundle(damaged, destination, max_bytes=4096)
        assert not destination.exists()
        assert not list(tmp_path.glob(f".{destination.name}.restore-*"))


def _write_manifest_archive(path: Path, member_name: str, *, duplicate: bool = False):
    content = b"payload"
    record = {
        "path": member_name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }
    manifest = (
        json.dumps(
            {
                "files": [record],
                "format": "tennis3d-producer-recovery",
                "version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    entries = [(member_name, content)]
    if duplicate:
        entries.append((member_name, content))
    entries.extend(
        [
            (".recovery/manifest.json", manifest),
            (
                ".recovery/manifest.sha256",
                (hashlib.sha256(manifest).hexdigest() + "\n").encode(),
            ),
        ]
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in entries:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payload)


def test_restore_rejects_path_traversal(tmp_path):
    bundle = tmp_path / "traversal.zip"
    _write_manifest_archive(bundle, "../outside")

    with pytest.raises(RecoveryBundleError, match="canonical relative"):
        restore_bundle(bundle, tmp_path / "restored", max_bytes=4096)

    assert not (tmp_path / "outside").exists()


def test_create_rejects_symlink_and_hardlink_sources(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    target = source / "state.json"
    target.write_text("state")
    symlink = source / "linked.json"
    symlink.symlink_to(target)

    with pytest.raises(RecoveryBundleError, match="regular file"):
        create_bundle(source, ["linked.json"], tmp_path / "symlink.zip", 4096)

    hardlink = source / "hardlink.json"
    hardlink.hardlink_to(target)
    with pytest.raises(RecoveryBundleError, match="hard-linked"):
        create_bundle(source, ["state.json"], tmp_path / "hardlink.zip", 4096)


def test_restore_rejects_symlink_member(tmp_path):
    bundle = tmp_path / "symlink-member.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        info = zipfile.ZipInfo("state.json")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "target")

    with pytest.raises(RecoveryBundleError, match="plain stored file"):
        restore_bundle(bundle, tmp_path / "restored", max_bytes=4096)


def test_restore_rejects_duplicate_members(tmp_path):
    bundle = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_manifest_archive(bundle, "state.json", duplicate=True)

    with pytest.raises(RecoveryBundleError, match="duplicate entries"):
        restore_bundle(bundle, tmp_path / "restored", max_bytes=4096)


@pytest.mark.parametrize(
    "private_path",
    [".aws/credentials", ".cache/huggingface/token", "token.json", "client.pem"],
)
def test_create_rejects_credentials_and_sdk_caches(tmp_path, private_path):
    source = tmp_path / "run"
    source.mkdir()

    with pytest.raises(RecoveryBundleError, match="forbidden"):
        create_bundle(source, [private_path], tmp_path / "recovery.zip", 4096)


def test_restore_requires_a_fresh_empty_destination(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    (source / "state.json").write_text("state")
    bundle = tmp_path / "recovery.zip"
    create_bundle(source, ["state.json"], bundle, 4096)
    destination = tmp_path / "restored"
    destination.mkdir()
    (destination / "unrelated.txt").write_text("keep")

    with pytest.raises(RecoveryBundleError, match="not empty"):
        restore_bundle(bundle, destination, 4096)

    assert (destination / "unrelated.txt").read_text() == "keep"


@pytest.mark.parametrize(
    "private_path",
    [
        ".publication/sdk/session.json",
        "tokens",
        "stored_tokens",
        "auth/session.json",
        "credentials/secret",
        "state/TOKENS/value",
    ],
)
def test_private_paths_are_rejected_on_create_and_restore(tmp_path, private_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(RecoveryBundleError, match="forbidden"):
        create_bundle(source, [private_path], tmp_path / "output.zip", 4096)
    bundle = tmp_path / "unsafe.zip"
    _write_manifest_archive(bundle, private_path)
    with pytest.raises(RecoveryBundleError, match="forbidden"):
        restore_bundle(bundle, tmp_path / "restored", 4096)
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize(
    "member",
    [".", "/outside", "a/../outside", "a//outside", "C:/outside", "a\\outside"],
)
def test_restore_rejects_noncanonical_archive_paths(tmp_path, member):
    bundle = tmp_path / "unsafe.zip"
    _write_manifest_archive(bundle, member)
    with pytest.raises(RecoveryBundleError):
        restore_bundle(bundle, tmp_path / "restored", 4096)
    assert not (tmp_path / "restored").exists()


def test_explicit_allowlist_omits_unrelated_and_auth_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "state.json").write_bytes(b"checkpoint")
    (source / "unrelated.bin").write_bytes(b"unrelated")
    (source / "tokens").write_bytes(b"secret")
    bundle = tmp_path / "bundle.zip"
    artifact = create_bundle(source, ["state.json"], bundle, 4096)
    destination = tmp_path / "restored"
    destination.mkdir()
    restore_bundle(bundle, destination, artifact.size_bytes)
    assert sorted(item.name for item in destination.iterdir()) == ["state.json"]
    assert b"secret" not in bundle.read_bytes()
    with pytest.raises(BundleSizeError):
        restore_bundle(bundle, tmp_path / "too-small", artifact.size_bytes - 1)
    assert not (tmp_path / "too-small").exists()


def _rewrite_member(bundle, name, transform):
    with zipfile.ZipFile(bundle) as archive:
        entries = [(info, archive.read(info)) for info in archive.infolist()]
    with zipfile.ZipFile(bundle, "w") as archive:
        for info, content in entries:
            archive.writestr(
                info, transform(content) if info.filename == name else content
            )


@pytest.mark.parametrize(
    "member", ["state.json", ".recovery/manifest.json", ".recovery/manifest.sha256"]
)
def test_restore_rejects_content_tampering_even_with_valid_zip_crc(tmp_path, member):
    bundle = tmp_path / "bundle.zip"
    _write_manifest_archive(bundle, "state.json")
    _rewrite_member(bundle, member, lambda data: bytes([data[0] ^ 1]) + data[1:])
    destination = tmp_path / "restored"
    destination.mkdir()
    with pytest.raises(RecoveryBundleError):
        restore_bundle(bundle, destination, 4096)
    assert list(destination.iterdir()) == []
    assert not list(tmp_path.glob(".restored.restore-*"))


def test_restore_hash_failure_leaves_destination_unpublished(tmp_path, monkeypatch):
    import tennis3d.synthetic.producer_recovery as recovery

    bundle = tmp_path / "bundle.zip"
    _write_manifest_archive(bundle, "state.json")

    def fail_hash(path):
        raise OSError("simulated read failure")

    monkeypatch.setattr(recovery, "_sha256_file", fail_hash)
    with pytest.raises(OSError, match="simulated read failure"):
        restore_bundle(bundle, tmp_path / "restored", 4096)
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".restored.restore-*"))


def test_create_rejects_symlink_parent_and_duplicate_allowlist(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "state.json").write_text("state")
    (source / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RecoveryBundleError, match="real directory"):
        create_bundle(source, ["linked/state.json"], tmp_path / "link.zip", 4096)
    with pytest.raises(RecoveryBundleError, match="duplicate"):
        create_bundle(source, ["a", "a"], tmp_path / "duplicate.zip", 4096)
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("failure", ["open", "fsync"])
def test_bundle_directory_sync_failure_does_not_report_durable_copy(
    tmp_path, monkeypatch, failure
):
    import stat

    from tennis3d.synthetic import producer_recovery

    root = tmp_path / "source"
    root.mkdir()
    (root / "record.json").write_bytes(b"{}")
    destination = tmp_path / "recovery.zip"
    fsync = producer_recovery.os.fsync

    def fail_directory(fd):
        if stat.S_ISDIR(producer_recovery.os.fstat(fd).st_mode):
            raise OSError("directory sync failed")
        fsync(fd)

    monkeypatch.setattr(producer_recovery.os, "fsync", fail_directory)
    if failure == "open":
        open_directory = producer_recovery.os.open

        def fail_open(path, *args, **kwargs):
            if Path(path) == destination.parent:
                raise OSError("directory sync failed")
            return open_directory(path, *args, **kwargs)

        monkeypatch.setattr(producer_recovery.os, "open", fail_open)
    with pytest.raises(OSError, match="directory sync failed"):
        producer_recovery.create_bundle(root, ["record.json"], destination, 1 << 20)
    assert not destination.exists()
    monkeypatch.undo()
    assert (
        producer_recovery.create_bundle(
            root, ["record.json"], destination, 1 << 20
        ).path
        == destination
    )
