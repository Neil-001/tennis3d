"""Actual GET server tests for mixed campaign types and fail-closed decisions."""

import json
import threading
import urllib.error
from contextlib import contextmanager

import pytest
from test_dashboard_runs import campaign, get
from throughput_dashboard_fixture import create

from tennis3d.autoresearch.dashboard import make_server
from tennis3d.autoresearch.log import append
from tennis3d.autoresearch.throughput_dashboard import canonical, digest


@contextmanager
def serve(run, archive):
    with make_server(run, 0, kind="throughput", archive_runs=[archive]) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def files(run):
    return {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}


def test_mixed_read_only_snapshot_and_truthful_retention(tmp_path, monkeypatch):
    run, archive = tmp_path / "throughput", tmp_path / "accuracy"
    create(run)
    campaign(archive, "old", 10)
    append(archive, "started", segment="old", trial="000001")
    before = files(tmp_path)
    # HTTP must not enter any execution, lock-writing or readiness path.
    from tennis3d.throughput import campaign as execution

    def forbidden(*args, **kwargs):
        pytest.fail("HTTP entered a campaign mutation/execution path")

    for name in (
        "_lock",
        "_state",
        "readiness",
        "initialize",
        "trial",
        "record",
        "decide",
    ):
        monkeypatch.setattr(execution, name, forbidden)
    with serve(run, archive) as base:
        state = get(base, "/api/snapshot")
        old, new = state["segments"]
        assert [s["kind"] for s in state["segments"]] == ["accuracy", "throughput"]
        assert new["data"]["steps"] == 128
        assert new["best"]["trial_id"] == "000001"
        assert new["qualification"]["cold"][0] == [100, 101]
        rows = [t for t in state["trials"] if t["kind"] == "throughput"]
        assert [t["status"] for t in rows] == ["kept", "discarded"]
        assert (
            rows[1]["measurements"][0]["candidate_elapsed_s"]
            < rows[0]["measurements"][0]["candidate_elapsed_s"]
        )
        assert state["best"][new["segment"]]["trial"] == "000001", (
            "never infer the incumbent from absolute timings"
        )
        assert not isinstance(rows[0].get("score"), (int, float))
        assert rows[0]["ratios"]["elapsed_s"]["geometric_ratio"] == pytest.approx(0.9)
        details = get(base, "/api/trial", segment=new["segment"], id="000001")
        assert details["diagnostics"][0]["candidate"]["stall_p95_s"] == 0.03
        assert details["diagnostics"][0]["candidate"][
            "windows_per_second"
        ] == pytest.approx(6144 / 90)
        assert (
            get(base, "/api/trial", segment=old["segment"], id="000001")["trial"][
                "kind"
            ]
            == "accuracy"
        )
        with pytest.raises(urllib.error.HTTPError):
            get(base, "/api/trial", id="000001")
        assert str(tmp_path) not in json.dumps(state)
        assert "private_path" not in json.dumps(details)
        for _ in range(3):
            get(base, "/api/snapshot")
        assert files(tmp_path) == before


@pytest.mark.parametrize(
    "fault", ["torn", "pointer", "result", "evidence", "cursor", "missing", "malformed"]
)
def test_incomplete_mirror_keeps_last_atomic_snapshot_and_other_campaign(
    tmp_path, fault
):
    run, archive = tmp_path / "throughput", tmp_path / "accuracy"
    create(run)
    campaign(archive, "old", 10)
    with serve(run, archive) as base:
        original = get(base, "/api/snapshot")
        if fault == "torn":
            with (run / "journal.jsonl").open("ab") as stream:
                stream.write(b'{"kind":')
        elif fault == "pointer":
            (run / "best.json").write_text("{}")
        elif fault in ("result", "evidence"):
            (run / "trials/000001" / f"{fault}.json").write_text("{}")
        elif fault == "cursor":
            (run / "cursor.json").write_text("{}")
        elif fault == "missing":
            (run / "protocol.json").unlink()
        else:
            (run / "protocol.json").write_text("[]")
        before = files(tmp_path)
        state = get(base, "/api/snapshot")
        assert state["segments"][-1]["status"] == "stale/incomplete"
        assert state["best"] == original["best"]
        assert state["trials"] == original["trials"]
        assert state["warnings"]
        assert files(tmp_path) == before
    # Cold start cannot label a partially mirrored retained trial as kept.
    with serve(run, archive) as base:
        state = get(base, "/api/snapshot")
        assert state["segments"][-1]["status"] == "incomplete"
        assert all(t["kind"] == "accuracy" for t in state["trials"])


def test_absent_throughput_source_has_explicit_kind_without_initializing(tmp_path):
    run, archive = tmp_path / "missing", tmp_path / "accuracy"
    campaign(archive, "old", 10)
    with serve(run, archive) as base:
        state = get(base, "/api/snapshot")
        assert state["segments"][-1]["kind"] == "throughput"
        assert state["segments"][-1]["status"] == "not initialized"
        assert not run.exists()


def test_invalid_retention_gate_is_rejected_even_with_consistent_hashes(tmp_path):
    run, archive = tmp_path / "throughput", tmp_path / "accuracy"
    entries = create(run)
    campaign(archive, "old", 10)
    path = run / "trials/000001/evidence.json"
    evidence = json.loads(path.read_bytes())
    evidence["passed"] = False
    path.write_bytes(canonical(evidence))
    path = run / "trials/000001/result.json"
    result = json.loads(path.read_bytes())
    result["gates"].update(passed=False, evidence_sha256=digest(canonical(evidence)))
    path.write_bytes(canonical(result))
    entries[2]["payload"]["result_sha256"] = digest(canonical(result))
    for i, entry in enumerate(entries):
        entry.pop("sha256")
        entry["previous"] = entries[i - 1]["sha256"] if i else "0" * 64
        entry["sha256"] = digest(canonical(entry))
    (run / "journal.jsonl").write_bytes(b"".join(canonical(e) for e in entries))
    (run / "cursor.json").write_bytes(
        canonical({"sequence": len(entries) - 1, "sha256": entries[-1]["sha256"]})
    )
    with serve(run, archive) as base:
        state = get(base, "/api/snapshot")
        assert state["segments"][-1]["status"] == "incomplete"
        assert not any(t["kind"] == "throughput" for t in state["trials"])


def reseal(run, entries, identifier, result, evidence):
    """Adversarial fixture envelopes remain hash-consistent for semantic checks."""
    directory = run / "trials" / identifier
    (directory / "evidence.json").write_bytes(canonical(evidence))
    result["gates"]["evidence_sha256"] = digest(canonical(evidence))
    (directory / "result.json").write_bytes(canonical(result))
    for entry in entries:
        if entry["kind"] == "record" and entry["payload"]["trial_id"] == identifier:
            entry["payload"]["result_sha256"] = digest(canonical(result))
    for i, entry in enumerate(entries):
        entry.pop("sha256")
        entry["previous"] = entries[i - 1]["sha256"] if i else "0" * 64
        entry["sha256"] = digest(canonical(entry))
    (run / "journal.jsonl").write_bytes(b"".join(canonical(e) for e in entries))
    (run / "cursor.json").write_bytes(
        canonical({"sequence": len(entries) - 1, "sha256": entries[-1]["sha256"]})
    )


@pytest.mark.parametrize(
    "fault",
    [
        "candidate_elapsed_s",
        "candidate_stream_s",
        "candidate_sustained_s",
        "incumbent_elapsed_s",
        "incumbent_stream_s",
        "incumbent_sustained_s",
        "score",
        "evidence_timing",
    ],
)
def test_discarded_result_requires_complete_metrics_and_exact_sealed_score(
    tmp_path, fault
):
    run, archive = tmp_path / "throughput", tmp_path / "accuracy"
    entries = create(run)
    campaign(archive, "old", 10)
    result = json.loads((run / "trials/000002/result.json").read_bytes())
    evidence = json.loads((run / "trials/000002/evidence.json").read_bytes())
    if fault == "score":
        result["score"].update(metric="accuracy_m", geometric_stream_ratio=999)
    elif fault == "evidence_timing":
        evidence["reports"][0]["candidate"]["stream_elapsed_s"] *= 2
    else:
        result["measurements"][0].pop(fault)
    reseal(run, entries, "000002", result, evidence)
    with serve(run, archive) as base:
        state = get(base, "/api/snapshot")
        assert state["segments"][-1]["status"] == "incomplete"
        assert state["warnings"]


def test_active_reservation_ignores_unsealed_outputs_and_shows_description(tmp_path):
    run, archive = tmp_path / "throughput", tmp_path / "accuracy"
    entries = create(run)
    campaign(archive, "old", 10)
    best = json.loads((run / "best.json").read_bytes())
    reservation = {
        "trial_id": "000003",
        "source_sha256": "3" * 64,
        "incumbent_sha256": best["source_sha256"],
        "job_budget_s": 3600,
        "description": "Decode only bounded row groups",
        "reserved_at": "2026-10-06T16:00:00Z",
    }
    directory = run / "trials/000003"
    directory.mkdir()
    (directory / "reservation.json").write_bytes(canonical(reservation))
    (directory / "measurement.json").write_text('{"status":"completed","score":0.001}')
    entry = {
        "sequence": len(entries),
        "previous": entries[-1]["sha256"],
        "kind": "trial",
        "payload": reservation,
    }
    entry["sha256"] = digest(canonical(entry))
    with (run / "journal.jsonl").open("ab") as stream:
        stream.write(canonical(entry))
    (run / "cursor.json").write_bytes(
        canonical({"sequence": entry["sequence"], "sha256": entry["sha256"]})
    )
    before = files(tmp_path)
    with serve(run, archive) as base:
        state = get(base, "/api/snapshot")
        assert state["segments"][-1]["status"] == "active reservation"
        assert state["segments"][-1]["active_trial"] == "000003"
        trial = get(
            base, "/api/trial", segment=state["segments"][-1]["segment"], id="000003"
        )["trial"]
        assert trial["status"] == "active"
        assert trial["hypothesis"] == reservation["description"]
        assert trial["reserved_at"] == reservation["reserved_at"]
        assert not trial["measurements"]
        assert files(tmp_path) == before


def test_failed_record_is_failed_even_after_discard(tmp_path):
    from tennis3d.throughput.campaign import _score

    run, archive = tmp_path / "throughput", tmp_path / "accuracy"
    entries = create(run)
    campaign(archive, "old", 10)
    result = json.loads((run / "trials/000002/result.json").read_bytes())
    evidence = json.loads((run / "trials/000002/evidence.json").read_bytes())
    result.update(status="failed", measurements=[])
    result["gates"]["passed"] = False
    evidence["passed"] = False
    result["score"] = _score(json.loads((run / "protocol.json").read_bytes()), result)
    reseal(run, entries, "000002", result, evidence)
    with serve(run, archive) as base:
        state = get(base, "/api/snapshot")
        trial = state["trials"][-1]
        assert trial["status"] == "failed"
        assert trial["decision"] == "discard"
        assert not trial["gates"]["passed"]
        assert state["best"][trial["segment"]]["trial"] == "000001"


def test_polling_does_not_traverse_unchanged_frozen_sampler_evidence(
    tmp_path, monkeypatch
):
    from tennis3d.autoresearch import throughput_dashboard as reader

    run, archive = tmp_path / "throughput", tmp_path / "accuracy"
    create(run)
    campaign(archive, "old", 10)
    with serve(run, archive) as base:
        assert not get(base, "/api/snapshot")["warnings"]
        serialize = reader.canonical

        def small_envelopes_only(value):
            assert "benchmark" not in value, (
                "polling traversed the full frozen protocol again"
            )
            return serialize(value)

        monkeypatch.setattr(reader, "canonical", small_envelopes_only)
        for _ in range(3):
            assert not get(base, "/api/snapshot")["warnings"]
