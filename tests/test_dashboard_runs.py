"""Read-only monitoring across independent campaign directories."""

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager

import pytest

from tennis3d.artifacts import write_json
from tennis3d.autoresearch.dashboard import make_server
from tennis3d.autoresearch.log import EventTail, append


def campaign(run, segment, loss):
    append(run, "segment", segment=segment, profile="standard")
    for trial, score in [("reference-001", 3), ("best-001", 2)]:
        append(run, "started", segment=segment, trial=trial, loss_label="Fixture loss")
        append(
            run,
            "finished",
            segment=segment,
            trial=trial,
            status="kept",
            score=score + loss,
            checks_passed=True,
        )
        directory = run / trial
        directory.mkdir()
        (directory / "curve.jsonl").write_text(
            json.dumps({"loss": loss, "train_seconds": 1}) + "\n"
        )
        write_json(directory / "metrics.json", {"source_loss": loss})


@contextmanager
def dashboard(run, archives):
    with make_server(run, 0, archive_runs=archives) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def get(base, endpoint, **query):
    url = base + endpoint
    if query:
        url += "?" + urllib.parse.urlencode(query)
    with urllib.request.urlopen(url, timeout=2) as response:
        return json.load(response)


@pytest.mark.parametrize("same_segment_name", [False, True])
def test_dashboard_keeps_duplicate_trial_ids_and_archive_status_separate(
    tmp_path, same_segment_name
):
    current, archive = tmp_path / "current", tmp_path / "archive"
    campaign(current, "current", 1)
    campaign(archive, "current" if same_segment_name else "old", 10)
    append(archive, "control", action="stop")
    archive_update = json.loads(
        (archive / "experiments.jsonl").read_text().splitlines()[-1]
    )["time"]
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with dashboard(current, [archive, archive / ".", current]) as base:
        state = get(base, "/api/snapshot")
        assert len(state["segments"]) == 2
        old, live = state["segments"]
        assert old["archived"] and old["stopped"]
        assert old["last_update"] == archive_update
        assert not live["archived"] and not state["stopped"]
        assert live["segment"] == "current"
        assert old["segment"] != live["segment"]
        assert old["segment_name"] == ("current" if same_segment_name else "old")
        assert len(state["trials"]) == 4
        assert state["best"][old["segment"]]["score"] == 12
        assert state["best"][live["segment"]]["score"] == 3
        for segment, expected in [(old["segment"], 10), (live["segment"], 1)]:
            for trial in ["reference-001", state["best"][segment]["trial"]]:
                details = get(base, "/api/trial", segment=segment, id=trial)
                assert details["trial"]["segment"] == segment
                assert details["curve"][0]["loss"] == expected
                assert details["metrics"]["source_loss"] == expected
        with pytest.raises(urllib.error.HTTPError) as error:
            get(base, "/api/trial", id="reference-001")
        assert error.value.code == 422
        assert "ambiguous trial ID" in json.load(error.value)["error"]
        with pytest.raises(urllib.error.HTTPError) as error:
            get(base, "/api/trial", segment="missing", id="reference-001")
        assert error.value.code == 422
        assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        append(current, "started", segment="current", trial="unique-current")
        assert (
            get(base, "/api/trial", id="unique-current")["trial"]["segment"]
            == "current"
        )
        append(archive, "started", segment=old["segment_name"], trial="unique-archive")
        assert get(base, "/api/trial", id="unique-archive")["trial"]["archived"]
        append(current, "control", action="stop")
        assert get(base, "/api/snapshot")["stopped"]
        append(current, "control", action="resume")
        assert not get(base, "/api/snapshot")["stopped"]


def test_archive_keys_survive_argument_order_and_server_restart(tmp_path):
    current, first, second = [
        tmp_path / name for name in ["current", "first", "second"]
    ]
    for run in [current, first, second]:
        campaign(run, "shared", 1)
    with dashboard(current, [first, second]) as base:
        keys = {
            s["source_run"]: s["segment"]
            for s in get(base, "/api/snapshot")["segments"]
        }
    with dashboard(current, [second, first]) as base:
        assert keys == {
            s["source_run"]: s["segment"]
            for s in get(base, "/api/snapshot")["segments"]
        }


def test_archived_trial_symlink_cannot_read_outside_its_run(tmp_path):
    current, archive = tmp_path / "current", tmp_path / "archive"
    campaign(current, "current", 1)
    campaign(archive, "old", 10)
    directory = archive / "reference-001"
    for artifact in directory.iterdir():
        artifact.unlink()
    directory.rmdir()
    directory.symlink_to(current / "reference-001", target_is_directory=True)
    with dashboard(current, [archive]) as base:
        segment = get(base, "/api/snapshot")["segments"][0]["segment"]
        with pytest.raises(urllib.error.HTTPError) as error:
            get(base, "/api/trial", segment=segment, id="reference-001")
        assert error.value.code == 422
        assert "inside this run" in json.load(error.value)["error"]


@pytest.mark.parametrize("fault", ["corruption", "orphan-journal", "io-error"])
def test_unavailable_archive_does_not_block_current_history(
    tmp_path, monkeypatch, fault
):
    current, archive = tmp_path / "current", tmp_path / "archive"
    campaign(current, "current", 1)
    campaign(archive, "old", 10)
    log = archive / "experiments.jsonl"
    original = log.read_bytes()
    journal = archive / "experiments.jsonl.recovery.json"
    if fault == "corruption":
        log.write_bytes(b"bad\n")
    elif fault == "orphan-journal":
        log.unlink()
        journal.write_text("{}")
    else:
        poll = EventTail.poll

        def unreadable(tail, **kwargs):
            if tail.path == log:
                raise OSError("fixture archive I/O failure")
            return poll(tail, **kwargs)

        monkeypatch.setattr(EventTail, "poll", unreadable)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with dashboard(current, [archive]) as base:
        state = get(base, "/api/snapshot")
        assert [s["segment"] for s in state["segments"]] == ["current"]
        assert len(state["trials"]) == 2
        assert state["best"]["current"]["trial"] == "best-001"
        assert len(state["warnings"]) == 1
        assert str(archive) not in state["warnings"][0]
        assert "Archive unavailable" in state["warnings"][0]
        assert (
            get(base, "/api/trial", id="reference-001")["metrics"]["source_loss"] == 1
        )
        append(current, "started", segment="current", trial="current-live")
        assert get(base, "/api/snapshot")["trials"][-1]["status"] == "running"
        # A later readable archive gets its independent projection back.
        if fault == "io-error":
            monkeypatch.setattr(EventTail, "poll", poll)
        else:
            assert all(
                p.read_bytes() == data
                for p, data in before.items()
                if p.parent == archive
            )
            log.write_bytes(original)
            journal.unlink(missing_ok=True)
        restored = get(base, "/api/snapshot")
        assert len(restored["segments"]) == 2
        assert not restored["warnings"]
        segment = restored["segments"][0]["segment"]
        assert (
            get(base, "/api/trial", segment=segment, id="reference-001")["metrics"][
                "source_loss"
            ]
            == 10
        )
        # Current corruption still rejects the snapshot rather than hiding it.
        (current / "experiments.jsonl").write_bytes(b"bad\n")
        with pytest.raises(urllib.error.HTTPError) as error:
            get(base, "/api/snapshot")
        assert error.value.code == 422


def test_missing_archive_without_journal_is_an_empty_history(tmp_path):
    current = tmp_path / "current"
    campaign(current, "current", 1)
    with dashboard(current, [tmp_path / "missing"]) as base:
        state = get(base, "/api/snapshot")
        assert [s["segment"] for s in state["segments"]] == ["current"]
        assert not state["warnings"]
