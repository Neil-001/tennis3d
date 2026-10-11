"""Fabricated browser-test history. These values are not training measurements."""

import sys
import time
from pathlib import Path
from unittest.mock import patch

from tennis3d.artifacts import write_json
from tennis3d.autoresearch.log import append

run, phase = Path(sys.argv[1]), sys.argv[2]
if phase in {"multirun-current", "multirun-archive"}:
    archived = phase == "multirun-archive"
    offset = 10 if archived else 0
    append(run, "segment", segment="fixture-shared", profile="standard")
    for trial, loss in [("reference-001", 3 + offset), ("best-001", 2 + offset)]:
        append(
            run,
            "started",
            segment="fixture-shared",
            trial=trial,
            loss_label="Fixture loss (dimensionless)",
        )
        append(
            run,
            "finished",
            segment="fixture-shared",
            trial=trial,
            status="kept",
            score=loss,
            checks_passed=True,
            reason="Archived fixture" if archived else "Current fixture",
        )
        directory = run / trial
        directory.mkdir()
        (directory / "curve.jsonl").write_text(f'{{"loss":{loss},"train_seconds":1}}\n')
        write_json(directory / "metrics.json", {"campaign": phase})
    with patch(
        "tennis3d.autoresearch.log.time.time",
        return_value=time.time() - (60 if archived else 0),
    ):
        append(
            run,
            "started",
            segment="fixture-shared",
            trial="archive-live" if archived else "current-live",
            phase="training",
        )
    if archived:
        append(run, "control", action="stop")
elif phase == "initial":
    append(
        run,
        "segment",
        segment="fixture-standard",
        profile="standard",
        hardware={"device": "fixture CPU"},
    )
    for index, score, status in [
        (1, 3.0, "kept"),
        (2, 2.5, "kept"),
        (3, None, "checks-failed"),
        (4, 2.8, "discarded"),
    ]:
        trial = f"fixture-{index}"
        append(
            run,
            "started",
            segment="fixture-standard",
            trial=trial,
            hypothesis=f"Fabricated browser fixture {index}",
            loss_label="Fixture squared Euclidean position loss, XYZ divided by 10 m (dimensionless)",
            baseline="reference" if index == 1 else None,
        )
        append(
            run,
            "finished",
            segment="fixture-standard",
            trial=trial,
            status=status,
            score=score,
            checks_passed=score is not None,
            reason="Deliberate fixture failure" if score is None else "fixture score",
            train_seconds=300,
            total_seconds=360,
            steps=100,
        )
        directory = run / trial
        directory.mkdir()
        (directory / "curve.jsonl").write_text(
            "\n".join(
                f'{{"step":{step},"loss":{index / step},"train_seconds":{step * index}}}'
                for step in range(1, 101)
            )
            + "\n"
        )
        write_json(directory / "metrics.json", {"fixture": True, "score": score})
    append(
        run,
        "segment",
        segment="fixture-smoke",
        profile="smoke",
        hardware={"device": "fixture CPU"},
    )
    append(
        run,
        "started",
        segment="fixture-smoke",
        trial="fixture-live",
        hypothesis="Live fixture",
        loss_label="Fixture squared Euclidean position loss, XYZ divided by 10 m (dimensionless)",
    )
    (run / "fixture-live").mkdir()
elif phase == "records":
    append(run, "segment", segment="fixture-records", profile="standard")
    for index, score, status, baseline, diagnostic in [
        (1, 0.1, "discarded", "small", True),
        (2, 0.2, "discarded", None, False),
        (3, 3.0, "kept", "reference", False),
        (4, 3.0, "discarded", None, False),
        (5, 2.5, "kept", None, False),
        (6, 2.8, "discarded", "reference", False),
        (7, 0.01, "discarded", "small", False),
        (8, 0.01, "discarded", None, True),
        (9, None, "checks-failed", None, False),
        (10, None, "running", None, False),
    ]:
        trial = f"fixture-record-{index}"
        append(
            run,
            "started",
            segment="fixture-records",
            trial=trial,
            baseline=baseline,
            diagnostic=diagnostic,
        )
        if status != "running":
            append(
                run,
                "finished",
                segment="fixture-records",
                trial=trial,
                score=score,
                status=status,
                checks_passed=score is not None,
                reason="<img src=x onerror=alert(1)>"
                if score is None
                else "fixture score",
            )
elif phase == "records-finished":
    append(
        run,
        "finished",
        segment="fixture-records",
        trial="fixture-record-10",
        status="interrupted",
        score=None,
    )
elif phase == "best-switch":
    append(
        run,
        "started",
        segment="fixture-standard",
        trial="fixture-5",
        loss_label="Fixture squared Euclidean position loss, XYZ divided by 10 m (dimensionless)",
    )
    append(
        run,
        "finished",
        segment="fixture-standard",
        trial="fixture-5",
        status="kept",
        score=2.0,
        checks_passed=True,
    )
    (run / "fixture-5").mkdir()
    (run / "fixture-5" / "curve.jsonl").write_text(
        '{"step":1,"loss":0.5,"train_seconds":1}\n{"step":2,"loss":0.25,"train_seconds":2}\n'
    )
elif phase == "live":
    append(
        run,
        "progress",
        segment="fixture-smoke",
        trial="fixture-live",
        phase="training",
        steps=2,
    )
    (run / "fixture-live" / "curve.jsonl").write_text(
        '{"step":1,"loss":0.8,"train_seconds":1}\n{"step":2,"loss":0.4,"train_seconds":2}\n'
    )
elif phase == "stale":
    with patch("tennis3d.autoresearch.log.time.time", return_value=time.time() - 60):
        append(
            run,
            "progress",
            segment="fixture-smoke",
            trial="fixture-live",
            phase="evaluation",
        )
elif phase == "interrupted":
    append(
        run,
        "finished",
        segment="fixture-smoke",
        trial="fixture-live",
        status="interrupted",
        score=None,
        reason="Fixture interruption",
    )
    append(run, "control", action="stop")
