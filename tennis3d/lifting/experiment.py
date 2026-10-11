"""Single fresh trial, frozen evaluation and durable comparison outcome."""

import json
import signal
import sys
import tempfile
import time
from pathlib import Path

from tennis3d.artifacts import (
    canonical_json,
    file_sha256,
    json_sha256,
    read_json,
    write_json,
)
from tennis3d.autoresearch.execution import TrialFailure, run_process
from tennis3d.autoresearch.inputs import loss_label
from tennis3d.autoresearch.log import StopReader, append, lock, reconcile, snapshot
from tennis3d.autoresearch.protocol import (
    Protocol,
    checkout,
    load_protocol,
    verify_sources,
)
from tennis3d.autoresearch.segments import (
    archive_candidate,
    create_segment,
    durable_directory,
    publish_best,
    read_segment,
    safe_name,
)


def child(stage, request, root, fd, deadline, stopped, update):
    run_process(
        [sys.executable, "-m", "tennis3d.autoresearch.worker", stage, str(request)],
        cwd=root,
        output=request.parent / "process.log",
        deadline=deadline,
        lock_fd=fd,
        stopped=stopped,
        update=update,
    )


def initialize(args):
    root, run = checkout(), args.run.resolve()
    durable_directory(root, run)
    protocol = load_protocol(args.config)
    with lock(run / ".trial.lock", blocking=False) as fd:
        reconcile(run)
        temporary = tempfile.mkdtemp(prefix=".prepare-", dir=run)
        request = Path(temporary) / "request.json"
        write_json(
            request,
            {"protocol": protocol.model_dump(mode="json"), "root": str(root)},
        )
        child(
            "prepare",
            request,
            root,
            fd,
            time.perf_counter() + max(30, protocol.total_seconds),
            lambda: False,
            lambda: None,
        )
        prepared = read_json(request.parent / "prepared.json")
        segment = create_segment(
            root,
            run,
            args.segment,
            protocol,
            data=prepared["data"],
            hardware=prepared["hardware"],
            training=prepared["training"],
            loss_label=loss_label(root, args.loss_label),
        )
        print(
            json.dumps(
                {
                    "segment": segment["segment"],
                    "profile": protocol.profile,
                    "sha256": segment["sha256"],
                }
            )
        )
    return 0


def trial(args):
    root, output = checkout(), args.output.resolve()
    run, trial_id = output.parent, safe_name(output.name)
    durable_directory(root, run)
    with lock(run / ".trial.lock", blocking=False) as fd:
        reconcile(run)
        publish_best(run)
        state = snapshot(run)
        if state["stopped"]:
            raise ValueError(
                "search remains stopped; explicit experiment-state resume is required"
            )
        segment = read_segment(run, args.segment)
        protocol = Protocol.model_validate_json(canonical_json(segment["protocol"]))
        if args.config and load_protocol(args.config) != protocol:
            raise ValueError("trial protocol differs from the initialized segment")
        if any(t["trial"] == trial_id for t in state["trials"]):
            raise ValueError("trial ID already appears in the log")
        stop_reader = StopReader(run)
        output.mkdir(exist_ok=False)
        started = time.perf_counter()
        deadline = started + protocol.total_seconds
        append(
            run,
            "started",
            segment=segment["segment"],
            trial=trial_id,
            hypothesis=args.hypothesis,
            baseline=args.baseline,
            phase="setup",
            profile=protocol.profile,
        )
        interrupted = False
        old_signals = {}

        def stop_signal(signum, frame):
            nonlocal interrupted
            interrupted = True

        for signum in (signal.SIGINT, signal.SIGTERM):
            old_signals[signum] = signal.signal(signum, stop_signal)

        def deadline_signal(signum, frame):
            raise TrialFailure(
                "timed-out",
                "total deadline exceeded in supervisor setup or final checks",
            )

        old_signals[signal.SIGALRM] = signal.signal(signal.SIGALRM, deadline_signal)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.perf_counter()))
        status, score, reason = "crashed", None, "supervisor interrupted"
        fields = {"diagnostic": args.baseline == "small"}
        try:
            identity = verify_sources(root, segment["frozen_files"])
            label = loss_label(root, args.loss_label)
            training = segment["training"]
            fields = {
                "revision": identity["revision"],
                "candidate_sha256": identity["candidate_sha256"],
                "config_sha256": json_sha256(training),
                "loss_label": label,
                "diagnostic": args.baseline == "small",
            }
            append(
                run,
                "progress",
                segment=segment["segment"],
                trial=trial_id,
                phase="setup",
                **fields,
            )
            archive_candidate(root, output, identity)
            request = output / "request.json"
            payload = {
                **segment,
                **identity,
                "protocol": protocol.model_dump(mode="json"),
                "root": str(root),
                "training": training,
                "baseline": args.baseline,
                "deadline": deadline,
            }
            write_json(request, payload)
            previous = 0.0
            stage = "setup"

            last_stop_check, last_stopped = 0.0, False

            def stopped(*, force=False):
                nonlocal last_stop_check, last_stopped
                if interrupted:
                    return True
                now = time.perf_counter()
                if force or now - last_stop_check >= protocol.loss_interval_seconds:
                    last_stop_check, last_stopped = now, stop_reader.poll()
                return last_stopped

            def update():
                nonlocal previous
                now = time.perf_counter()
                if now - previous < protocol.loss_interval_seconds:
                    return
                previous = now
                progress = (
                    read_json(output / "progress.json")
                    if stage == "train" and (output / "progress.json").exists()
                    else {}
                )
                append(
                    run,
                    "progress",
                    segment=segment["segment"],
                    trial=trial_id,
                    phase=progress.get("phase", stage),
                    total_seconds=now - started,
                    **{
                        k: progress[k]
                        for k in ("train_seconds", "step", "loss")
                        if k in progress
                    },
                )

            for stage in ("resolve", "train", "evaluate", "checks"):
                append(
                    run,
                    "progress",
                    segment=segment["segment"],
                    trial=trial_id,
                    phase=stage,
                )
                child(stage, request, root, fd, deadline, stopped, update)
                if stage == "resolve":
                    training = read_json(output / "config.resolved.json")
                    if (
                        args.baseline == "reference"
                        and identity["candidate_sha256"]
                        != segment["reference"]["candidate_sha256"]
                    ):
                        raise ValueError(
                            "reference baseline must use the unchanged initialized candidate"
                        )
                    if args.baseline == "small":
                        expected = {
                            **segment["training"],
                            "model": {
                                **segment["training"]["model"],
                                "width": 64,
                                "depth": 4,
                            },
                        }
                        if training != expected or any(
                            identity["candidate_files"][p]
                            != segment["reference"]["candidate_files"][p]
                            for p in identity["candidate_files"]
                            if p != "configs/lifter.yaml"
                        ):
                            raise ValueError(
                                "small baseline must change only reference width/depth to 64/4"
                            )
                    fields["config_sha256"] = json_sha256(training)
                    payload["training"] = training
                    write_json(request, payload, overwrite=True)
            verify_sources(root, segment["frozen_files"], identity["candidate_files"])
            if stopped(force=True):
                raise TrialFailure(
                    "interrupted", "search stopped before result publication"
                )
            if time.perf_counter() >= deadline:
                raise TrialFailure(
                    "timed-out", "total deadline exceeded before result publication"
                )
            checks = read_json(output / "checks.json")
            if checks.get("passed") is not True:
                raise ValueError("fixed checks failed")
            metrics = read_json(output / "metrics.json")
            score = metrics["modes"]["corrupted"]["val_xyz_m"]
            observed = read_json(output / "observed.json")
            summary = read_json(output / "training/summary.json")
            fields.update(
                observed,
                windows=observed["batches"] * training["batch_size"],
                checks_passed=True,
                parameter_count=checks["parameter_count"],
                peak_allocated_bytes=summary["peak_allocated_bytes"],
                checkpoint_sha256=metrics["checkpoint_sha256"],
                metrics_sha256=file_sha256(output / "metrics.json"),
            )
            incumbent = snapshot(run)["best"].get(segment["segment"])
            diagnostic = args.baseline == "small"
            status = (
                "kept"
                if not diagnostic and (incumbent is None or score < incumbent["score"])
                else "discarded"
            )
            reason = (
                "diagnostic baseline below the parameter floor; never retained"
                if diagnostic
                else "first passing baseline"
                if incumbent is None
                else "improved retained score"
                if status == "kept"
                else "did not improve retained score"
            )
        except TrialFailure as exc:
            status, reason = exc.status, str(exc)
            if (output / "failure.json").exists() and status == "crashed":
                failure = read_json(output / "failure.json")
                reason = f"{failure['stage']}: {failure['reason']}"
                if failure["type"] == "ValueError" or failure["stage"] in {
                    "evaluate",
                    "checks",
                }:
                    status = "checks-failed"
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
            status, reason = "checks-failed", str(exc)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for signum, handler in old_signals.items():
                signal.signal(signum, handler)
            result = append(
                run,
                "finished",
                segment=segment["segment"],
                trial=trial_id,
                status=status,
                phase="complete",
                score=score if status in {"kept", "discarded"} else None,
                reason=reason,
                total_seconds=time.perf_counter() - started,
                **fields,
            )
            write_json(output / "result.json", result)
            publish_best(run)
        print(json.dumps(result, indent=2))
        return 0 if status in {"kept", "discarded"} else 2


def command(args):
    try:
        return initialize(args) if args.command == "experiment-init" else trial(args)
    except (ValueError, OSError, RuntimeError, TrialFailure) as exc:
        print(f"experiment: {exc}")
        return 2


def register_parser(commands):
    init = commands.add_parser(
        "experiment-init", help="freeze a reviewed data/runtime comparison segment"
    )
    init.add_argument("--run", type=Path, required=True)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--segment", required=True)
    init.add_argument("--loss-label")
    init.set_defaults(handler=command)
    parser = commands.add_parser(
        "experiment", help="run one fresh timed trial and fixed final EMA evaluation"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--segment")
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--baseline", choices=("reference", "small"))
    parser.add_argument("--loss-label")
    parser.set_defaults(handler=command)
