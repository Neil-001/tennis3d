"""Isolated inference process. Parent owns cancellation and process lifecycle."""

import contextlib
import os
import sys
import time
from pathlib import Path

from tennis3d.artifacts import read_json, write_json
from tennis3d.config import load_config
from tennis3d.inference.settings import ReconstructionConfig
from tennis3d.perception.assets import AssetIntegrityError, MissingAssetError


def discard_uncommitted(run):
    """Retain completed stages; remove only this app's unfinished stage outputs."""
    import shutil

    if not (run / "detect.json").exists():
        shutil.rmtree(run, ignore_errors=True)
        return
    for stage, files in {
        "calibrate": ("calibrate.config.json", "cameras.npz"),
        "lift": ("lift.config.json", "predictions_raw.npz"),
        "export": ("export.config.json", "result.json", "metrics.json"),
    }.items():
        if not (run / f"{stage}.json").exists():
            for name in files:
                (run / name).unlink(missing_ok=True)


SUMMARY_FIELDS = (
    "status",
    "exit_code",
    "frame_count",
    "coverage",
    "failure_reasons",
    "error",
    "reused_stages",
    "invocation_seconds",
    "invocation_timings",
)


def main(directory):
    directory = Path(directory)
    request = read_json(directory / "request.json")
    progress_path = directory / "progress.json"
    last_update = 0.0
    frames = None

    def progress(stage, completed, total, unit):
        nonlocal last_update
        now = time.monotonic()
        if stage == "detect" and total is None:
            total = frames
        if now - last_update >= 0.15 or completed == 0 or completed == total:
            # Windows refuses the replace while the server is reading the file;
            # the next update carries the same information.
            with contextlib.suppress(PermissionError):
                write_json(
                    progress_path,
                    {
                        "stage": stage,
                        "progress": {
                            "completed": completed,
                            "total": total,
                            "unit": unit,
                        },
                    },
                    overwrite=True,
                )
            last_update = now

    try:
        from tennis3d.app.media import playback_copy
        from tennis3d.inference.pipeline import reconstruct

        config = load_config(ReconstructionConfig, overrides=request["config"])
        source, run = directory / request["source_name"], directory / "inference"
        try:
            # The copy lets the video play while detection runs, and its exact
            # frame count scales detection progress.
            frames = len(
                playback_copy(source, directory / "playback.mp4", progress=progress)[
                    "frames"
                ]
            )
        except Exception as error:  # noqa: BLE001
            # Reconstruction still reports why an undecodable video failed.
            print(f"playback copy unavailable: {error}", flush=True)
        discard_uncommitted(run)
        try:
            summary = reconstruct(source, run, config, progress=progress)
        except ValueError as error:
            accelerator = sys.modules.get("tennis3d.perception.accelerate")
            if accelerator is not None and accelerator.STALLED:
                raise RuntimeError(f"Intel GPU detection stalled: {error}") from error
            if not (run / "detect.json").exists():
                raise
            # Stages saved by another app version or settings are not reusable;
            # this run owns its directory, so compute them again.
            summary = reconstruct(
                source, run, config, overwrite=True, progress=progress
            )
        write_json(
            directory / "outcome.json",
            {
                "summary": {k: summary[k] for k in SUMMARY_FIELDS if k in summary},
                # An undecodable video ends without writing a result of its own.
                "produced_result": "result" in summary,
            },
            overwrite=True,
        )
        return summary["exit_code"]
    except Exception as error:  # noqa: BLE001 - process boundary
        unavailable = (
            ImportError,
            RuntimeError,
            MissingAssetError,
            AssetIntegrityError,
        )
        code = 3 if isinstance(error, unavailable) else 2
        write_json(
            directory / "outcome.json",
            {"error": str(error), "error_type": type(error).__name__},
            overwrite=True,
        )
        return code


def process_main(directory):
    code = main(directory)
    accelerator = sys.modules.get("tennis3d.perception.accelerate")
    if accelerator is not None and accelerator.STALLED:
        # Reconstruction may have exported the decoded prefix. Keep those
        # bytes viewable, but a stalled detector still makes this attempt fail.
        path = Path(directory) / "outcome.json"
        try:
            outcome = read_json(path)
            summary = outcome.get("summary") or {}
            outcome["summary"] = summary | {"status": "failed", "exit_code": 3}
            outcome["detector_stalled"] = True
            outcome.setdefault("error", "Intel GPU detection stalled.")
            write_json(path, outcome, overwrite=True)
        except (OSError, ValueError):
            pass  # The parent can recover the committed detection failure.
        # Normal native-request destruction
        # may block on the stalled device, as in the CLI entry point.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except OSError:
            pass
        finally:
            os._exit(3)
    return code


if __name__ == "__main__":
    raise SystemExit(process_main(sys.argv[1]))
