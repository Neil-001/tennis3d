"""One bounded subprocess group, shared by local and SLURM execution."""

import os
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path


class TrialFailure(Exception):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status


def run_process(
    command: list[str],
    *,
    cwd: Path,
    output: Path,
    deadline: float,
    lock_fd: int,
    stopped: Callable[[], bool],
    update: Callable[[], None],
) -> None:
    """Count setup, training, evaluation and checks under one absolute deadline.

    Inherit the trial lock so an orphan cannot overlap a new attempt. Kill the
    whole process group on timeout, user stop, supervisor signal or exception.
    """
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TrialFailure("timed-out", "trial exceeded the total wall-clock deadline")
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        # Avoid stale bytecode after candidate edits with unchanged file sizes/times.
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(output.parent / ".bytecode"),
    }
    with output.open("ab") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
            pass_fds=(lock_fd,),
        )
        try:
            while process.poll() is None:
                if stopped():
                    raise TrialFailure("interrupted", "user stopped the search")
                if time.perf_counter() >= deadline:
                    raise TrialFailure(
                        "timed-out", "trial exceeded the total wall-clock deadline"
                    )
                update()
                time.sleep(min(0.2, max(0.001, deadline - time.perf_counter())))
            if time.perf_counter() >= deadline:
                raise TrialFailure(
                    "timed-out", "trial exceeded the total wall-clock deadline"
                )
            if process.returncode:
                raise TrialFailure(
                    "crashed",
                    f"trial process exited {process.returncode}; see process.log",
                )
            update()
        finally:
            # Descendants can survive even after their immediate parent exits.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
