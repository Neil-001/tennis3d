"""Host capability discovery cannot block app requests or choose CPU fallback."""

import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tennis3d.app import capabilities


def test_unsupported_platform_does_not_start_native_probe(monkeypatch):
    monkeypatch.setattr(capabilities, "supported", lambda: False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("native probe"))
    assert capabilities.bounded_probe() == capabilities.unavailable(
        "unsupported-platform"
    )


@pytest.mark.parametrize(
    "payload,code,expected",
    [
        ('{"status":"available","device":"GPU"}', 0, "available"),
        ('{"status":"unavailable","reason":"no-gpu"}', 0, "unavailable"),
        ("native error", 1, "unavailable"),
        ("not json", 0, "unavailable"),
        ("[]", 0, "unavailable"),
        ('{"status":"unknown"}', 0, "unavailable"),
    ],
)
def test_bounded_probe_handles_missing_or_broken_runtime(
    monkeypatch, payload, code, expected
):
    monkeypatch.setattr(capabilities, "supported", lambda: True)

    def run(command, **kwargs):
        assert command == [sys.executable, "-I", "-m", "tennis3d.app.capabilities"]
        assert kwargs["timeout"] == capabilities.PROBE_TIMEOUT_S
        return SimpleNamespace(returncode=code, stdout=payload)

    monkeypatch.setattr(subprocess, "run", run)
    assert capabilities.bounded_probe()["status"] == expected


def test_native_stall_is_killed_by_outer_timeout(monkeypatch):
    monkeypatch.setattr(capabilities, "supported", lambda: True)
    native_run = subprocess.run

    def stalled_probe(command, **kwargs):
        return native_run(
            [sys.executable, "-c", "import time; time.sleep(60)"], **kwargs
        )

    monkeypatch.setattr(subprocess, "run", stalled_probe)
    started = time.monotonic()
    assert capabilities.bounded_probe(timeout=0.1) == capabilities.unavailable(
        "timeout"
    )
    assert time.monotonic() - started < 3


def test_pending_probe_snapshots_are_nonblocking(monkeypatch):
    entered, finish = threading.Event(), threading.Event()

    def delayed_probe():
        entered.set()
        assert finish.wait(timeout=5)
        return {"status": "available", "device": "GPU"}

    monkeypatch.setattr(capabilities, "bounded_probe", delayed_probe)
    probe = capabilities.IntelCapability()
    try:
        assert entered.wait(timeout=5)
        assert probe.snapshot() == {"status": "checking"}
    finally:
        finish.set()
        probe.thread.join(timeout=5)
    assert probe.snapshot()["status"] == "available"
    snapshot = probe.snapshot()
    snapshot["status"] = "unavailable"
    assert probe.snapshot()["status"] == "available"


def test_gpu_discovery_without_gpu_does_not_compile_or_fallback(monkeypatch):
    monkeypatch.setattr(capabilities, "supported", lambda: True)
    core = SimpleNamespace(available_devices=["CPU", "NPU"])
    runtime = SimpleNamespace(Core=lambda: core, opset13=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "openvino", runtime)
    assert capabilities.probe() == capabilities.unavailable("no-gpu")


def test_child_reports_missing_runtime_as_normal_unavailability(monkeypatch):
    monkeypatch.setattr(capabilities, "supported", lambda: True)
    monkeypatch.setitem(sys.modules, "openvino", None)
    assert json.dumps(capabilities.probe()) == json.dumps(
        capabilities.unavailable("runtime-not-installed")
    )
