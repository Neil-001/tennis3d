"""Weight-free Intel runtime check, isolated from the server's request threads.

The explicit GPU target and float32 compile check establish host availability,
not detector parity or model qualification. Native discovery, compilation,
inference and teardown all run in one subprocess with an outer timeout.
"""

import json
import platform
import subprocess
import sys
import threading

PROBE_TIMEOUT_S = 20
DEVICES = ("auto", "cpu", "cuda", "intel-gpu")


def supported() -> bool:
    return sys.platform == "linux" and platform.machine().lower() == "x86_64"


def unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


def probe() -> dict:
    """Check the installed host runtime. Called only inside the bounded child."""
    if not supported():
        return unavailable("unsupported-platform")
    try:
        import numpy as np
        import openvino as ov
        from openvino import opset13 as ops
    except ImportError:
        return unavailable("runtime-not-installed")
    try:
        core = ov.Core()
        if not any(device.split(".")[0] == "GPU" for device in core.available_devices):
            return unavailable("no-gpu")
        parameter = ov.op.Parameter(ov.Type.f32, ov.Shape([1]))
        output = ops.add(parameter, ops.constant(np.array([1], dtype=np.float32)))
        # OpenVINO's Model overloads annotate their bound self as the native
        # Model instead of the exported wrapper. This constructor works at runtime.
        model = ov.Model([output], [parameter])  # ty: ignore[no-matching-overload]
        compiled = core.compile_model(model, "GPU", {"INFERENCE_PRECISION_HINT": "f32"})
        result = compiled([np.array([2], dtype=np.float32)])[0]
        if not np.array_equal(result, np.array([3], dtype=np.float32)):
            return unavailable("runtime-check-failed")
        return {
            "status": "available",
            "openvino": ov.get_version(),
            "device": "GPU",
            "device_name": core.get_property("GPU", "FULL_DEVICE_NAME"),
            "inference_precision": "f32",
            "lifting_device": "cpu",
        }
    except Exception:  # noqa: BLE001 - native runtime failures mean unavailable
        return unavailable("runtime-check-failed")


def bounded_probe(timeout=PROBE_TIMEOUT_S) -> dict:
    if not supported():
        return unavailable("unsupported-platform")
    try:
        # Isolated mode ignores PYTHONPYCACHEPREFIX, so pass the launcher's on.
        cache = sys.pycache_prefix
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                *(["-X", f"pycache_prefix={cache}"] if cache else []),
                "-m",
                "tennis3d.app.capabilities",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            return unavailable("runtime-check-failed")
        report = json.loads(result.stdout)
        if not isinstance(report, dict) or report.get("status") not in (
            "available",
            "unavailable",
        ):
            return unavailable("runtime-check-failed")
        return report
    except subprocess.TimeoutExpired:
        return unavailable("timeout")
    except (OSError, ValueError):
        return unavailable("runtime-check-failed")


class IntelCapability:
    """One bounded startup check. Snapshot reads never discover or compile."""

    def __init__(self):
        self.lock = threading.Lock()
        self.report = {"status": "checking"}
        self.thread = threading.Thread(target=self._check, daemon=True)
        self.thread.start()

    def _check(self):
        report = bounded_probe()
        with self.lock:
            self.report = report

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.report)


if __name__ == "__main__":
    print(json.dumps(probe()))
