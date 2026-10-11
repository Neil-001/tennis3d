"""CPU-only app subprocess: real detector timeout with blocking mock destruction."""

import ctypes
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tennis3d.app.worker import process_main
from tennis3d.artifacts import write_json
from tennis3d.inference import inputs
from tennis3d.perception import accelerate
from tennis3d.perception import calibrate as calibration
from tennis3d.perception import detect as stage

directory = Path(sys.argv[1])
force_deadline = len(sys.argv) > 2 and sys.argv[2].startswith("force-deadline")
if force_deadline:
    patch.object(accelerate, "EXIT_GRACE_S", 1).start()


class NativeObject:
    block = ctypes.PyDLL(None).sleep

    def __del__(self):
        if accelerate.STALLED:
            (directory / "native-destruction-entered").touch()
            self.block(60)  # Match a native destructor that holds the GIL.


class Request(NativeObject):
    def __init__(self, compiled):
        self.compiled = compiled
        self.calls = 0

    def start_async(self, inputs):
        self.calls += 1

    def wait_for(self, timeout):
        # Fail on the fourth source row, after successful court and ball calls.
        return self.compiled.channels != 3 or self.calls < 4

    @property
    def results(self):
        if self.compiled.channels == 3:
            return {0: np.zeros((1, 15, 360, 640), np.float32)}
        return SimpleNamespace(
            to_tuple=lambda: (
                np.zeros((1, 360, 640), np.int64),
                np.zeros((1, 360, 640), np.float32),
            )
        )


class Compiled(NativeObject):
    def __init__(self, channels):
        self.channels = channels

    def create_infer_request(self):
        return Request(self)


class Core(NativeObject):
    available_devices = ("GPU",)

    def compile_model(self, model, target, properties):
        assert target == "GPU" and properties["INFERENCE_PRECISION_HINT"] == "f32"
        return Compiled(model.channels)

    def get_property(self, target, name):
        return "CPU mock of stalled Intel GPU"


patch.object(
    accelerate,
    "_runtime",
    lambda: SimpleNamespace(
        Core=Core,
        convert_model=lambda module, example_input, input: SimpleNamespace(
            channels=example_input.shape[1]
        ),
        get_version=lambda: "mock",
    ),
).start()
patch.object(
    stage, "load_network", lambda path, name, device: torch.nn.Identity()
).start()
# The mock runtime has no installed OpenVINO distribution or native driver.
patch.object(
    stage,
    "runtime_versions",
    lambda dependencies: {name: "mock" for name in dependencies},
).start()
# Distribution identity is deliberately mocked along with OpenVINO. After the
# real detector returns, stop before numerical stages using fixture weights.
patch.object(inputs, "check_runtime", lambda *args: None).start()


def stop_after_detection(*args, **kwargs):
    assert accelerate.STALLED
    assert len(accelerate.STALLED) == 1, "Stalled detection must never retry"
    if force_deadline:
        if sys.argv[2] == "force-deadline-after-prefix":
            shutil.copy(sys.argv[3], directory / "inference/result.json")
            write_json(
                directory / "outcome.json",
                {
                    "summary": {"status": "partial", "exit_code": 3},
                    "produced_result": True,
                },
            )
        time.sleep(60)
    if len(sys.argv) > 2 and sys.argv[2] == "invalid-after-detection":
        raise ValueError("CPU fixture stops after published detector failure")
    raise RuntimeError("CPU fixture stops after published detector failure")


patch.object(calibration, "calibrate", stop_after_detection).start()
raise SystemExit(process_main(directory))
