"""Optional OpenVINO execution of the unchanged heatmap checkpoints.

Only the device mapped below is supported by this optional backend.
Float32 is forced; half precision moved court landmarks in those checks.
"""

import faulthandler
import os
import stat
import sys

import numpy as np
import torch
from torch import nn

from tennis3d.perception.provenance import OPENVINO_DEVICES

PRECISION = "f32"
# Inference takes a fraction of a second. A faulty compute runtime can stall
# the device forever, so a stalled request becomes an ordinary stage failure.
TIMEOUT_MS = 120_000
# Destroying a request waits for it to finish, and on a stalled device
# destroying the session's other handles blocked too. After a stall every native
# handle of the session is kept alive here, and the process must leave with
# os._exit instead of normal teardown.
STALLED = []
# The process gets this long after a stall to finish its stages, publish and
# call os._exit. If a native cleanup blocks it anyway, it ends with status 1.
EXIT_GRACE_S = 300


class _BallLabels(nn.Module):
    """Reduce on the device: copying 256 logit maps to the host costs more."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        logits = self.network(x)
        # Any infinity or NaN makes its channel sum non-finite.
        return logits.argmax(dim=1), logits.sum(dim=1)


def _runtime():
    try:
        import openvino
    except ImportError as error:
        raise RuntimeError(
            "Intel GPU inference needs OpenVINO; install the intel extra"
        ) from error
    return openvino


def _end_process_after_grace():
    # The timer is a C thread that needs no interpreter lock, which a blocked
    # native destructor may hold forever. It writes every thread's stack before
    # it exits, and that write must not block: a full pipe would keep the
    # process alive. Only a regular file receives the stacks, through a
    # descriptor of its own that later redirection of stderr cannot change.
    sink = None
    try:
        if stat.S_ISREG(os.fstat(sys.stderr.fileno()).st_mode):
            sink = os.dup(sys.stderr.fileno())
    except (AttributeError, ValueError, OSError):
        pass
    if sink is None:
        sink = os.open(os.devnull, os.O_WRONLY)
    faulthandler.dump_traceback_later(EXIT_GRACE_S, exit=True, file=sink)


def _infer(session, request, array):
    request.start_async({0: array})
    if not request.wait_for(TIMEOUT_MS):
        STALLED.append(session)
        _end_process_after_grace()
        raise RuntimeError(
            "Intel GPU inference stalled; use a compatible Intel compute driver "
            "or retry with CPU or NVIDIA CUDA"
        )
    return request.results


def _compile(openvino, session, module, channels, size, target):
    """Compile on the session's core. The session owns every native handle."""
    shape = (1, channels, *size)
    with torch.no_grad():
        model = openvino.convert_model(
            module, example_input=torch.zeros(shape), input=[shape]
        )
    compiled = session[0].compile_model(
        model, target, {"INFERENCE_PRECISION_HINT": PRECISION}
    )
    session.append(compiled)
    request = compiled.create_infer_request()
    session.append(request)
    return request


def load_inference(networks, device, size=(360, 640)):
    """Compile both CPU-loaded networks for one fixed (height, width)."""
    openvino = _runtime()
    target = OPENVINO_DEVICES[device]
    core = openvino.Core()
    if not any(name.split(".")[0] == target for name in core.available_devices):
        raise RuntimeError(
            f"{device} requested but OpenVINO reports no {target} device; "
            "install a compatible Intel compute driver or use CPU or NVIDIA CUDA"
        )
    session = [core]
    court_request = _compile(openvino, session, networks["court"], 3, size, target)
    ball_request = _compile(
        openvino, session, _BallLabels(networks["ball"]), 9, size, target
    )

    def court(frame):
        logits = _infer(session, court_request, frame)[0]
        if not np.isfinite(logits).all():
            raise RuntimeError("Non-finite court network output")
        return (torch.from_numpy(logits[0]).sigmoid().numpy() * 255).astype(np.uint8)

    def ball(frames):
        labels, total = _infer(session, ball_request, frames).to_tuple()
        if not np.isfinite(total).all():
            raise RuntimeError("Non-finite ball network output")
        return labels[0].astype(np.uint8)

    identity = {
        "backend": "openvino",
        "openvino": openvino.get_version(),
        "device": target,
        "device_name": core.get_property(target, "FULL_DEVICE_NAME"),
        "inference_precision": PRECISION,
    }
    return {"court": court, "ball": ball}, identity
