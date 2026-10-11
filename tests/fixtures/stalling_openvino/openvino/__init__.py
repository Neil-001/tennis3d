"""Stand-in OpenVINO runtime whose device can stop answering.

It follows the real runtime's behaviour after the i915 fence expiry recorded in
the optional Intel graphics backend. The request never completes. From then on, destroying any
handle on the device blocks without releasing the interpreter lock.
Networks run through PyTorch on the CPU.
"""

import ctypes
import os
import threading
import time

import torch


class _Device:
    stall_at = int(os.environ.get("STANDIN_STALL_AT", "-1"))
    non_finite_at = int(os.environ.get("STANDIN_NON_FINITE_AT", "-1"))
    log = os.open(os.environ["STANDIN_LOG"], os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    # PyDLL keeps the interpreter lock during the call, as a native destructor does.
    block = ctypes.PyDLL(None).sleep
    write = os.write
    requests = 0
    stalled = False


class _Handle:
    device = _Device

    def __del__(self):
        name = type(self).__name__
        if self.device.stalled:
            self.device.write(self.device.log, f"blocked {name}\n".encode())
            self.device.block(1_000_000)
        self.device.write(self.device.log, f"destroyed {name}\n".encode())


class _DeviceName(_Handle, str):
    """A handle that reaches the caller, so the backend cannot keep it alive."""


class _Results(tuple):
    def to_tuple(self):
        return tuple(self)


class InferRequest(_Handle):
    def __init__(self, module):
        self.module = module
        self.results = None
        self.finished = True

    def start_async(self, inputs):
        index = self.device.requests
        self.device.requests += 1
        if index == self.device.stall_at:
            self.device.stalled = True
            self.finished = False
            # Native worker threads are not joined at interpreter exit either.
            threading.Thread(target=threading.Event().wait, daemon=True).start()
            return
        output = self.module(torch.from_numpy(inputs[0]))
        output = output if isinstance(output, tuple) else (output,)
        arrays = [tensor.numpy().copy() for tensor in output]
        if index == self.device.non_finite_at:
            arrays[-1] = arrays[-1].astype("float32") * float("nan")
        self.results = _Results(arrays)

    def wait_for(self, milliseconds):
        if not self.finished:
            time.sleep(milliseconds / 1000)
        return self.finished


class CompiledModel(_Handle):
    def __init__(self, module):
        self.module = module

    def create_infer_request(self):
        return InferRequest(self.module)


class Core(_Handle):
    available_devices = ("CPU", "GPU")

    def compile_model(self, model, target, properties):
        if os.environ.get("STANDIN_COMPILE_ERROR"):
            raise RuntimeError("[GPU] clWaitForEvents, error code: -14")
        return CompiledModel(model)

    def get_property(self, target, name):
        if os.environ.get("STANDIN_UNOWNED_HANDLE"):
            return _DeviceName("Stand-in graphics")
        return "Stand-in graphics"


def convert_model(module, example_input, input):
    return module


def get_version():
    return "stand-in"
