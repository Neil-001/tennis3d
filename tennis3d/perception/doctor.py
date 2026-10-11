"""Explicit runtime/asset probe without downloads."""

import importlib.metadata
from pathlib import Path


def check(device, manifest_path: Path):
    import numpy as np
    import torch
    import torch.version

    from tennis3d.perception.accelerate import load_inference
    from tennis3d.perception.assets import manifest, verify
    from tennis3d.perception.detect import device_name
    from tennis3d.perception.network import load_network
    from tennis3d.perception.provenance import OPENVINO_DEVICES

    selected = device_name(device)
    assets = manifest(manifest_path)
    torch.set_num_threads(2)
    accelerated = selected in OPENVINO_DEVICES
    results, models = {}, {}
    for name in ("ball", "court"):
        asset = assets.assets[name]
        verify(asset)
        model = load_network(asset.path, name, "cpu" if accelerated else selected)
        models[name] = model
        with torch.inference_mode():
            # Same operations and channel layout with small spatial dimensions.
            output = model(
                torch.zeros(
                    1,
                    9 if name == "ball" else 3,
                    8,
                    8,
                    device="cpu" if accelerated else selected,
                )
            )
        if not torch.isfinite(output).all():
            raise RuntimeError(f"Non-finite {name} checkpoint output")
        results[name] = {
            "sha256": asset.sha256,
            "keys": len(model.state_dict()),
            "parameters": sum(p.numel() for p in model.parameters()),
            "compatible": True,
        }
        del model, output
    accelerator = None
    if accelerated:
        # Compiles and executes on the device; the functions reject non-finite output.
        infer, accelerator = load_inference(models, selected, size=(8, 8))
        infer["court"](np.zeros((1, 3, 8, 8), np.float32))
        infer["ball"](np.zeros((1, 9, 8, 8), np.float32))
    return {
        "device": selected,
        "accelerator": accelerator,
        "assets": results,
        "cuda": torch.version.cuda if selected == "cuda" else "not used",
        "perception_dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "opencv-python-headless",
                "scipy",
                "av",
                "pykalman",
                "scikit-image",
            )
        },
    }
