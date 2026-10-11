import json
import subprocess
import sys

import pytest

from tennis3d.cli import main


def test_doctor_and_version(capsys):
    assert main(["doctor", "--device", "cpu"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["device"] == "cpu"
    assert main(["version"]) == 0
    assert "tennis3d 0.1.0" in capsys.readouterr().out
    assert main(["doctor", "--device", "cuda"]) == 3


@pytest.mark.parametrize("probe", ["perception", "lifting"])
def test_doctor_failed_cuda_probe(probe, monkeypatch, capsys):
    def fail(*args):
        raise RuntimeError("probe failed")

    target = (
        "tennis3d.perception.doctor.check"
        if probe == "perception"
        else "tennis3d.inference.doctor.check_lifter"
    )
    monkeypatch.setattr(target, fail)
    flags = ["--perception"] if probe == "perception" else ["--checkpoint", "x.pt"]
    assert main(["doctor", "--device", "cuda", *flags]) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["cuda"] == "CUDA verification failed"
    assert report[f"{probe}_error"] == "probe failed"


def test_clean_import_has_no_optional_stack():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import tennis3d
import tennis3d.cli
import tennis3d.schemas
import tennis3d.geometry.projection
assert not {'torch', 'cv2', 'av', 'mujoco'} & sys.modules.keys()
""",
        ],
        check=True,
    )


def test_simulate_malformed_yaml(tmp_path, capsys):
    config = tmp_path / "invalid.yaml"
    config.write_text("train_rallies: [")
    output = tmp_path / "output"
    assert main(["simulate", "--config", str(config), "--output", str(output)]) == 2
    assert "Simulation failed:" in capsys.readouterr().out
    assert not output.exists()
