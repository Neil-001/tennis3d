import copy
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from tennis3d.lifting.checkpoint import (
    load_lifter,
    random_state,
    read_checkpoint,
    restore_random_state,
    save_checkpoint,
)
from tennis3d.lifting.model import Lifter, ModelConfig
from tennis3d.provenance import code_identity, code_revision


def test_checkpoint_loads_ema_and_restores_rng(tmp_path):
    model = Lifter(ModelConfig(width=16, depth=2))
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    ema.update_parameters(model)
    initial = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update_parameters(model)
    torch.testing.assert_close(
        ema.module.state_dict()["ball.weight"], initial["ball.weight"] + 0.001
    )
    path = tmp_path / "last.pt"
    rng = random_state()
    state = {
        "schema_version": "1.0",
        "world_scale_m": 10.0,
        "config": {"model": model.config.model_dump()},
        "ema": ema.state_dict(),
        "rng": rng,
    }
    save_checkpoint(path, state)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    loaded, saved = load_lifter(path)
    torch.testing.assert_close(
        loaded.ball.weight, ema.module.state_dict()["ball.weight"]
    )
    assert int(saved["ema"]["n_averaged"]) == 2
    restore_random_state(read_checkpoint(path)["rng"], "cpu")
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])


def test_source_provenance_does_not_depend_on_caller_directory(tmp_path, monkeypatch):
    revision = code_revision()
    identity = code_identity()
    monkeypatch.chdir(tmp_path)
    assert code_revision() == revision
    assert code_identity() == identity
    assert "tennis3d/lifting/model.py" in identity["source_files"]


def test_content_fingerprint_tracks_dirty_candidate_bytes(tmp_path, monkeypatch):
    from tennis3d import provenance

    package = tmp_path / "installed" / "tennis3d"
    package.mkdir(parents=True)
    candidate = package / "model.py"
    candidate.write_text("width = 64\n")
    monkeypatch.setattr(provenance, "__file__", str(package / "provenance.py"))
    first = provenance.code_identity()
    assert first["code_revision"] == "unavailable"
    candidate.write_text("width = 128\n")
    assert provenance.code_identity()["source_sha256"] != first["source_sha256"]


def test_invalid_checkpoint_is_an_input_error(tmp_path):
    path = tmp_path / "broken.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        read_checkpoint(path)
    torch.save([], path)
    with pytest.raises(TypeError, match="mapping"):
        read_checkpoint(path)


@pytest.mark.parametrize("replacement_phase", ["load", "score"])
def test_evaluation_rejects_replaced_checkpoint(
    tmp_path, monkeypatch, replacement_phase
):
    from tennis3d.evaluation import synthetic

    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"original checkpoint")
    replacement = tmp_path / "new.pt"
    replacement.write_bytes(b"new checkpoint")

    def load(*args, **kwargs):
        if replacement_phase == "load":
            replacement.replace(checkpoint)
        return object(), {"step": 1, "config": {"artifact_label": "test"}}

    def score(*args, **kwargs):
        if replacement_phase == "score":
            replacement.replace(checkpoint)
        return {}

    monkeypatch.setattr(synthetic, "load_lifter", load)
    monkeypatch.setattr(synthetic, "evaluate_model", score)
    with pytest.raises(ValueError, match="checkpoint changed"):
        synthetic.evaluate_checkpoint(checkpoint, "unused.manifest.json", device="cpu")
