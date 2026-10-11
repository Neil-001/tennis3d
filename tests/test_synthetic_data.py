"""Small seeded pools, legal transitions, masks, projection and bounded failure."""

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from tennis3d.artifacts import ArtifactProvenance
from tennis3d.geometry.camera import Camera, ImageSize
from tennis3d.geometry.projection import project
from tennis3d.synthetic import KINDS
from tennis3d.synthetic.cameras import observe
from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.data import validate_view
from tennis3d.synthetic.export import result_payload
from tennis3d.synthetic.generate import excerpt
from tennis3d.synthetic.rally import assemble
from tennis3d.synthetic.shot_pool import GenerationError, load_or_build
from tennis3d.synthetic.simulate import Simulator


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    cfg = SimulationConfig()
    sim = Simulator(cfg.physics)
    path = tmp_path_factory.mktemp("pool") / "pool.npz"
    provenance = ArtifactProvenance(
        source_sha256="0" * 64, config_sha256="1" * 64, code_revision="test"
    )
    first = load_or_build(path, provenance, sim, cfg, 42)
    cached = load_or_build(path, provenance, sim, cfg, 42)
    for kind in KINDS:
        np.testing.assert_array_equal(first.states[kind], cached.states[kind])
    return cfg, sim, first


def test_diagnostic_tennis_rules_continuity_and_export(pool):
    cfg, sim, states = pool
    cfg = cfg.model_copy(update={"rally_mode": "diagnostic"})
    rally = assemble(sim, states, cfg, 0)
    a = rally.arrays
    np.testing.assert_array_equal(a["hit_before"][:, :3], a["hit_after"][:, :3])
    assert np.all(np.diff(a["t_s"]) > 0)
    hits = a["event_t_s"][a["event_code"] == 2]
    bounce_t = a["event_t_s"][a["event_code"] == 1]
    assert len(bounce_t) == 5
    assert not np.any(
        (bounce_t > hits[1]) & (bounce_t < hits[2])
    )  # incoming volley has no bounce
    serve_bounce = a["event_xyz_m"][a["event_code"] == 1][0]
    assert 0 < serve_bounce[0] < 4.115 and 0 < serve_bounce[1] < 6.4
    crossings = a["event_t_s"][a["event_code"] == 3]
    assert hits[0] < crossings[0] < bounce_t[0] < hits[1]
    np.testing.assert_allclose(a["event_xyz_m"][a["event_code"] == 3, 1], 0, atol=1e-12)
    assert a["event_code"][-1] == 4
    assert abs(a["xyz_m"][-1, 1]) > 11.885 or abs(a["xyz_m"][-1, 0]) > 4.115
    observed = observe(a, cfg, 100)
    validate_view(observed)
    assert (~observed["ball_observed"]).any() and observed["label_valid"].all()
    size = ImageSize(cfg.image_width, cfg.image_height)
    camera = Camera(
        observed["camera_K"][10],
        observed["camera_R"][10],
        observed["camera_t"][10],
        size,
    )
    uv, valid = project(camera, a["xyz_m"][10])
    assert valid
    np.testing.assert_allclose(uv, observed["ball_clean_uv"][10])
    payload = result_payload(
        observed,
        name="test",
        source_hash="0" * 64,
        config_hash="1" * 64,
        revision="test",
    )
    assert not any(
        e["kind"] == "net" for e in payload["synthetic_diagnostics"]["events"]
    )
    assert len(payload["frames"]) == len(payload["synthetic_diagnostics"]["frames"])
    short, offset = excerpt(a, 20, 100)
    assert offset == a["t_s"][20] and short["t_s"][0] == 0
    assert short["xyz_m"][0, 2] > cfg.physics.radius_m
    assert short["event_t_s"].min() >= 0


def test_seeded_varied_sequences_and_scale_config(pool):
    cfg, sim, states = pool
    sequences = []
    for seed in (0, 1, 2):
        rally = assemble(sim, states, cfg, seed)
        sequences.append(tuple(rally.arrays["segment_kind"]))
    assert len(set(sequences)) == 3
    assert any(
        sum(k == KINDS.index("return") for k in sequence) >= 2 for sequence in sequences
    )
    repeat = assemble(sim, states, cfg, 0)
    again = assemble(sim, states, cfg, 0)
    np.testing.assert_array_equal(repeat.arrays["xyz_m"], again.arrays["xyz_m"])
    assert (
        SimulationConfig(
            train_rallies=200000,
            val_rallies=5000,
            test_rallies=5000,
            families_per_split=100,
            pool_size=1000,
        ).train_rallies
        == 200000
    )


def test_bounded_pool_failure(tmp_path):
    cfg = SimulationConfig(pool_size=2, pool_attempts=1)
    provenance = ArtifactProvenance(
        source_sha256="0" * 64, config_sha256="1" * 64, code_revision="test"
    )
    with pytest.raises(GenerationError, match="only 1/2 valid states after 1 attempts"):
        load_or_build(
            tmp_path / "impossible.npz", provenance, Simulator(cfg.physics), cfg, 0
        )
    assert not (tmp_path / "impossible.npz").exists()
