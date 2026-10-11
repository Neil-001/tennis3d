"""Main-only identities and bounded admission preserve every physical row."""

import pickle
from collections import Counter
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("pyarrow")

from tennis3d.datasets.arrays import validate_record
from tennis3d.datasets.schema import Pool
from tennis3d.synthetic import producer_records as producer
from tennis3d.synthetic.config import SimulationConfig
from tennis3d.synthetic.generate import family_seed_for
from tennis3d.synthetic.priors import RallyPriors
from tennis3d.synthetic.rally import Rally


def evidence(n=400):
    return {
        "t_s": np.arange(n, dtype=np.float64) / 50,
        "xyz_m": np.tile([0.0, 0.0, 1.0], (n, 1)),
        "label_valid": np.ones(n, dtype=bool),
        "camera_valid": np.ones(n, dtype=bool),
        "ball_in_frame": np.ones(n, dtype=bool),
        "ball_observed": np.ones(n, dtype=bool),
        "camera_R": np.tile(np.eye(3), (n, 1, 1)),
        "camera_K": np.tile(np.eye(3) * 1000, (n, 1, 1)),
        "camera_t": np.tile([0.0, 20.0, -10.0], (n, 1)),
    }


@pytest.mark.parametrize(
    "key,reason",
    [
        ("ball_in_frame", "no_clean_evidence"),
        ("ball_observed", "no_corrupted_evidence"),
    ],
)
def test_clean_and_corrupted_checks_are_independent(key, reason):
    arrays = evidence()
    arrays[key][:] = False
    result = producer.admission_failures(arrays)
    assert not result["accepted"]
    assert [w["index"] for w in result["rejected_windows"]] == [0, 1, 2]
    assert all(w["reasons"] == [reason] for w in result["rejected_windows"])


def test_terminal_short_window_and_invalid_camera_intervals():
    arrays = evidence()
    arrays["ball_observed"][:] = False
    arrays["ball_observed"][[0, 125]] = True
    result = producer.admission_failures(arrays)
    assert result["rejected_windows"] == [
        {"index": 2, "start": 250, "stop": 400, "reasons": ["no_corrupted_evidence"]}
    ]
    arrays = evidence(260)
    arrays["camera_valid"][249:255] = False
    arrays["ball_in_frame"][255:] = False
    assert producer.admission_failures(arrays)["rejected_windows"][0] == {
        "index": 1,
        "start": 255,
        "stop": 260,
        "reasons": ["no_clean_evidence"],
    }


def test_no_supported_rows_is_failure_even_without_windows():
    arrays = evidence()
    arrays["label_valid"][:] = False
    assert producer.admission_failures(arrays)["reasons"] == ["no_supported_labels"]
    arrays["camera_valid"][:] = False
    result = producer.admission_failures(arrays)
    assert not result["accepted"]
    assert result["window_count"] == 0


def test_seed_identity_is_split_owned_and_count_order_independent():
    cfg = SimulationConfig(train_rallies=13, train_families=4, views_per_rally=2)
    larger = cfg.model_copy(update={"train_rallies": 100, "train_families": 7})
    for split in producer.SPLITS:
        assert producer.plan_family(
            cfg, split, 0, "0" * 64, {"numpy": "test"}
        ) == producer.plan_family(larger, split, 0, "0" * 64, {"numpy": "test"})
    assert producer.rally_seed(cfg.seed, "train", 2) == producer.rally_seed(
        larger.seed, "train", 2
    )
    plans = list(producer.plan_rallies(cfg, cohort_size=2))
    assert [p.rally_index for p in plans if p.split == "train"] == [
        0,
        1,
        4,
        5,
        8,
        9,
        12,
        2,
        3,
        6,
        7,
        10,
        11,
    ]
    assert sorted((p.split, p.rally_index) for p in plans) == sorted(
        (s, i) for s in producer.SPLITS for i in range(getattr(cfg, f"{s}_rallies"))
    )
    audit = producer.audit_plan_seeds(cfg)
    assert audit["proposals"] == 15 * 2 * 8
    assert audit["unique_seeds"] == 6 + 15 + 240
    assert audit == producer.audit_plan_seeds(cfg)
    assert max(p.seed for p in plans) > 2**32
    assert all(0 <= p.seed < 2**63 for p in plans)
    assert len({producer.family_seed(42, s, 0) for s in producer.SPLITS}) == 3


def test_collision_audit_includes_unused_proposals(monkeypatch):
    original = producer.proposal_seed
    monkeypatch.setattr(
        producer,
        "proposal_seed",
        lambda seed, view, attempt: original(
            seed, view, 0 if attempt == 7 else attempt
        ),
    )
    with pytest.raises(ValueError, match="seed collision"):
        producer.audit_plan_seeds(SimulationConfig())


@pytest.fixture
def builder(monkeypatch, tmp_path):
    cfg = SimulationConfig(views_per_rally=2)
    source = "0" * 64
    runtimes = {"numpy": "test"}
    family = producer.plan_family(cfg, "train", 0, source, runtimes)
    pool = Pool(
        family_id=family.family_id,
        split="train",
        sha256="1" * 64,
        config=family.pool_config,
        attempts={"toss": 1},
    )
    physical = {
        k: v for k, v in evidence().items() if k in {"t_s", "xyz_m", "label_valid"}
    }
    physical.update(
        {
            "velocity_m_s": np.zeros((400, 3)),
            "spin_rad_s": np.zeros((400, 3)),
            "segment_id": np.zeros(400, dtype=np.int64),
            "prior_id": np.array([0], dtype=np.int64),
            "segment_kind": np.array([0], dtype=np.int64),
            "event_t_s": np.array([1.0, 2.0, 7.0]),
            "event_code": np.ones(3, dtype=np.int64),
            "event_xyz_m": np.zeros((3, 3)),
            "event_precision_s": np.zeros(3),
            "hit_before": np.empty((0, 9)),
            "hit_after": np.empty((0, 9)),
        }
    )
    cached = SimpleNamespace(
        sim=None, pool=SimpleNamespace(diagnostics=Counter()), descriptor=pool
    )
    monkeypatch.setattr(producer, "_get_family", lambda *args: (cached, {}))
    calls = []

    def assemble(*args):
        calls.append(args[-1])
        return Rally({k: v.copy() for k, v in physical.items()}, 1, 50)

    monkeypatch.setattr(producer, "assemble", assemble)

    def run(*, recipe=None, **kwargs):
        return producer.produce_record(
            recipe or cfg, "train", 0, pool, tmp_path, source, runtimes, **kwargs
        )

    return run, cfg, physical, calls


def test_retry_replay_preserves_physics_excerpt_and_original_times(
    builder, monkeypatch
):
    run, cfg, physical, calls = builder
    seed = producer.rally_seed(cfg.seed, "train", 0)
    rejected = {producer.proposal_seed(seed, view, 0) for view in range(2)}
    seen = []

    def observe(arrays, cfg, proposed):
        seen.append((proposed, arrays["t_s"].copy(), arrays["label_valid"].copy()))
        result = {**evidence(len(arrays["t_s"])), **arrays}
        if proposed in rejected:
            result["ball_observed"][:] = False
        return result

    monkeypatch.setattr(producer, "observe", observe)
    record, diagnostics = run()
    again, repeated = run()
    assert diagnostics == repeated
    assert calls == [seed, seed]
    assert len(seen) == 8
    for key, values in physical.items():
        np.testing.assert_array_equal(record.physical[key], values)
        np.testing.assert_array_equal(again.physical[key], values)
    np.testing.assert_array_equal(seen[0][1], seen[1][1])
    np.testing.assert_array_equal(seen[2][1], seen[3][1])
    for first, second in zip(record.views, again.views):
        assert first.metadata == second.metadata
        for key in first.arrays:
            np.testing.assert_array_equal(first.arrays[key], second.arrays[key])
    excerpt = record.views[1]
    assert excerpt.metadata.physical_time_origin_s == physical["t_s"][80]
    np.testing.assert_array_equal(excerpt.arrays["xyz_m"], physical["xyz_m"][80:320])
    np.testing.assert_array_equal(
        excerpt.arrays["t_s"], physical["t_s"][80:320] - physical["t_s"][80]
    )
    np.testing.assert_array_equal(
        excerpt.arrays["event_t_s"], [2 - physical["t_s"][80]]
    )
    for index, view in enumerate(diagnostics["views"]):
        assert view["attempt_count"] == 2
        assert view["original_seed"] == producer.proposal_seed(seed, index, 0)
        assert view["accepted_seed"] == record.views[index].metadata.view_seed
        assert view["proposals"][0]["summary"]["missing_fraction"] == 1.0
        assert view["proposals"][1]["summary"]["missing_fraction"] == 0.0


def test_exhaustion_preserves_planned_id_and_all_eight_rejections(builder, monkeypatch):
    run, _cfg, _, calls = builder

    def observe(arrays, cfg, seed):
        result = {**evidence(len(arrays["t_s"])), **arrays}
        result["ball_in_frame"][:] = False
        return result

    monkeypatch.setattr(producer, "observe", observe)
    with pytest.raises(producer.AdmissionError) as caught:
        run()
    error = caught.value
    assert error.rally_id == "train-rally-000000"
    assert error.view_id == 0
    assert len(calls) == 1
    view = error.diagnostics["views"][0]
    assert view["accepted_seed"] is None
    assert view["attempt_count"] == 8
    assert len({p["seed"] for p in view["proposals"]}) == 8
    assert all(p["rejected_windows"] for p in view["proposals"])
    restored = pickle.loads(pickle.dumps(error))
    assert restored.diagnostics == error.diagnostics
    assert str(restored) == str(error)


def test_legacy_family_seed_and_attempt_zero_are_unchanged():
    for split in range(3):
        expected = int(
            np.random.SeedSequence(42, spawn_key=(split,)).generate_state(1)[0]
        )
        assert family_seed_for(42, split, 0) == expected
    # Main deliberately uses a new namespace even for proposal zero.
    seed = producer.rally_seed(42, "train", 0)
    old = int(np.random.SeedSequence([seed, 0, 2]).generate_state(1)[0])
    assert producer.proposal_seed(seed, 0, 0) != old


def test_pool_cache_bounded_immutable_and_reused(monkeypatch, tmp_path):
    producer.close_worker_cache()
    cfg = SimulationConfig(train_rallies=4, train_families=4, pool_size=1)
    loads = []
    monkeypatch.setattr(producer, "Simulator", lambda physics: object())

    def load(path, provenance, sim, cfg, seed):
        loads.append(seed)
        if not path.exists():
            path.write_bytes(str(seed).encode())
        return SimpleNamespace(
            states={"toss": np.zeros((1, 9))},
            attempts={"toss": 1},
            diagnostics=Counter(),
        )

    monkeypatch.setattr(producer, "load_or_build", load)
    args = (tmp_path, "0" * 64, {"numpy": "test"})
    first, diag = producer.prepare_family(cfg, "train", 0, *args, cache_size=2)
    assert diag["pool_built"]
    again, diag = producer.prepare_family(cfg, "train", 0, *args, cache_size=2)
    assert first == again and diag["cache_hit"]
    assert len(loads) == 1
    for index in [1, 2]:
        producer.prepare_family(cfg, "train", index, *args, cache_size=2)
    assert len(producer._CACHE) == 2
    again, diag = producer.prepare_family(cfg, "train", 0, *args, cache_size=2)
    assert not diag["pool_built"] and not diag["cache_hit"]
    assert first == again
    plan = producer.plan_family(cfg, "train", 0, args[1], args[2])
    (tmp_path / plan.cache_name).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="immutable pool changed"):
        producer.prepare_family(cfg, "train", 0, *args, cache_size=2)
    producer.close_worker_cache()
    assert not producer._CACHE


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_worker_byte_limit_is_validated_before_physics(builder, limit):
    run, _, _, calls = builder
    with pytest.raises(ValueError, match="positive integer"):
        run(max_record_bytes=limit)
    assert not calls


def test_recipe_frame_reservation_rejects_before_physics(builder):
    run, cfg, _, calls = builder
    reservation = producer.recipe_frame_bytes(cfg)
    with pytest.raises(producer.RecordSizeError) as caught:
        run(max_record_bytes=reservation - 1)
    assert not calls
    error = caught.value
    assert error.rally_id == "train-rally-000000"
    assert error.stage == "recipe frame reservation"
    assert error.measured == reservation
    restored = pickle.loads(pickle.dumps(error))
    assert str(restored) == str(error)
    assert restored.limit == reservation - 1


def test_frame_reservation_accounts_for_cadence_returns_and_all_views():
    cfg = SimulationConfig(views_per_rally=2)
    reservation = producer.recipe_frame_bytes(cfg)
    assert reservation < producer.DEFAULT_MAX_RECORD_BYTES
    for updates in ({"views_per_rally": 3}, {"max_returns": 9}):
        assert producer.recipe_frame_bytes(cfg.model_copy(update=updates)) > reservation
    fast = cfg.model_copy(
        update={"rally_priors": RallyPriors(cadence_fps=[240], cadence_weights=[1.0])}
    )
    assert producer.recipe_frame_bytes(fast) > reservation
    diagnostic = cfg.model_copy(update={"rally_mode": "diagnostic"})
    assert producer.recipe_frame_bytes(diagnostic) == producer.recipe_frame_bytes(
        diagnostic.model_copy(update={"max_returns": 10000})
    )


@pytest.mark.parametrize("admitted", [True, False])
def test_serialized_limit_includes_metadata_and_diagnostics(
    builder, monkeypatch, admitted
):
    run, cfg, _, _ = builder
    monkeypatch.setattr(
        producer,
        "observe",
        lambda arrays, cfg, seed: {**evidence(len(arrays["t_s"])), **arrays},
    )
    result = run()
    size = len(ForkingPickler.dumps(result))
    assert producer._check_worker_result(result, size) == size
    with pytest.raises(producer.RecordSizeError, match="serialized worker result"):
        producer._check_worker_result(result, size - 1)
    # Small cadence reservation, large proposal diagnostics: array-only checks
    # pass, but the worker must reject the complete result before returning it.
    recipe = cfg.model_copy(
        update={"rally_priors": RallyPriors(cadence_fps=[1], cadence_weights=[1.0])}
    )
    monkeypatch.setattr(
        producer, "proposal_summary", lambda *args: {"details": "x" * (2 << 20)}
    )
    if not admitted:

        def missing(arrays, cfg, seed):
            result = {**evidence(len(arrays["t_s"])), **arrays}
            result["ball_observed"][:] = False
            return result

        monkeypatch.setattr(producer, "observe", missing)
    with pytest.raises(producer.RecordSizeError, match="serialized worker result"):
        run(recipe=recipe, max_record_bytes=1 << 20)


def test_reloaded_pool_descriptor_is_verified_against_recorded_pool(
    monkeypatch, tmp_path
):
    producer.close_worker_cache()
    cfg = SimulationConfig(pool_size=1)
    source, runtimes = "0" * 64, {"numpy": "test"}
    monkeypatch.setattr(producer, "Simulator", lambda physics: object())

    def load(path, *args):
        if not path.exists():
            path.write_bytes(b"original")
        return SimpleNamespace(
            states={"toss": np.zeros((1, 9))},
            attempts={"toss": 1},
            diagnostics=Counter(),
        )

    monkeypatch.setattr(producer, "load_or_build", load)
    pool, _ = producer.prepare_family(cfg, "train", 0, tmp_path, source, runtimes)
    producer.close_worker_cache()
    path = tmp_path / producer.plan_family(cfg, "train", 0, source, runtimes).cache_name
    path.unlink()
    path.write_bytes(b"rebuilt-but-different")
    with pytest.raises(ValueError, match="planned pool descriptor mismatch"):
        producer.produce_record(cfg, "train", 0, pool, tmp_path, source, runtimes)
    producer.close_worker_cache()


def test_real_records_replay_after_reordering_cache_reload_and_pool_rebuild(tmp_path):
    producer.close_worker_cache()
    cfg = SimulationConfig(
        train_rallies=2, views_per_rally=2, min_returns=1, max_returns=2
    )
    source = "0" * 64
    runtimes: dict[str, str] = {"numpy": np.__version__}
    try:
        pool, _ = producer.prepare_family(cfg, "train", 0, tmp_path, source, runtimes)

        def run(index):
            result = producer.produce_record(
                cfg, "train", index, pool, tmp_path, source, runtimes
            )
            validate_record(result[0])
            return result

        def equal(first, second):
            a, ad = first
            b, bd = second
            assert a.metadata == b.metadata
            assert ad == bd
            for key in a.physical:
                assert a.physical[key].dtype == b.physical[key].dtype
                np.testing.assert_array_equal(a.physical[key], b.physical[key])
            for av, bv in zip(a.views, b.views, strict=True):
                assert av.metadata == bv.metadata
                for key in av.arrays:
                    assert av.arrays[key].dtype == bv.arrays[key].dtype
                    np.testing.assert_array_equal(av.arrays[key], bv.arrays[key])

        first, second = run(0), run(1)
        equal(first, run(0))
        producer.close_worker_cache()
        equal(second, run(1))
        equal(first, run(0))
        producer.close_worker_cache()
        path = (
            tmp_path
            / producer.plan_family(cfg, "train", 0, source, runtimes).cache_name
        )
        path.unlink()
        equal(first, run(0))
    finally:
        producer.close_worker_cache()
