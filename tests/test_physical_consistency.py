"""Independent analytic/simulator controls, never final model or held-out data reads."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("scipy")

from tennis3d.evaluation.annotations import Annotation, AnnotationManifest
from tennis3d.evaluation.physical import FitConfig, FlightParameters, evaluate_physical
from tennis3d.evaluation.physical_controls import (
    apply_frozen_policy,
    freeze_validation_policy,
    summarize_physical,
)
from tennis3d.evaluation.physical_ode import fit_flight, flight_rhs, integrate_flight
from tennis3d.geometry.camera import Camera, ImageSize
from tennis3d.geometry.projection import project

VACUUM = FlightParameters(density_kg_m3=0.0, viscosity_pa_s=0.0)


def analytic(times, p=(0.0, 0.0, 3.0), v=(2.0, 15.0, 1.0)):
    t = np.asarray(times)[:, None]
    return np.array(p) + t * np.array(v) + 0.5 * t**2 * np.array([0, 0, -9.81])


def metadata(**changes):
    return {
        "artifact_id": "controlled",
        "domain": "controlled_fixture",
        "split": "development",
        "time_basis": "simulation",
        "timing_note": "locally constructed numerical fixture",
        "source_identity": {"fixture": "analytic"},
        "prediction_identity": {"fixture": "controlled"},
        "segmentation_source": "simulator",
        "segmentation_method": "known_fixture_contacts",
        **changes,
    }


def arrays(times, xyz):
    n = len(times)
    return {
        "t_s": np.asarray(times),
        "xyz_m": np.asarray(xyz),
        "available": np.ones(n, dtype=bool),
        "input_observed": np.ones(n, dtype=bool),
        "event_t_s": np.empty(0),
        "event_code": np.empty(0, dtype=int),
    }


def test_gravity_irregular_seconds_and_solver_convergence():
    times = np.array([0, 0.031, 0.056, 0.094, 0.135, 0.207, 0.309, 0.417])
    state = np.r_[[1, 2, 4], [3, 22, 5], [-150, 20, 30]]
    vacuum = integrate_flight(times, state, VACUUM)
    np.testing.assert_allclose(
        vacuum[:, :3], analytic(times, state[:3], state[3:6]), atol=2e-12
    )
    default = integrate_flight(times, state)
    tighter = integrate_flight(
        times,
        state,
        config=replace(FitConfig(), rtol=1e-11, atol=1e-13, max_step_s=0.025),
    )
    assert np.max(np.linalg.norm(default[:, :3] - tighter[:, :3], axis=1)) < 1e-7
    for bad in ([0, 0.1, 0.1], [0, -0.1], [0, np.nan]):
        with pytest.raises(ValueError, match="times"):
            integrate_flight(bad, state)


@pytest.mark.parametrize("speed,spin", [(5, 0), (30, -200), (60, 250), (75, -500)])
def test_force_and_flight_agreement_with_actual_mujoco(speed, spin):
    pytest.importorskip("mujoco")
    from tennis3d.synthetic.config import PhysicsConfig
    from tennis3d.synthetic.simulate import Simulator, State

    cfg = PhysicsConfig()
    initial = np.r_[[1, 2, 10], [3, speed, 2], [spin, 30, -20]]
    sim = Simulator(cfg)
    sim.reset(State(initial[:3], initial[3:6], initial[6:9]))
    np.testing.assert_allclose(
        sim.data.qacc, flight_rhs(initial, FlightParameters())[3:], atol=3e-5, rtol=1e-6
    )
    t, xyz = [0.0], [initial[:3]]
    for i in range(350):
        sim.step()
        if i % 10 == 9:
            assert not sim.in_contact
            t.append(float(sim.data.time))
            xyz.append(sim.state().xyz)
    curve = integrate_flight(t, initial)
    # Generator's 1 ms implicit stepping has position truncation error.
    assert np.max(np.linalg.norm(curve[:, :3] - xyz, axis=1)) < 0.035
    if speed == 30:
        fitted = fit_flight(t, xyz)
        assert fitted["status"] == "fitted"
        assert fitted["rmse_m"] < 0.0002
        assert fitted["max_discrepancy_m"] < 0.0005


def test_varied_tennis_parameters_and_magnus_added_mass_sign():
    pytest.importorskip("mujoco")
    from tennis3d.synthetic.config import PhysicsConfig
    from tennis3d.synthetic.simulate import Simulator, State

    for magnus in (0.0, 0.5, 0.8):
        cfg = PhysicsConfig(
            mass_kg=0.059,
            radius_m=0.034,
            density=1.1,
            viscosity=0.00002,
            magnus=magnus,
            blunt_drag=0.28,
            inertia_factor=0.54,
        )
        p = FlightParameters(
            mass_kg=cfg.mass_kg,
            radius_m=cfg.radius_m,
            density_kg_m3=cfg.density,
            viscosity_pa_s=cfg.viscosity,
            magnus=cfg.magnus,
            blunt_drag=cfg.blunt_drag,
            inertia_factor=cfg.inertia_factor,
        )
        initial = np.r_[[0, 0, 10], [4, 35, 2], [-200, 35, 10]]
        sim = Simulator(cfg)
        sim.reset(State(initial[:3], initial[3:6], initial[6:9]))
        np.testing.assert_allclose(
            sim.data.qacc, flight_rhs(initial, p)[3:], atol=1e-5, rtol=1e-6
        )
        assert np.sign(p.coefficients["magnus_kg"]) == np.sign(magnus - 0.5)


def test_robust_fit_keeps_raw_outliers_and_discriminates_nonphysical_motion():
    times = np.linspace(0, 0.5, 26)
    truth = analytic(times)
    clean = fit_flight(times, truth, VACUUM)
    assert clean["rmse_m"] < 1e-7
    assert clean["limits"][
        "state_weakly_identified"
    ]  # Spin cannot be identified in vacuum.
    outlier = truth.copy()
    outlier[12, 0] += 3.0
    bad = fit_flight(times, outlier, VACUUM)
    assert bad["max_discrepancy_m"] > 2.9
    assert bad["rmse_m"] > 0.55
    zigzag = truth.copy()
    zigzag[:, 0] += 0.7 * (-1.0) ** np.arange(len(times))
    nonphysical = fit_flight(times, zigzag, VACUUM)
    assert nonphysical["rmse_m"] > 0.65
    assert np.max(np.linalg.norm(np.array(bad["fitted_xyz_m"]) - truth, axis=1)) < 0.02
    exhausted = fit_flight(times, outlier, VACUUM, replace(FitConfig(), max_nfev=1))
    assert exhausted["status"] == "failed"
    assert "rmse_m" not in exhausted


def test_contacts_between_samples_gaps_and_exclusion_counts():
    times = np.r_[np.linspace(10.0, 10.4, 11), np.linspace(10.44, 10.84, 11)]
    xyz = np.r_[
        analytic(times[:11] - 10),
        analytic(times[11:] - 10.44, p=(1, 6, 0.0335), v=(2, 15, 6)),
    ]
    data = arrays(times, xyz)
    data.update(
        event_t_s=np.array([10.42]),
        event_end_t_s=np.array([10.425]),
        event_code=np.array([1]),
        event_precision_s=np.array([0.001]),
    )
    data["available"][3] = False
    data["xyz_m"][3] = np.nan
    data["input_observed"][11:] = False
    report = evaluate_physical(data, metadata=metadata(), physics=VACUUM)
    assert report["counts"]["candidate_segments"] == 3
    assert report["counts"]["unavailable_frames"] == 1
    assert report["counts"]["fitted_segments"] == 0
    assert report["reason_counts"]["too_few_observed_frames"] == 2
    assert report["reason_counts"]["too_short"] == 1
    assert all(
        not (min(s["frame_indices"]) < 11 <= max(s["frame_indices"]))
        for s in report["segments"]
    )
    data["available"][3] = True
    data["xyz_m"][3] = xyz[3]
    data["input_observed"][:] = True
    # xyz aliases the copied array; regenerate the repaired row explicitly.
    data["xyz_m"][3] = analytic([times[3] - 10])[0]
    report = evaluate_physical(data, metadata=metadata(), physics=VACUUM)
    assert report["counts"]["fitted_segments"] == 2
    assert max(s["rmse_m"] for s in report["segments"]) < 1e-7
    json.dumps(report, allow_nan=False)
    bad_times = data.copy()
    bad_times["t_s"] = times.copy()
    bad_times["t_s"][1] = times[0]
    with pytest.raises(ValueError, match="increasing"):
        evaluate_physical(bad_times, metadata=metadata())
    wrong_mask = data.copy()
    wrong_mask["available"] = data["available"].astype(int)
    with pytest.raises(ValueError, match="boolean"):
        evaluate_physical(wrong_mask, metadata=metadata())


def test_automatic_segmentation_and_large_timestamp_gap():
    t = np.linspace(0, 0.8, 41)
    path = analytic(t)
    path[21:] = analytic(t[21:] - 0.4, p=path[20], v=(-12, 8, 3))
    data = arrays(t, path)
    del data["event_t_s"], data["event_code"]
    report = evaluate_physical(
        data,
        metadata=metadata(
            segmentation_source="automatic", segmentation_method="velocity_jump_v1"
        ),
        physics=VACUUM,
    )
    assert report["segmentation"]["contacts"]
    assert report["counts"]["candidate_segments"] == 2
    assert report["counts"]["contact_guard_frames"] > 0
    data = arrays(np.r_[t[:10], t[10:] + 0.4], analytic(np.r_[t[:10], t[10:] + 0.4]))
    report = evaluate_physical(data, metadata=metadata(), physics=VACUUM)
    assert report["counts"]["fitted_segments"] == 2


def test_true_error_controls_translation_blindness_bins_and_frozen_policy():
    t = np.linspace(0, 0.5, 26)
    truth = analytic(t)
    reports = []
    for j, noise in enumerate((0.01, 0.1, 0.6, 1.4)):
        prediction = truth.copy()
        prediction[:, 0] += noise * (-1.0) ** np.arange(len(t))
        data = arrays(t, prediction)
        data.update(truth_xyz_m=truth, label_valid=np.ones(len(t), dtype=bool))
        reports.append(
            evaluate_physical(
                data,
                metadata=metadata(
                    artifact_id=f"fixture-{j}",
                    domain="synthetic",
                    split="reserved_validation",
                    mode="corrupted",
                ),
                physics=VACUUM,
            )
        )
    control = summarize_physical(reports)
    discrimination = control["cohorts"][0]["discrimination"]["rmse_m"]
    assert discrimination["spearman_rho"] > 0.99
    assert (
        sum(b["residual_m"]["count"] for b in discrimination["by_true_xyz_rmse_bin"])
        == 4
    )
    shifted = arrays(t, truth + [5, 0, 0])
    shifted.update(truth_xyz_m=truth, label_valid=np.ones(len(t), dtype=bool))
    shifted_report = evaluate_physical(shifted, metadata=metadata(), physics=VACUUM)
    segment = shifted_report["segments"][0]
    assert segment["rmse_m"] < 1e-7 and segment["true_xyz_error_m"][
        "rmse_m"
    ] == pytest.approx(5)
    policy = freeze_validation_policy(
        reports,
        metric="rmse_m",
        quantile=0.9,
        validation_identity={"fixture_only": True},
    )
    scored = apply_frozen_policy(reports[0], policy)
    assert scored["passed_segments"] == 1
    with pytest.raises(ValueError, match="reserved"):
        freeze_validation_policy(
            [shifted_report],
            metric="rmse_m",
            quantile=0.9,
            validation_identity={"fixture": True},
        )
    changed = copy.deepcopy(policy)
    changed["threshold_m"] += 1
    with pytest.raises(ValueError, match="hash"):
        apply_frozen_policy(reports[0], changed)
    with pytest.raises(ValueError, match="identical"):
        summarize_physical([reports[0], {**reports[0], "config": {}}])


def reference_camera():
    center = np.array([2.0, -24.0, 12.0])
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    rotation = np.array([right, np.cross(forward, right), forward])
    return Camera(
        np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]]),
        rotation,
        -rotation @ center,
        ImageSize(1280, 720),
    )


def test_independent_reprojection_and_landing_at_exact_contact_source_time():
    t = np.linspace(0, 0.5, 26)
    # Independent analytic path lands at ball-centre radius at the last timestamp.
    truth = analytic(t, p=(0, 0, 0.0335 + 9.81 * 0.5**2 / 2), v=(2, 15, 0))
    camera = reference_camera()
    uv, _ = project(camera, truth)
    labels = AnnotationManifest(
        source_sha256="a" * 64,
        labels_sha256="b" * 64,
        recording_group="fixture",
        clip_id="fixture",
        split="development",
        width=1280,
        height=720,
        source_frame_count=len(t),
        time_basis="source_pts",
        timing_note="controlled verified PTS",
        pixel_convention="ball centre",
        rights_and_split_note="generated local fixture",
        annotations=[
            Annotation(
                frame_index=i,
                image_name=f"{i}.jpg",
                t_s=float(t[i]),
                visibility=1,
                uv=uv[i].tolist(),
                event="bounce" if i == len(t) - 1 else "flight",
            )
            for i in range(len(t))
        ],
    )
    meta = metadata(
        domain="real",
        time_basis="source_pts",
        source_identity={"source_sha256": "a" * 64},
        segmentation_source="annotated",
        segmentation_method="consecutive_annotated_events",
        camera_source="controlled calibrated pinhole",
        bounce_convention="ball_center",
    )
    data = arrays(t, truth + [2, 0, 0])
    del data["event_t_s"], data["event_code"]
    report = evaluate_physical(
        data, metadata=meta, physics=VACUUM, labels=labels, cameras=[camera] * len(t)
    )
    fit = report["segments"][0]
    assert fit["rmse_m"] < 1e-7
    assert fit["agreement"]["reprojection"]["error_px"]["mean"] > 30
    landing = fit["agreement"]["landing"]["records"][0]
    assert landing["error_xy_m"] == pytest.approx(2, abs=1e-6)
    assert landing["frame_index"] == 25 and landing["extrapolation_s"] == pytest.approx(
        0.02
    )
    with pytest.raises(ValueError, match="timestamps"):
        evaluate_physical(
            {**data, "t_s": t + 0.001},
            metadata=meta,
            labels=labels,
            cameras=[camera] * len(t),
        )
    with pytest.raises(ValueError, match="source"):
        evaluate_physical(
            data,
            metadata={**meta, "source_identity": {"source_sha256": "c" * 64}},
            labels=labels,
        )
    missing_camera = evaluate_physical(
        data, metadata=meta, physics=VACUUM, labels=labels
    )
    assert (
        missing_camera["segments"][0]["agreement"]["landing"]["records"][0]["reason"]
        == "invalid_camera"
    )
    with pytest.raises(ValueError, match="source PTS"):
        evaluate_physical(
            data, metadata={**meta, "time_basis": "declared_nominal"}, labels=labels
        )


def test_native_contact_episode_is_excluded_before_fitting_both_flights():
    pytest.importorskip("mujoco")
    from tennis3d.synthetic.config import PhysicsConfig
    from tennis3d.synthetic.simulate import Simulator, State

    sim = Simulator(PhysicsConfig())
    sim.reset(
        State(
            np.array([0.0, 0.0, 2.0]),
            np.array([1.0, 18.0, -2.0]),
            np.array([-100.0, 0.0, 0.0]),
        )
    )
    times, positions = [0.0], [sim.state().xyz]
    onset, separation, next_sample = None, None, 0.01
    while sim.data.time < 0.8:
        _, started, ended = sim.step()
        if started:
            onset = float(sim.data.time)
        if ended:
            separation = float(sim.data.time)
        if sim.data.time >= next_sample or started or ended:
            times.append(float(sim.data.time))
            positions.append(sim.state().xyz)
            next_sample = max(next_sample + 0.01, float(sim.data.time) + 0.01)
    assert onset is not None and separation is not None and separation > onset
    data = arrays(times, positions)
    data.update(
        event_t_s=np.array([onset]),
        event_end_t_s=np.array([separation]),
        event_code=np.array([1]),
        event_precision_s=np.array([sim.config.contact_step_s]),
    )
    report = evaluate_physical(
        data, metadata=metadata(segmentation_method="native_onset_and_separation")
    )
    assert report["counts"]["fitted_segments"] == 2
    assert report["counts"]["contact_guard_frames"] >= 2
    assert max(s["rmse_m"] for s in report["segments"]) < 0.001
    for segment in report["segments"]:
        source_times = np.asarray(times)[segment["frame_indices"]]
        assert not np.any((source_times >= onset) & (source_times <= separation))


def test_summary_consumes_generator_once_without_retaining_curves():
    import weakref

    class Curve:
        pass

    curves = []
    t = np.linspace(0, 0.3, 16)
    base = evaluate_physical(
        arrays(t, analytic(t)), metadata=metadata(), physics=VACUUM
    )

    def reports():
        for i in range(20):
            assert sum(ref() is not None for ref in curves) <= 2
            curve = Curve()
            curves.append(weakref.ref(curve))
            report = copy.deepcopy(base)
            report["metadata"]["artifact_id"] = f"fixture-{i}"
            report["segments"][0]["fitted_xyz_m"] = curve
            yield report

    result = summarize_physical(reports())
    cohort = result["cohorts"][0]
    assert cohort["counts"]["fitted_segments"] == 20
    assert len(cohort["artifacts"]) == 20
    assert cohort["artifacts"][0]["metadata"]["source_identity"] == {
        "fixture": "analytic"
    }
    json.dumps(result, allow_nan=False)
