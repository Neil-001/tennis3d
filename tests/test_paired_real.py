"""Paired resolution summary on two controlled evaluation roots of one model."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")

from test_cached_real import cached_gallery, checkpoint, release_manifest

from tennis3d.artifacts import file_sha256, read_json, read_npz, write_json, write_npz
from tennis3d.lifting.model import ModelConfig

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter, pairs, product = (
    load("cached_real"),
    load("paired_real"),
    load("product_manifest"),
)


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    """One model on a full cohort and on a low cohort whose detector saw a cut."""
    root = tmp_path_factory.mktemp("paired")
    release = checkpoint(root / "model.pt", ModelConfig(width=16, depth=2), 1)
    provenance = {"release": {"checkpoint": {"sha256": release.checkpoint.sha256}}}
    patch = pytest.MonkeyPatch()
    patch.setattr(adapter, "checked_release", lambda _: (release, None, provenance))
    (root / "unused-release.json").write_text("{}")
    cohorts = {}
    for cohort, cut in (("full", False), ("low", True)):
        (root / cohort).mkdir()
        fixture = cached_gallery(root / cohort, cut=cut)
        adapter.cached_real(
            root / "unused-release.json",
            fixture.inventory,
            fixture.gallery,
            fixture.manifest,
            release_manifest(fixture),
            root / cohort / "evaluated",
        )
        # The release binding whose per-file hashes the pairing checks.
        release.run_id, release.step = "fixture-run", 1
        patch.setattr(product, "checked_release", lambda _: (release, None, provenance))
        write_json(
            root / cohort / "clips.json",
            {
                "schema_version": 1,
                "clips": [
                    {
                        "id": name,
                        "title": name,
                        "source_sha256": fixture.clip.source_sha256,
                        "timeline_sha256": fixture.clip.timeline_sha256,
                    }
                    for name in ("controlled", "barred")
                ],
            },
        )
        product.product(
            root / cohort / "evaluated",
            fixture.inventory,
            file_sha256(fixture.inventory),
            root / cohort / "clips.json",
            root / "unused-release.json",
            root / cohort / "product",
        )
        cohorts[cohort] = fixture
    patch.undo()
    clip = cohorts["full"].clip
    pairing = root / "pairing.json"
    write_json(
        pairing,
        {
            "parent_inventory_sha256": file_sha256(cohorts["full"].inventory),
            "clips": [
                {
                    "clip_id": name,
                    "parent_clip_id": name,
                    "frame_count": 60,
                    "parent_source_sha256": clip.source_sha256,
                    "parent_timeline_sha256": clip.timeline_sha256,
                    "source_sha256": cohorts["low"].clip.source_sha256,
                    "timeline_sha256": cohorts["low"].clip.timeline_sha256,
                }
                for name in ("controlled", "barred")
            ],
        },
    )
    return root, cohorts, pairing


def arguments(evaluated):
    root, cohorts, pairing = evaluated
    return (
        (pairing, file_sha256(pairing)),
        {
            name: (cohorts[name].inventory, file_sha256(cohorts[name].inventory))
            for name in ("full", "low")
        },
        {
            "model": {name: root / name / "evaluated" for name in ("full", "low")}
            | {
                "bindings": {
                    name: root / name / "product/release-binding.json"
                    for name in ("full", "low")
                }
            }
        },
    )


def test_pairs_count_the_same_source_frames_in_both_results(evaluated, tmp_path):
    root = evaluated[0]
    result = pairs.paired(*arguments(evaluated), tmp_path / "out", ["Mixed devices"])
    assert result == read_json(tmp_path / "out/paired.json")
    assert result["pairs"] == 2 and result["source_frames"] == 120
    model = result["models"][0]
    controlled, barred = model["pairs"]
    # The ineligible pair stays listed without frame counts.
    assert barred["frames"] is None
    assert barred["status_change"] == ["ineligible", "ineligible"]
    assert model["pairs_with_both_results"] == 1
    assert controlled["status_change"] == ["partial", "failed"]
    assert model["status_changes"] == {
        "ineligible -> ineligible": 1,
        "partial -> failed": 1,
    }
    frames = {
        cohort: read_json(root / cohort / "evaluated/clips/controlled/result.json")[
            "frames"
        ]
        for cohort in ("full", "low")
    }
    for flag, state in (
        ("camera_valid", lambda f: f["camera_valid"]),
        ("exported_xyz", lambda f: f["xyz_m"] is not None),
        ("observed", lambda f: f["observation"] == "detected"),
    ):
        table = controlled["frames"][flag]
        assert sum(table.values()) == 60
        both = [
            (state(a), state(b))
            for a, b in zip(frames["full"], frames["low"], strict=True)
        ]
        assert table == {
            "both": both.count((True, True)),
            "full_only": both.count((True, False)),
            "low_only": both.count((False, True)),
            "neither": both.count((False, False)),
        }
    # The cut removed the low cohort's cameras from frame 40 on.
    assert controlled["frames"]["camera_valid"]["full_only"] == 20
    assert controlled["frames"]["camera_valid"]["low_only"] == 0
    assert controlled["full"]["rows"]["camera_valid"] == 59
    assert controlled["full"]["physical"]["time_basis"] == "source_pts"
    assert (
        model["physical_counts"]["full"]["candidate_segments"]
        == (controlled["full"]["physical"]["counts"]["candidate_segments"])
    )
    assert result["notes"] == ["Mixed devices"]
    report = (tmp_path / "out/paired.md").read_text()
    assert "| camera_valid | 39 | 20 | 0 | 1 |" in report and "Mixed devices" in report


def test_inputs_outside_the_frozen_pairing_are_refused(evaluated, tmp_path):
    root, _, pairing = evaluated
    pairing_pin, inventories, models = arguments(evaluated)
    with pytest.raises(pairs.PairingError, match="pairing map differs"):
        pairs.paired((pairing, "0" * 64), inventories, models, tmp_path / "a")
    # A pairing map that names another source for the low clip.
    changed = tmp_path / "pairing.json"
    document = read_json(pairing)
    document["clips"][0]["source_sha256"] = "f" * 64
    write_json(changed, document)
    with pytest.raises(pairs.PairingError, match="pair differs"):
        pairs.paired(
            (changed, file_sha256(changed)), inventories, models, tmp_path / "b"
        )
    # A map that leaves a clip out.
    document = read_json(pairing)
    document["clips"].pop()
    write_json(changed, document, overwrite=True)
    with pytest.raises(pairs.PairingError, match="every clip exactly once"):
        pairs.paired(
            (changed, file_sha256(changed)), inventories, models, tmp_path / "c"
        )
    # A map that lists a pair twice.
    document = read_json(pairing)
    document["clips"].append(document["clips"][0])
    write_json(changed, document, overwrite=True)
    with pytest.raises(pairs.PairingError, match="every clip exactly once"):
        pairs.paired(
            (changed, file_sha256(changed)), inventories, models, tmp_path / "e"
        )
    run = root / "low/evaluated/clips/controlled"
    metrics = root / "low/evaluated/metrics.json"

    def refused(path, change, message, target):
        saved = path.read_bytes()
        change(path)
        try:
            with pytest.raises(pairs.PairingError, match=message):
                pairs.paired(pairing_pin, inventories, models, tmp_path / target)
        finally:
            path.chmod(0o600)
            path.write_bytes(saved)

    def shifted_rows(path):
        arrays, provenance = read_npz(path)
        write_npz(path, arrays | {"pts": arrays["pts"] + 7}, provenance, overwrite=True)

    def more_segments(path):
        report = read_json(path)
        report["counts"]["candidate_segments"] = 999999
        write_json(path, report, overwrite=True)

    def better_status(path):
        document = read_json(path)
        document["records"][0]["status"] = "ok"
        write_json(path, document, overwrite=True)

    # Stage rows changed without their recorded hash.
    refused(run / "frames.npz", shifted_rows, "differs from its recorded hash", "f")
    # A physical report that no longer has its released bytes.
    refused(run / "physical.json", more_segments, "release binding", "g")

    def moved_between_counts(path):
        # Totals stay the same; one fit moves between fitted and excluded.
        report = read_json(path)
        report["counts"]["fitted_segments"] += 1
        report["counts"]["excluded_segments"] -= 1
        write_json(path, report, overwrite=True)

    # Even with a binding that names the changed report, its counts must
    # follow its own segment statuses.
    binding_path = root / "low/product/release-binding.json"
    binding_saved = binding_path.read_bytes()
    report_saved = (run / "physical.json").read_bytes()
    moved_between_counts(run / "physical.json")
    binding = read_json(binding_path)
    binding["clips"][0]["artifacts"]["physical.json"]["sha256"] = file_sha256(
        run / "physical.json"
    )
    write_json(binding_path, binding, overwrite=True)
    try:
        with pytest.raises(pairs.PairingError, match="physical report"):
            pairs.paired(pairing_pin, inventories, models, tmp_path / "i")
    finally:
        binding_path.write_bytes(binding_saved)
        (run / "physical.json").write_bytes(report_saved)
    # A record that claims a better status than the released evaluation.
    refused(metrics, better_status, "release binding is for another root", "h")
    # A result changed after its evaluation.
    result = root / "low/evaluated/clips/controlled/result.json"
    saved = result.read_bytes()
    result.write_bytes(saved + b" ")
    try:
        with pytest.raises(
            pairs.PairingError, match="differs from the release binding"
        ):
            pairs.paired(pairing_pin, inventories, models, tmp_path / "d")
    finally:
        result.write_bytes(saved)
    assert not any((tmp_path / name).exists() for name in "abcdefghi")


def test_a_model_without_a_result_does_not_hide_the_other_models_pair():
    read = {"upstream_sha256": {"frames.npz": "a" * 64}}
    other = {"upstream_sha256": {"frames.npz": "b" * 64}}
    without_result = {"status": "failed"}
    assert pairs.same_stages([read, read])
    assert pairs.same_stages([read, without_result])
    assert pairs.same_stages([without_result, without_result])
    assert not pairs.same_stages([read, other])
    assert not pairs.same_stages([read, without_result, other])
