"""Portable private galleries and separately identified evaluation supplements."""

import hashlib
import html
import json
import shutil
from pathlib import Path

from tennis3d.artifacts import _loads, file_sha256, read_json
from tennis3d.evaluation.inventory import Clip, read_inventory
from tennis3d.schemas import Result, read_result


def verify_contact_inputs(real_path, contacts, *, packaged_real=None):
    """Reject changed positive and negative artifact evidence used by the adapter."""
    root = Path(real_path).resolve()
    for row in contacts["records"]:
        clip_id = row["clip_id"]
        # IDs were joined to schema-validated Clip instances before this call.
        run = root / "clips" / clip_id
        identity = row["artifact_identity"]
        hashes, absent = identity["files_sha256"], identity["absent_artifacts"]
        if set(hashes) & set(absent):
            raise ValueError("contact artifact cannot be both present and absent")
        for name in [*hashes, *absent]:
            if (
                Path(name).name != name
                or name in (".", "..")
                or "\\" in name
                or not (run / name).resolve().is_relative_to(root)
            ):
                raise ValueError("contact input artifact path differs")
        for name, digest in hashes.items():
            if file_sha256(run / name) != digest:
                raise ValueError("contact gallery input artifact changed")
            if packaged_real is not None:
                copied = Path(packaged_real) / "clips" / clip_id / name
                if copied.exists() and file_sha256(copied) != digest:
                    raise ValueError("packaged contact gallery artifact differs")
        if any((run / name).exists() for name in absent):
            raise ValueError("contact previously absent gallery artifact appeared")


def checked_supplements(
    real_path,
    real_result,
    contacts_path,
    stress_path,
    stress_inventory_path=None,
    *,
    real_metrics_bytes=None,
    stress_metrics_bytes=None,
):
    """Check joins and contact-file hashes before creating the package directory."""
    release = real_result["provenance"]["release"]
    contacts, stress, contact_files = None, None, None
    if contacts_path is not None:
        root = Path(contacts_path)
        metrics_bytes = (root / "metrics.json").read_bytes()
        contacts = _loads(metrics_bytes)
        if (
            contacts["checkpoint_sha256"] != release["checkpoint"]["sha256"]
            or contacts["gallery_provenance"]["release"] != release
            or contacts["gallery_metrics_sha256"]
            != hashlib.sha256(
                real_metrics_bytes
                if real_metrics_bytes is not None
                else (Path(real_path) / "metrics.json").read_bytes()
            ).hexdigest()
            or contacts["snapshot"]["inventory_sha256"]
            != real_result["inventory_sha256"]
        ):
            raise ValueError("contact supplement differs from the packaged gallery")
        real_by_id = {row["clip_id"]: row for row in real_result["records"]}
        rows = contacts["records"]
        if len(rows) != len(real_by_id) or {r["clip_id"] for r in rows} != set(
            real_by_id
        ):
            raise ValueError("contact supplement must retain every gallery clip")
        expected = {"metrics.json", "report.md"}
        for row in rows:
            clip = Clip.model_validate(real_by_id[row["clip_id"]]["source"])
            if (
                row["source_identity"]["source_sha256"] != clip.source_sha256
                or row["source_identity"]["timeline_sha256"] != clip.timeline_sha256
                or set(row["cohorts"]) != {"raw", "exported"}
            ):
                raise ValueError("contact source or raw/export cohorts differ")
            for cohort, values in row["cohorts"].items():
                relative = f"clips/{clip.clip_id}/contacts-{cohort}.json"
                if values["contact_path"] != relative:
                    raise ValueError("contact report path differs from clip identity")
                expected.add(relative)
                if set(values["physical_paths"]) != {"automatic", "manual_events"}:
                    raise ValueError("contact supplement needs both ODE cohorts")
                for segmentation, relative in values["physical_paths"].items():
                    wanted = (
                        f"clips/{clip.clip_id}/physical-{cohort}-{segmentation}.json"
                    )
                    if relative != wanted:
                        raise ValueError("contact ODE path differs from clip identity")
                    expected.add(relative)
        manifest_bytes = (root / "files.json").read_bytes()
        files = _loads(manifest_bytes)
        if not isinstance(files, dict):
            raise ValueError("contact file manifest must be a mapping")
        if set(files) != expected:
            raise ValueError("contact file manifest differs from declared reports")
        if files["metrics.json"] != hashlib.sha256(metrics_bytes).hexdigest():
            raise ValueError("contact metrics changed during validation")
        for relative, digest in files.items():
            path = root / relative
            if (
                not path.resolve().is_relative_to(root.resolve())
                or file_sha256(path) != digest
            ):
                raise ValueError("contact supplement file hash or location differs")
        contact_files = (files, manifest_bytes)
        verify_contact_inputs(real_path, contacts)
    if stress_path is not None:
        stress = _loads(
            stress_metrics_bytes
            if stress_metrics_bytes is not None
            else (Path(stress_path) / "metrics.json").read_bytes()
        )
        if (
            stress["kind"] != "end_to_end_real"
            or stress["provenance"]["release"] != release
            or stress["inventory"]["intended_role"] != "resolution_stress"
            or stress["inventory_sha256"] == real_result["inventory_sha256"]
        ):
            raise ValueError("resolution-stress supplement needs its own inventory")
        frozen_path = (
            Path(stress_inventory_path)
            if stress_inventory_path is not None
            else Path(stress_path) / "inventory.json"
        )
        if file_sha256(frozen_path) != stress["inventory_sha256"]:
            raise ValueError(
                "stress frozen inventory hash differs; supply --resolution-stress-inventory"
            )
        inventory = read_inventory(frozen_path)
        if read_inventory(Path(stress_path) / "inventory.json") != inventory:
            raise ValueError(
                "stress original inventory differs from run-local inventory"
            )
        rows = stress["records"]
        by_id = {row["clip_id"]: row for row in rows}
        if (
            inventory.summary() != stress["inventory"]
            or len(rows) != len(inventory.clips)
            or len(by_id) != len(rows)
            or set(by_id) != {clip.clip_id for clip in inventory.clips}
        ):
            raise ValueError("stress records must retain every frozen inventory clip")
        for clip in inventory.clips:
            if Clip.model_validate(by_id[clip.clip_id]["source"]) != clip:
                raise ValueError("stress record source differs from frozen inventory")
    return contacts, stress, contact_files


def copy_contact_reports(source, target, validated_files):
    """Copy only the exact manifest snapshot accepted during supplement validation."""
    root, target = Path(source), Path(target)
    files, manifest_bytes = validated_files
    if (root / "files.json").read_bytes() != manifest_bytes:
        raise ValueError("contact manifest changed during packaging")
    for relative, digest in files.items():
        destination = target / relative
        if not (root / relative).resolve().is_relative_to(
            root.resolve()
        ) or not destination.resolve().is_relative_to(target.resolve()):
            raise ValueError("contact copy path escapes its source or destination")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
        if file_sha256(destination) != digest:
            raise ValueError("contact supplement changed during packaging")
    (target / "files.json").write_bytes(manifest_bytes)


def copy_real_reports(source, target, result, checkpoint_sha256):
    """Copy confined byte snapshots and return their hashes for final verification."""
    root, target = Path(source).resolve(), Path(target).resolve()
    copied = {}

    def snapshot(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("real supplement copy source escapes its directory")
        return path.read_bytes()

    def copy_bytes(relative, payload):
        destination = target / relative
        if not destination.resolve().is_relative_to(target):
            raise ValueError("real supplement copy destination escapes its directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        copied[relative] = hashlib.sha256(payload).hexdigest()

    target.mkdir(parents=True, exist_ok=True)
    for filename in ("metrics.json", "report.md", "inventory.json", "config.json"):
        if (root / filename).is_file():
            copy_bytes(filename, snapshot(filename))
    for row in result["records"]:
        clip = Clip.model_validate(row["source"])
        if row["clip_id"] != clip.clip_id:
            raise ValueError("real supplement clip identity differs")
        paths = [
            (row.get("result_path"), "result.json", None),
            (row.get("physical_path"), "physical.json", "end_to_end"),
        ]
        if row.get("labelled_lifting"):
            paths.append(
                (
                    row["labelled_lifting"].get("physical_path"),
                    "labelled-physical.json",
                    "manual_2D_input",
                )
            )
        for relative, filename, mode in paths:
            if relative is None:
                continue
            if relative != f"clips/{clip.clip_id}/{filename}":
                raise ValueError("real supplement artifact path differs")
            payload = snapshot(relative)
            if filename == "result.json":
                artifact = Result.model_validate(_loads(payload))
                source_sha256 = artifact.source.sha256
                checkpoint = artifact.provenance.lifter_checkpoint_sha256
                if artifact.source.frame_count != clip.frame_count or (
                    artifact.source.width,
                    artifact.source.height,
                ) != (clip.width, clip.height):
                    raise ValueError("real supplement source dimensions/rows differ")
            else:
                artifact = _loads(payload)
                metadata = artifact["metadata"]
                if (
                    metadata["artifact_id"] != clip.clip_id
                    or metadata.get("mode") != mode
                ):
                    raise ValueError("real supplement physical clip differs")
                source_sha256 = metadata["source_identity"]["source_sha256"]
                checkpoint = metadata["prediction_identity"]["checkpoint_sha256"]
            if source_sha256 != clip.source_sha256 or checkpoint != checkpoint_sha256:
                raise ValueError("real supplement source/checkpoint differs")
            copy_bytes(relative, payload)
    return copied


def write_playable_gallery(root, result, viewer_path, *, title="Frozen real gallery"):
    """Embed each unchanged validated result in its own fully offline viewer."""
    root = Path(root)
    viewer = Path(viewer_path).read_text()
    cards = []
    seen = set()
    for row in result["records"]:
        clip = Clip.model_validate(row["source"])
        if row["clip_id"] != clip.clip_id or clip.clip_id in seen:
            raise ValueError("gallery clip identity differs")
        seen.add(clip.clip_id)
        label = html.escape(clip.clip_id)
        status = html.escape(row.get("status", "unavailable"))
        links, coverage = "Result unavailable", "Coverage unavailable"
        if row.get("result_path"):
            expected = f"clips/{clip.clip_id}/result.json"
            if row["result_path"] != expected:
                raise ValueError("playable gallery result path differs")
            data = read_result(root / expected)
            if (
                data.source.sha256 != clip.source_sha256
                or data.source.frame_count != clip.frame_count
                or (data.source.width, data.source.height) != (clip.width, clip.height)
                or data.provenance.lifter_checkpoint_sha256
                != result["provenance"]["release"]["checkpoint"]["sha256"]
            ):
                raise ValueError("playable gallery source/checkpoint differs")
            payload = json.dumps(read_json(root / expected), ensure_ascii=True)
            payload = payload.replace("<", "\\u003c").replace("&", "\\u0026")
            embedded = (
                '<script id="packaged-result" type="application/json">'
                + payload
                + "</script><script>"
                + "const input=document.getElementById('file');"
                + "const transfer=new DataTransfer();"
                + "transfer.items.add(new File([document.getElementById('packaged-result').textContent],"
                + "'result.json',{type:'application/json'}));"
                + "input.files=transfer.files;"
                + "input.dispatchEvent(new Event('change',{bubbles:true}));"
                + "</script>"
            )
            (root / expected).with_name("viewer.html").write_text(viewer + embedded)
            links = (
                f'<a href="clips/{label}/viewer.html">View trajectory</a> '
                f'<a href="{html.escape(expected, quote=True)}">Result JSON</a>'
            )
            coverage = (
                f"{data.summary.valid_frame_fraction:.1%} exported frame coverage"
            )
        reasons = html.escape("; ".join(row.get("failure_reasons", [])))
        cards.append(
            f"<article><h2>{label}</h2><p>{status} · {coverage}</p>"
            f"<p>{links}</p><p>{reasons}</p></article>"
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "gallery.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        f"<title>{html.escape(title)}</title><style>"
        "body{font:16px system-ui;margin:0;padding:24px;color:#dfebe9;background:#091820}"
        "main{max-width:1000px;margin:auto}h1{font-size:28px}h2{font-size:18px;overflow-wrap:anywhere}"
        "section{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:16px}"
        "article{border:1px solid #29434b;padding:16px;border-radius:8px;overflow-wrap:anywhere}"
        "a{color:#b9ea86;display:inline-block;margin-right:12px;margin-bottom:8px}"
        "</style></head><body><main>"
        f"<h1>{html.escape(title)}</h1>"
        "<p>Every preselected clip and failure is retained. Playback uses source timestamps and null gaps. "
        "Dense real XYZ accuracy is unmeasured. Original media and detector weights are excluded. "
        "These derived trajectories are private review artifacts.</p><section>"
        + "".join(cards)
        + "</section></main></body></html>"
    )
