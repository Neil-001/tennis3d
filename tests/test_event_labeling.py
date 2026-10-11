"""Source joins, durable persistence, conflict prevention and private HTTP routes."""

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from tennis3d.labeling.dashboard import Conflict, Gallery, handler, sha


@pytest.fixture
def gallery(tmp_path):
    pytest.importorskip("av")
    from event_labeling_fixture import make_gallery

    inventory = make_gallery(tmp_path)
    return Gallery(inventory, tmp_path / "previews", tmp_path / "labels")


def payload(gallery, events=None, revision=0):
    return {
        "revision": revision,
        "identity": gallery.identity("fixture0"),
        "events": events or [],
        "annotator": "synthetic verification",
    }


def test_exact_frame_mapping_and_preview(gallery):
    gallery.prepare()
    clip = gallery.clips["fixture0"]
    assert clip["frames"][0]["pts"] > 0
    assert clip["times"][0] == 0
    assert len({round(b - a, 5) for a, b in zip(clip["times"], clip["times"][1:])}) > 1
    assert gallery.image("fixture0", 0).startswith(b"\x89PNG")
    assert gallery.image("fixture0", 17).startswith(b"\x89PNG")
    event = gallery.event("fixture0", 17, "hit")
    assert event["original_frame_index"] == 88
    assert event["source_frame_index"] == 17
    assert event["pts"] == clip["frames"][17]["pts"]
    assert event["source_time_seconds"] > event["elapsed_clip_seconds"]
    receipt = json.loads(
        gallery.preview_path("fixture0").with_suffix(".json").read_text()
    )
    assert all(
        abs(a - b) <= 0.00051
        for a, b in zip(receipt["normalized_times"], clip["times"], strict=True)
    )
    assert receipt["preview_sha256"] == sha(gallery.preview_path("fixture0"))
    receipt["normalized_times"] = []
    receipt_path = gallery.preview_path("fixture0").with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt))
    gallery.prepare()
    assert (
        len(json.loads(receipt_path.read_text())["normalized_times"])
        == clip["frame_count"]
    )


def test_empty_nonempty_resume_and_correction(gallery):
    empty = gallery.save("fixture0", payload(gallery))
    assert empty["submitted"] and empty["events"] == [] and empty["revision"] == 1
    assert gallery.read("fixture1")["revision"] == 0
    events = [{"frame_index": 17, "type": "hit"}, {"frame_index": 17, "type": "bounce"}]
    saved = gallery.save("fixture0", payload(gallery, events, revision=1))
    resumed = Gallery(
        gallery.labels.parent / "inventory.json", gallery.cache, gallery.labels
    )
    assert resumed.read("fixture0") == saved
    assert resumed.catalog()["clips"][0]["submitted"]
    assert (
        json.loads((gallery.labels / "revisions/fixture0-r1.json").read_text()) == empty
    )
    with pytest.raises(Conflict):
        resumed.save("fixture0", payload(gallery, revision=1))
    assert resumed.read("fixture0") == saved


@pytest.mark.parametrize(
    "events",
    [
        [{"frame_index": -1, "type": "hit"}],
        [{"frame_index": 40, "type": "hit"}],
        [{"frame_index": True, "type": "hit"}],
        [{"frame_index": 0, "type": "uv"}],
        [{"frame_index": 0, "type": "hit"}] * 2,
        [{"frame_index": 0, "type": "hit", "path": "/tmp/elsewhere"}],
    ],
)
def test_invalid_events_rejected(gallery, events):
    with pytest.raises(ValueError):
        gallery.save("fixture0", payload(gallery, events))
    assert gallery.read("fixture0")["revision"] == 0


def test_identity_and_failed_atomic_write(gallery):
    invalid = payload(gallery)
    invalid["identity"]["source_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        gallery.save("fixture0", invalid)
    first = gallery.save("fixture0", payload(gallery))
    with (
        patch(
            "tennis3d.labeling.dashboard.os.replace",
            side_effect=OSError("fixture disk failure"),
        ),
        pytest.raises(OSError),
    ):
        gallery.save(
            "fixture0",
            payload(gallery, [{"frame_index": 1, "type": "bounce"}], revision=1),
        )
    assert gallery.read("fixture0") == first
    assert not list(gallery.labels.glob(".pending-*"))
    source = gallery.clips["fixture0"]["source_path"]
    with open(source, "ab") as file:
        file.write(b"corruption")
    with pytest.raises(ValueError, match="Source SHA256"):
        gallery.save("fixture0", payload(gallery, revision=1))
    assert gallery.read("fixture0") == first


def test_startup_rejects_changed_timeline(gallery):
    clip = gallery.clips["fixture0"]

    timeline = Path(clip["timeline_path"])
    content = json.loads(timeline.read_text())
    content["frames"][1]["original_frame_index"] = 999
    timeline.write_text(json.dumps(content))
    inventory = gallery.labels.parent / "inventory.json"
    value = json.loads(inventory.read_text())
    value["clips"][0]["timeline_sha256"] = sha(timeline)
    inventory.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="frame relationship"):
        Gallery(inventory, gallery.cache, gallery.labels)


def test_competing_process_revision_lock(gallery):
    other = Gallery(
        gallery.labels.parent / "inventory.json", gallery.cache, gallery.labels
    )
    results = []

    def save(instance):
        try:
            instance.save("fixture0", payload(gallery))
            results.append("saved")
        except Conflict:
            results.append("conflict")

    threads = [
        threading.Thread(target=save, args=(instance,)) for instance in [gallery, other]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == ["conflict", "saved"]


def test_http_private_routes_and_receipt(gallery):
    gallery.prepare()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(gallery))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = urllib.request.Request(
            base + "/api/clips/fixture0/labels",
            data=json.dumps(payload(gallery)).encode(),
            headers={
                "Content-Type": "application/json",
                "Origin": base,
                "X-Label-Token": gallery.token,
            },
        )
        with urllib.request.urlopen(request) as response:
            receipt = json.load(response)
        assert receipt["saved"] and receipt["labels"] == gallery.read("fixture0")
        # The page's theme, typeface and font license are served locally.
        for path, kind in [
            ("/", "text/html"),
            ("/theme.css", "text/css"),
            ("/dashboard.css", "text/css"),
            ("/GoogleSansCode-latin.woff2", "font/woff2"),
            ("/font-license.txt", "text/plain"),
        ]:
            with urllib.request.urlopen(base + path) as response:
                assert response.headers["Content-Type"].startswith(kind)
                assert response.read()
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/dashboard.py")
        for path in [
            "/api/clips/../../etc/passwd",
            "/api/clips/fixture0/frames/40",
            "/api/clips/unknown/video",
        ]:
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(base + path)
        hostile = urllib.request.Request(
            base + "/api/gallery", headers={"Host": "evil.example"}
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(hostile)
        assert error.value.code == 403
        original = base + "/api/clips/fixture0/frames/17"
        with urllib.request.urlopen(original) as response:
            assert response.headers["Cache-Control"] == "no-store"
        with urllib.request.urlopen(
            original + "?identity=" + gallery.inventory_sha256
        ) as response:
            assert "immutable" in response.headers["Cache-Control"]
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(original + "?identity=wrong")
        request.headers["X-label-token"] = "stale-before-restart"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
        assert gallery.read("fixture0")["revision"] == 1
        request.headers["X-label-token"] = gallery.token
        request.headers["Origin"] = "http://evil.example"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
        range_request = urllib.request.Request(
            base + "/api/clips/fixture0/video", headers={"Range": "bytes=0-99"}
        )
        with urllib.request.urlopen(range_request) as response:
            assert response.status == 206 and len(response.read()) == 100
    finally:
        server.shutdown()
        server.server_close()


def test_lossless_prepared_frames_reuse_and_integrity(gallery):
    import hashlib
    import io

    import av

    gallery.prepare()
    for cid, clip in gallery.clips.items():
        for index in [0, 17, 39]:
            data = gallery.image(cid, index)
            with av.open(io.BytesIO(data), mode="r") as source:
                frame = next(source.decode(video=0))
                assert (
                    hashlib.sha256(
                        frame.to_ndarray(format="rgb24").tobytes()
                    ).hexdigest()
                    == clip["frames"][index]["decoded_rgb_sha256"]
                )
    # A prepared restart does not reopen and seek video for each requested image.
    resumed = Gallery(
        gallery.labels.parent / "inventory.json", gallery.cache, gallery.labels
    )
    with patch("av.open", side_effect=AssertionError("unexpected source decode")):
        resumed.prepare()
        resumed.image("fixture0", 17)
    path = resumed.originals.directory(resumed.clips["fixture0"]) / "18.png"
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="cache hash mismatch"):
        resumed.image("fixture0", 18)


def test_frame_cache_budgets_and_save_lock_independence(gallery):
    gallery.prepare()
    gallery.originals.MEMORY_BYTES = 400
    for index in range(40):
        gallery.image("fixture0", index)
        assert gallery.originals.memory_bytes <= 400
    entered, release = threading.Event(), threading.Event()
    original = gallery.originals.directory(gallery.clips["fixture0"])

    read_bytes = Path.read_bytes

    def delayed(path):
        if path == original / "0.png":
            entered.set()
            assert release.wait(5)
        return read_bytes(path)

    gallery.originals.images.clear()
    gallery.originals.memory_bytes = 0
    with patch.object(Path, "read_bytes", delayed):
        thread = threading.Thread(target=gallery.image, args=("fixture0", 0))
        thread.start()
        assert entered.wait(2)
        try:
            assert gallery.save("fixture0", payload(gallery))["revision"] == 1
        finally:
            release.set()
            thread.join(5)
    limited = Gallery(
        gallery.labels.parent / "inventory.json",
        gallery.cache,
        gallery.labels,
        gallery.cache / "limited",
        frame_cache_bytes=100,
    )
    with pytest.raises(ValueError, match="disk budget"):
        limited.prepare()
    assert (
        sum(p.stat().st_size for p in limited.originals.root.rglob("*") if p.is_file())
        <= 100
    )


def test_frame_cache_does_not_rewrite_existing_submissions(gallery):
    first = gallery.save(
        "fixture0", payload(gallery, [{"frame_index": 17, "type": "hit"}])
    )
    second = gallery.save(
        "fixture0",
        payload(gallery, [{"frame_index": 18, "type": "bounce"}], revision=1),
    )
    before = {
        p.relative_to(gallery.labels): p.read_bytes()
        for p in gallery.labels.rglob("*.json")
    }
    resumed = Gallery(
        gallery.labels.parent / "inventory.json", gallery.cache, gallery.labels
    )
    resumed.prepare()
    assert before == {
        p.relative_to(gallery.labels): p.read_bytes()
        for p in gallery.labels.rglob("*.json")
    }
    assert resumed.read("fixture0") == second
    assert (
        json.loads((gallery.labels / "revisions/fixture0-r1.json").read_bytes())
        == first
    )


def test_sigterm_drains_inflight_submission(gallery, tmp_path):
    repo = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(repo), environment.get("PYTHONPATH")])
    )
    gallery.prepare()
    log = tmp_path / "server.log"
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tennis3d.labeling.dashboard",
                "--inventory",
                str(gallery.labels.parent / "inventory.json"),
                "--cache",
                str(gallery.cache),
                "--labels",
                str(gallery.labels),
                "--port",
                "0",
            ],
            cwd=repo,
            env=environment,
            stdout=output,
            stderr=output,
        )
    try:
        deadline = time.monotonic() + 10
        while "Labeling dashboard: " not in log.read_text():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        base = log.read_text().split("Labeling dashboard: ")[1].splitlines()[0]
        with urllib.request.urlopen(base + "/api/gallery") as response:
            token = json.load(response)["token"]
        result = []
        started = threading.Event()

        def submit():
            request = urllib.request.Request(
                base + "/api/clips/fixture0/labels",
                data=json.dumps(payload(gallery)).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Origin": base,
                    "X-Label-Token": token,
                },
            )
            started.set()
            with urllib.request.urlopen(request, timeout=10) as response:
                result.append(json.load(response))

        with (gallery.labels / ".save.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            thread = threading.Thread(target=submit)
            thread.start()
            assert started.wait(2)
            # Allow the localhost request to enter the cross-process revision lock.
            time.sleep(0.15)
            os.kill(process.pid, signal.SIGTERM)
            time.sleep(0.15)
            assert process.poll() is None
            fcntl.flock(lock, fcntl.LOCK_UN)
        thread.join(10)
        assert result[0]["saved"]
        assert process.wait(10) == 0
        assert gallery.read("fixture0")["revision"] == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_bundled_typeface_keeps_its_license():
    root = Path(__file__).parents[1]
    theme = root / "tennis3d" / "resources" / "theme"
    notice = (theme / "OFL.txt").read_text()
    assert "SIL OPEN FONT LICENSE Version 1.1" in notice
    assert notice == (root / "licenses" / "Google-Sans-Code-OFL-1.1.txt").read_text()
    assert "url(GoogleSansCode-latin.woff2)" in (theme / "theme.css").read_text()
    # The self-contained app page embeds the font and its license text.
    page = (root / "tennis3d" / "resources" / "dashboard.html").read_text()
    assert "data:font/woff2;base64," in page
    assert "SIL OPEN FONT LICENSE Version 1.1" in page


def test_theme_text_colours_meet_contrast_minimum():
    """Small text tokens keep 4.5:1 on every surface, in light and dark."""
    import re

    css = (
        Path(__file__).parents[1] / "tennis3d" / "resources" / "theme" / "theme.css"
    ).read_text()
    light, dark = (
        dict(re.findall(r"--([a-z-]+):(#[0-9a-f]{6})\b", block))
        for block in css.split("@media(prefers-color-scheme:dark)")[:2]
    )
    dark = light | dark

    def luminance(colour):
        parts = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        r, g, b = (
            c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in parts
        )
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    for tokens in (light, dark):
        for text in ("fg", "muted", "accent-text", "warn", "bad"):
            for surface in ("bg", "card", "bar", "soft"):
                high, low = sorted(
                    (luminance(tokens[text]), luminance(tokens[surface])), reverse=True
                )
                assert (high + 0.05) / (low + 0.05) >= 4.5, (text, surface)
