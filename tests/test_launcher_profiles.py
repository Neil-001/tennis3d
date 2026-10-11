"""The source launcher installs into its own folder and restarts without setup."""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "tennis3d" / "resources"
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX launcher")

# Stands in for uv. `venv` creates a Python that records its calls, reports
# weights as ready once `setup` has run, and exits 7 for the `fail` command.
FAKE_UV = r"""#!/bin/sh
log=$TENNIS3D_HOME/log
printf 'uv %s | cwd=%s cache=%s python=%s credentials=%s tmp=%s config=%s\n' "$*" "$PWD" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_CREDENTIALS_DIR" "$TMPDIR" "$UV_NO_CONFIG" >> "$log"
if [ "$1" = venv ]; then
  for last; do :; done
  mkdir -p "$last/bin"
  cat > "$last/bin/python" <<'PYTHON'
#!/bin/sh
log=$TENNIS3D_HOME/log
printf 'python %s | neo=%s cwd=%s path=%s\n' "$*" "${NEO_CACHE_DIR-unset}" "$PWD" "${PYTHONPATH-unset}" >> "$log"
case "$*" in
  "-m tennis3d setup --check") [ -f "$TENNIS3D_HOME/weights/ready" ] ;;
  "-m tennis3d setup")
    [ ! -f "$TENNIS3D_HOME/fail-weights" ] || exit 3
    mkdir -p "$TENNIS3D_HOME/weights" && : > "$TENNIS3D_HOME/weights/ready" ;;
  "-m tennis3d fail") exit 7 ;;
esac
PYTHON
  chmod +x "$last/bin/python"
fi
if [ "$1" = pip ] && [ -f "$TENNIS3D_HOME/fail-install" ]; then exit 1; fi
"""


class Clone:
    """A copy of the launcher and its locks in a folder whose name has a space."""

    def __init__(self, base: Path, name="source clone"):
        self.base = base
        self.root = base / name
        (self.root / "tennis3d" / "resources").mkdir(parents=True)
        shutil.copy2(ROOT / "tennis3d.sh", self.root)
        (self.root / "pyproject.toml").write_text("fixture")
        for lock in [*RESOURCES.glob("inference-*.txt"), RESOURCES / "build.txt"]:
            (self.root / "tennis3d" / "resources" / lock.name).write_text(lock.name)
        self.outside = base / "outside"
        self.outside.mkdir(exist_ok=True)
        binaries = base / "bin"
        binaries.mkdir(exist_ok=True)
        # Make the platform explicit so every profile is tested on macOS too.
        self.script(
            binaries / "uname",
            '#!/bin/sh\nif [ "$1" = -s ]; then echo Linux; else echo x86_64; fi\n',
        )
        self.path = str(binaries) + os.pathsep + os.environ["PATH"]
        self.script(self.runtime / "uv-0.12.18" / "uv", FAKE_UV)

    @property
    def home(self) -> Path:
        return self.root / "local-app"

    @property
    def runtime(self) -> Path:
        return self.home / "runtime"

    @staticmethod
    def script(path: Path, text: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(0o755)

    def run(self, *args, profile="cpu", **environment):
        # Ambient settings that would move the install must be ignored.
        ambient = {
            name: str(self.outside / name)
            for name in (
                "TENNIS3D_HOME",
                "UV_CACHE_DIR",
                "UV_PYTHON_INSTALL_DIR",
                "UV_PROJECT_ENVIRONMENT",
                "PYTHONPATH",
            )
        }
        return subprocess.run(
            ["sh", str(self.root / "tennis3d.sh"), *args],
            env=os.environ
            | ambient
            | {"TENNIS3D_PROFILE": profile, "PATH": self.path}
            | environment,
            cwd=self.outside,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def log(self) -> list[str]:
        return (self.home / "log").read_text().splitlines()


@pytest.fixture
def clone(tmp_path):
    return Clone(tmp_path)


def test_intel_lock_preserves_every_cpu_package_and_distribution_hash():
    cpu = (RESOURCES / "inference-cpu.txt").read_bytes()
    intel = (RESOURCES / "inference-intel.txt").read_bytes()
    assert intel.startswith(cpu)
    assert b"openvino==2026.4.1" in intel[len(cpu) :]


@pytest.mark.parametrize(
    "profile,index", [("cpu", "cpu"), ("cu126", "cu126"), ("intel", "cpu")]
)
def test_first_start_installs_inside_the_clone_and_the_next_start_skips_setup(
    clone, profile, index
):
    completed = clone.run(profile=profile)
    assert completed.returncode == 0, completed.stderr
    uv = [line for line in clone.log() if line.startswith("uv ")]
    python = [line.split(" | ")[0] for line in clone.log() if line.startswith("python")]
    runtime = clone.runtime
    settings = (
        f"cache={runtime}/cache python={runtime}/python "
        f"credentials={runtime}/credentials tmp={runtime}/tmp config=1"
    )
    assert len(uv) == 3 and all(line.endswith(settings) for line in uv)
    assert f"--python 3.12 {runtime}/venv" in uv[0]
    assert "--require-hashes" in uv[1]
    assert f"--extra-index-url https://download.pytorch.org/whl/{index} " in uv[1]
    assert f"-r {clone.root}/tennis3d/resources/inference-{profile}.txt" in uv[1]
    # The editable install names its paths relative to the clone, because uv
    # splits them at spaces.
    assert uv[2].startswith(
        f"uv pip install --python {runtime}/venv/bin/python --no-deps "
        f"--build-constraints tennis3d/resources/build.txt -e . | cwd={clone.root} "
    )
    assert python[-3:] == [
        "python -m tennis3d setup --check",
        "python -m tennis3d setup",
        f"python -m tennis3d dashboard --gallery {clone.root}/examples/gallery-v2/gallery.json",
    ]
    # The app runs from the caller's directory without an inherited import path.
    # Only the Intel profile needs its graphics compiler cache moved.
    cache = f"{runtime}/neo-cache" if profile == "intel" else "unset"
    assert clone.log()[-1].endswith(f"| neo={cache} cwd={clone.outside} path=unset")
    assert list(clone.outside.iterdir()) == []

    # The next start needs neither uv nor a weight download.
    shutil.rmtree(runtime / "uv-0.12.18")
    before = len(clone.log())
    completed = clone.run("reconstruct", "my clip.mp4", profile=profile)
    assert completed.returncode == 0, completed.stderr
    assert [line.split(" | ")[0] for line in clone.log()[before:]] == [
        "python -m tennis3d setup --check",
        "python -m tennis3d reconstruct my clip.mp4",
    ]
    assert clone.run("fail", profile=profile).returncode == 7


@pytest.mark.parametrize("stage", ["install", "weights"])
def test_failed_setup_is_not_accepted_and_the_next_start_completes_it(clone, stage):
    clone.home.mkdir(exist_ok=True)
    (clone.home / f"fail-{stage}").touch()
    failed = clone.run("version")
    assert failed.returncode != 0
    assert not clone.log()[-1].startswith("python -m tennis3d version")
    assert (clone.runtime / "installed").exists() == (stage == "weights")
    assert list((clone.runtime / "setup-locks").iterdir()) == []

    (clone.home / f"fail-{stage}").unlink()
    assert clone.run("version").returncode == 0
    assert (clone.home / "weights" / "ready").is_file()
    assert clone.log()[-1].startswith("python -m tennis3d version")
    environments = [line for line in clone.log() if line.startswith("uv venv")]
    assert len(environments) == (2 if stage == "install" else 1)


def test_only_changed_pins_profile_or_location_reinstall_and_app_data_survives(
    clone,
):
    def installs():
        return sum(line.startswith("uv venv") for line in clone.log())

    assert clone.run("version").returncode == 0
    run = clone.home / "runs" / "kept" / "result.json"
    run.parent.mkdir(parents=True)
    run.write_text("{}")

    # Source edits take effect without setup; the package runs from the clone.
    (clone.root / "tennis3d" / "cli.py").write_text("changed")
    assert clone.run("version").returncode == 0
    assert installs() == 1

    changed = 1
    for name in (
        "tennis3d/resources/inference-cpu.txt",
        "tennis3d/resources/build.txt",
        "pyproject.toml",
    ):
        (clone.root / name).write_text("new")
        assert clone.run("version").returncode == 0
        changed += 1
        assert installs() == changed
    assert clone.run("version", profile="cu126").returncode == 0
    assert installs() == changed + 1

    moved = clone.base / "moved clone"
    clone.root.rename(moved)
    clone.root = moved
    assert clone.run("version").returncode == 0
    assert installs() == changed + 2
    assert any(f"{moved}/local-app/runtime/venv" in line for line in clone.log())
    assert (moved / "local-app" / "runs" / "kept" / "result.json").read_text() == "{}"
    assert (moved / "local-app" / "weights" / "ready").is_file()
    assert list(clone.outside.iterdir()) == []


def test_an_explicit_setup_command_runs_without_the_automatic_weight_fetch(clone):
    (clone.home / "fail-weights").parent.mkdir(exist_ok=True)
    (clone.home / "fail-weights").touch()
    assert clone.run("setup", "--weights", "my weights", "--offline").returncode == 0
    calls = [line.split(" | ")[0] for line in clone.log() if line.startswith("python")]
    assert calls[-2:] == [
        "python -m tennis3d version",
        "python -m tennis3d setup --weights my weights --offline",
    ]


def test_a_second_start_gives_way_to_a_running_setup_and_ignores_a_dead_one(clone):
    # A marker left by a process that has gone must not block setup.
    gone = subprocess.Popen(["true"])
    gone.wait()
    markers = clone.runtime / "setup-locks"
    markers.mkdir()
    (markers / str(gone.pid)).touch()
    # The first start stays inside `uv venv` until the test lets it continue.
    clone.script(
        clone.runtime / "uv-0.12.18" / "uv",
        FAKE_UV
        + 'while [ "$1" = venv ] && [ ! -f "$TENNIS3D_HOME/continue" ]; do sleep 0.05; done\n',
    )
    first = subprocess.Popen(
        ["sh", str(clone.root / "tennis3d.sh"), "version"],
        env=os.environ | {"TENNIS3D_PROFILE": "cpu", "PATH": clone.path},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 20
        while not (clone.home / "log").exists():
            assert time.monotonic() < deadline and first.poll() is None
            time.sleep(0.05)
        second = clone.run("version")
        assert second.returncode == 2
        assert "Another Tennis3D setup is running" in second.stderr
        assert (clone.runtime / "venv" / "bin" / "python").exists()
    finally:
        (clone.home / "continue").touch()
        assert first.wait(timeout=20) == 0
    assert sum(line.startswith("uv venv") for line in clone.log()) == 1
    assert list(markers.iterdir()) == []


def test_a_setup_finished_by_another_start_while_waiting_is_not_rebuilt(clone):
    assert clone.run("version").returncode == 0
    stamp = clone.runtime / "installed"
    saved = clone.base / "stamp"
    stamp.rename(saved)
    # The other start finishes between this start's check and its turn.
    clone.script(
        clone.base / "bin" / "mkdir",
        f"#!/bin/sh\ncp '{saved}' '{stamp}'\nexec /bin/mkdir \"$@\"\n",
    )
    assert clone.run("version").returncode == 0
    assert sum(line.startswith("uv venv") for line in clone.log()) == 1


def test_downloaded_uv_must_match_its_pinned_checksum(clone):
    shutil.rmtree(clone.runtime / "uv-0.12.18")
    clone.script(
        clone.base / "bin" / "curl",
        "#!/bin/sh\n"
        'while [ "$#" -gt 1 ]; do [ "$1" = -o ] && out=$2; url=$2; shift; done\n'
        'printf "%s" "$url" > "$out"\n',
    )
    completed = clone.run("version")
    assert completed.returncode == 2
    assert "does not match its pinned SHA-256" in completed.stderr
    assert not (clone.runtime / "uv-0.12.18").exists()
    assert not (clone.runtime / "installed").exists()
    requested = (clone.runtime / "tmp" / "uv-download" / "uv.tar.gz").read_text()
    assert requested == (
        "https://github.com/astral-sh/uv/releases/download/0.12.18/"
        "uv-x86_64-unknown-linux-gnu.tar.gz"
    )


def test_intel_profile_rejects_unsupported_host_before_installing(clone):
    clone.script(clone.base / "bin" / "uname", "#!/bin/sh\necho Darwin\n")
    completed = clone.run("version", profile="intel")
    assert completed.returncode == 2
    assert "Linux x86_64 only" in completed.stderr
    assert not (clone.home / "log").exists()
