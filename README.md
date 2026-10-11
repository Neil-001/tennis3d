# Tennis3D

## Get started: launch the app

[View the 20 examples in your browser](https://neil-001.github.io/tennis3d/).
The hosted dashboard is read-only and needs no installation. To reconstruct
your own video, launch the local app below.

Download and extract this repository, or clone it:

```sh
git clone https://github.com/Neil-001/tennis3d.git
```

Then start the app from the repository folder.

**Windows.** Double-click `Tennis3D.bat`.

**Linux, or macOS on Apple silicon.** Run this in a terminal:

```sh
./tennis3d.sh
```

You do not need Python, uv, Node.js, a compiler or administrator rights. The
script works from any current directory, and the folder path may contain
spaces. It has no pinned setup for Intel Macs and stops with a message there.

The first run needs internet access. It downloads a pinned uv, a private Python
3.12, the locked packages and the model weights into `local-app/` in the
repository folder. On Linux without an NVIDIA GPU this takes about 1.5 GB.
When `nvidia-smi` works, the script installs the CUDA packages instead, which
are several gigabytes larger. Set `TENNIS3D_PROFILE=cpu` before the first run
to install the CPU packages anyway.

The app then opens in your browser at `http://127.0.0.1:8770/`, or another free
port. Choose a video to reconstruct it, then play the video beside its 3D ball
path. Stop the server with Ctrl+C.

Later runs use the same command and work offline. They skip the setup and start
the app.

The twenty precomputed examples are in the repository with their video clips,
98 MB in total. After the first run they play beside their 3D results with
nothing else to download. See
[the example gallery](examples/gallery-v2/README.md) for their sources.

### What the script writes

Everything the script and the app create is in `local-app/`: uv, Python, the
package cache, the model weights, your runs and your settings. The script
installs nothing outside the repository folder and does not change your
`PATH`, shell profile, registry or any other Python. Git ignores the folder.

To update, replace the source files or run `git pull`, and keep `local-app/`.
Your runs, settings and weights stay. If you move or rename the repository
folder, the next run needs internet access again. It rebuilds the Python
environment and reuses the downloaded packages and weights.

The examples are checked against the SHA-256 of their exact bytes, so Git
leaves the line endings of their results, timelines and clips alone. A clone
made before this rule, with Git's `core.autocrlf` on as Git for Windows
installs it, has results and timelines with Windows line endings. Every
example then reports `Example result does not match its recorded SHA-256`.
Run `git pull` and start the app again. At startup the app restores a result
or timeline when the restored bytes have the recorded SHA-256. It leaves any
other changed file alone and still refuses to show it. Git may then list the
restored files as modified although `git diff` shows no change in them;
`git add examples/gallery-v2` clears that.

To remove Tennis3D, delete the repository folder. To remove only the installed
runtime and your runs, delete `local-app/`.

Arguments after the script name run other commands with the same installation.
For example, `./tennis3d.sh setup --check` reports whether the packages and
weights are ready, and `Tennis3D.bat setup --check` does the same in a Windows
terminal.

Windows status: the Windows script has not been run on Windows. An earlier
packaged CPU build, which used the same package lock, was tested there.

## Essential commands

The commands below are for training, development and command-line use. They
need a separately installed
[uv](https://docs.astral.sh/uv/getting-started/installation/) and do not use
`local-app/`. Run them from the repository folder. Use
`uv run tennis3d --help` for all commands, or add `--help` after a command for
its options.

```sh
uv sync --locked --extra app
uv run tennis3d setup
```

These two commands install the app packages into `.venv/` and the model weights
into `~/.tennis3d`. On Windows, this route downloads the 2.6 GB CUDA-enabled
PyTorch package also when you use the CPU.

### Check your installation

```sh
uv run tennis3d setup --check
uv run tennis3d doctor --device cpu
```

`setup --check` checks that the app dependencies and weights are ready. `doctor`
reports the installed runtime. To probe the installed detectors, add
`--perception --assets-manifest "$HOME/.tennis3d/assets.yaml"`.

### Start training

This small CPU example generates synthetic data and trains a new lifter for ten
updates. It checks the training workflow; it does not produce a useful trained
model.

```sh
uv sync --locked --extra app --extra lifting --extra synthetic
uv run tennis3d simulate --config configs/simulation.yaml --output data/synthetic
uv run tennis3d train --config configs/lifting-smoke.yaml --output runs/train --device cpu
```

Resume from its checkpoint and extend the limits:

```sh
uv run tennis3d train --resume runs/train/checkpoints/last.pt --max-steps 20 --max-train-seconds 120
```

Score the checkpoint on the validation split:

```sh
uv run tennis3d evaluate --checkpoint runs/train/checkpoints/best.pt --manifest data/synthetic/val.manifest.json --output runs/train/evaluation.json --device cpu
```

For full training on the published
[N-il/tennis-3d-synthetic dataset](https://huggingface.co/datasets/N-il/tennis-3d-synthetic),
install the data extra and use the production recipe on a machine with an
NVIDIA GPU:

```sh
uv sync --locked --extra app --extra lifting --extra data
uv run tennis3d train --config configs/stage4-preparation-main-v1/full-run.yaml --output runs/full-training --device cuda
```

That recipe streams the pinned training and reserved-validation splits and
allows up to 83,334 updates. It needs substantial storage, downloads and GPU
time. Edit its YAML to choose your limits. `configs/lifter.yaml` supplies the
production architecture with smaller execution limits; `configs/lifting-smoke.yaml`
is the tiny CPU example. Paths in YAML are relative to your current directory.
The streamed-data tooling requires Linux or macOS.

### Reconstruct a video from the command line

After `tennis3d setup`, fetch the detector weights into the source checkout and
use the installed production lifter:

```sh
uv run tennis3d assets fetch --manifest configs/assets.yaml
uv run tennis3d reconstruct clip.mp4 --output runs/clip --checkpoint "$HOME/.tennis3d/weights/tennis3d-lifter-stage4.pt" --device cpu
```

The checkpoint path above uses the default app directory. If you set
`TENNIS3D_HOME`, use that directory's `weights` folder instead. Use a new output
folder for each clip. `runs/clip/result.json` keeps every source frame, including
gaps and their reasons. You can load that file in the app. Use `--device cuda`
for an available NVIDIA GPU; CPU is the portable default.

The app also offers optional Intel GPU detection on Linux x86_64 when the
`intel` extra and a compatible Intel compute driver are installed. The lifter
still runs on CPU. Intel device faults have occurred; use CPU or NVIDIA when
that backend fails. Installing the extra alone does not install a GPU driver.

## Set up a development environment

Clone the repository so Git hooks can be installed, then run:

```sh
uv sync --locked --all-extras
uv run pre-commit install
```

Run the checks before committing:

```sh
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
uv build
```

For frontend changes, install Node.js 22 or newer and build the standalone
viewer:

```sh
npm --prefix viewer ci
npm --prefix viewer run build
npm --prefix viewer test
```

The build writes `dist/viewer.html`. Open it locally to inspect a result JSON.
To build the read-only example dashboard for GitHub Pages, run
`npm --prefix viewer run build:pages`. Serve the resulting `_site/` directory
with a static HTTP server. The Pages workflow deploys it on pushes to `main`;
CI checks run only on pull requests.
Editable app and viewer files are in `viewer/`; the packaged dashboard is
rebuilt from them. Configuration files retained in `configs/` support CLI
commands, training, qualification scripts and regression fixtures.

## What the results mean

Tennis3D combines image detection, court calibration and a temporal model to
estimate a metric 3D ball path from a single broadcast camera. The released
lifter is the EMA export at training step 83,334. Its synthetic-test mean XYZ
error is 0.068 m on clean inputs and 0.099 m on corrupted inputs.

The twenty real examples contain 10,749 source frames, with exported XYZ on
10,134. Their untyped contact heuristic scores F1 0.59 against manual event
marks at a 100 ms tolerance. Contact timing uncertainty is uncalibrated, and
independent real XYZ accuracy is unmeasured. Failed camera estimates and
missing observations remain explicit gaps. See
[the example gallery](examples/gallery-v2/README.md) for source information.

## License

Project code and project-trained lifter weights are **GPL-3.0-only**. See
[LICENSE](LICENSE) and [third-party notices](licenses/README.md). Dependencies,
detector weights and footage keep their owners' terms. That includes the
example clips in `examples/gallery-v2/`. This project grants no rights to
third-party footage or weights.
