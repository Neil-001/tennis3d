#!/bin/sh
# Start Tennis3D from this source folder on Linux or macOS.
#
# The first run downloads a pinned uv, a private Python, the locked packages
# and the model weights into local-app/ beside this script, then opens the
# app. Later runs start offline. Nothing is installed or changed outside this
# folder, and deleting local-app/ removes everything this script created.
#
#   ./tennis3d.sh                    open the app with the bundled examples
#   ./tennis3d.sh COMMAND [ARGS]     run any other tennis3d command
set -eu
root=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
app_home=$root/local-app
runtime=$app_home/runtime
uv_version=0.12.18
python_version=3.12

fail() {
  echo "Tennis3D: $*" >&2
  exit 2
}

system=$(uname -s)
machine=$(uname -m)
profile=${TENNIS3D_PROFILE:-}
if [ -z "$profile" ]; then
  if [ "$system" = Linux ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then profile=cu126; else profile=cpu; fi
fi
case "$profile" in cpu|cu126|intel) ;; *) fail "TENNIS3D_PROFILE must be cpu, cu126 or intel." ;; esac
torch_profile=$profile
if [ "$profile" = intel ]; then
  if [ "$system" != Linux ] || [ "$machine" != x86_64 ]; then
    fail "The Intel profile supports Linux x86_64 only. Use cpu or cu126 on this platform."
  fi
  torch_profile=cpu
fi
lock=$root/tennis3d/resources/inference-$profile.txt
build_lock=$root/tennis3d/resources/build.txt
[ -f "$lock" ] && [ -f "$build_lock" ] && [ -f "$root/pyproject.toml" ] || fail "Part of this folder is missing. Download the whole repository and run this script from it."

# Only this folder's uv settings apply, so an ambient uv, Python or Tennis3D
# configuration cannot move the install elsewhere. These changes last for this
# process only.
for name in $(env | sed -n 's/^\(UV_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$name"; done
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV
export TENNIS3D_HOME="$app_home"
export UV_NO_CONFIG=1 UV_MANAGED_PYTHON=1 UV_PYTHON_INSTALL_BIN=0
export UV_CACHE_DIR="$runtime/cache" UV_PYTHON_INSTALL_DIR="$runtime/python"
export UV_PYTHON_BIN_DIR="$runtime/python-bin" UV_TOOL_DIR="$runtime/tools" UV_TOOL_BIN_DIR="$runtime/tools-bin"
export UV_CREDENTIALS_DIR="$runtime/credentials"
# Run this source tree only: no user site-packages or current-directory imports,
# with bytecode and framework caches kept in local-app/.
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPYCACHEPREFIX="$runtime/pycache"
export TORCH_HOME="$runtime/torch" PYTORCH_KERNEL_CACHE_PATH="$runtime/torch/kernels" CUDA_CACHE_PATH="$runtime/cuda"
if [ "$profile" = intel ]; then
  # Importing OpenVINO reports usage and writes ~/intel unless it sees a CI
  # run; it has no other switch that leaves the home directory alone. The
  # Intel graphics compiler cache defaults to the home directory too.
  export CI=true NEO_CACHE_DIR="$runtime/neo-cache"
  mkdir -p "$NEO_CACHE_DIR"
fi

uv=$runtime/uv-$uv_version/uv
python=$runtime/venv/bin/python
stamp=$runtime/installed
setup_locks=$runtime/setup-locks
# The environment holds absolute paths, so a moved folder is installed again.
identity="$profile uv-$uv_version python-$python_version $(cat "$lock" "$build_lock" "$root/pyproject.toml" | cksum | tr ' ' -) $root"

# Setup commands keep their temporary files in local-app/ too.
locally() {
  env TMPDIR="$runtime/tmp" "$@"
}

fetch_uv() {
  case "$system-$machine" in
    Linux-x86_64) target=x86_64-unknown-linux-gnu sha256=89eadd7c76fc063887959510d5ba0ab1264dfd5f1143b925ddb73021a40acf16 ;;
    Linux-aarch64|Linux-arm64) target=aarch64-unknown-linux-gnu sha256=afb6291f3f0a6b4521fc67b947822506c41dde5b60d2189dd8f3695b2ac8c9e7 ;;
    Darwin-arm64) target=aarch64-apple-darwin sha256=cf40e0c6a202190ccd9e0406dcfdd5b2d6668a9a5c779b17948963df32aafe5b ;;
    *) fail "No pinned setup exists for $system $machine." ;;
  esac
  url=https://github.com/astral-sh/uv/releases/download/$uv_version/uv-$target.tar.gz
  staging=$runtime/tmp/uv-download
  rm -rf "$staging"
  mkdir -p "$staging"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf --retry 3 -o "$staging/uv.tar.gz" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --no-hsts -O "$staging/uv.tar.gz" "$url"
  else
    fail "curl or wget is needed to download the setup tools."
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$staging/uv.tar.gz")
  else
    actual=$(shasum -a 256 "$staging/uv.tar.gz")
  fi
  [ "${actual%% *}" = "$sha256" ] || fail "The downloaded uv $uv_version does not match its pinned SHA-256."
  tar -xzf "$staging/uv.tar.gz" -C "$staging"
  rm -rf "$runtime/uv-$uv_version"
  mv "$staging/uv-$target" "$runtime/uv-$uv_version"
  rm -rf "$staging"
}

installed() {
  [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$identity" ] && [ -x "$python" ]
}

# One setup at a time: a second start would delete the environment the first
# is building. Each start adds a marker named after its process and gives way
# to any other live one, so two starts can never both proceed. A start removes
# only its own marker and those of processes that have gone.
begin_setup() {
  mkdir -p "$setup_locks"
  : >"$setup_locks/$$"
  trap 'rm -f "$setup_locks/$$"' EXIT
  trap 'exit 130' INT TERM HUP
  for marker in "$setup_locks"/*; do
    owner=${marker##*/}
    [ "$owner" != "$$" ] || continue
    if kill -0 "$owner" 2>/dev/null; then
      fail "Another Tennis3D setup is running in this folder. Wait for it to finish, then start again."
    fi
    rm -f "$marker"
  done
}

end_setup() {
  rm -f "$setup_locks/$$"
  trap - EXIT INT TERM HUP
}

install() {
  echo "Setting up Tennis3D ($profile) in $app_home. This needs an internet connection once."
  mkdir -p "$runtime/tmp"
  # The stamp is written last, so an interrupted setup starts again here.
  rm -f "$stamp"
  [ -x "$uv" ] || fetch_uv
  rm -rf "$runtime/venv"
  locally "$uv" venv --quiet --no-project --python "$python_version" "$runtime/venv"
  locally "$uv" pip install --python "$python" --require-hashes \
    --extra-index-url "https://download.pytorch.org/whl/$torch_profile" --index-strategy unsafe-best-match \
    -r "$lock"
  # An editable install runs the package from this folder, so a source update
  # needs no reinstall. uv splits these paths at spaces, so they are relative.
  (cd "$root" && locally "$uv" pip install --python "$python" --no-deps --build-constraints tennis3d/resources/build.txt -e .)
  locally "$python" -m tennis3d version >/dev/null
  printf '%s\n' "$identity" >"$stamp.new"
  mv "$stamp.new" "$stamp"
}

if ! installed; then
  begin_setup
  # Another start may have finished the setup before this one took its turn.
  installed || install
  end_setup
fi

# An explicit `setup` command chooses how to get the weights itself.
if [ "${1:-}" != setup ] && ! "$python" -m tennis3d setup --check >/dev/null 2>&1; then
  echo "Fetching the model weights into $app_home/weights."
  mkdir -p "$runtime/tmp"
  locally "$python" -m tennis3d setup
fi
if [ "$#" -gt 0 ]; then exec "$python" -m tennis3d "$@"; fi
exec "$python" -m tennis3d dashboard --gallery "$root/examples/gallery-v2/gallery.json"
