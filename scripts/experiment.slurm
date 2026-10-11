#!/usr/bin/env bash
#SBATCH --job-name=tennis-trial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
# One job runs ONE `tennis3d experiment` trial (one trial per job; run several
# trials by submitting several jobs against the same TENNIS3D_RUN, one after the
# other). Supply your site's --account, --partition, GPU request (--gpus=... or
# --gres=gpu:1) and --output (the Slurm log) with sbatch. Keep site values out
# of this file.
#
# Why node-local: the trainer writes progress files after every step and that
# time is charged to the trial's training budget. Over a network file system the
# run directory costs a third of the steps, and random reads of the dataset are
# slow too. The job therefore copies the data and the run directory to the
# node's local disk, runs the trial there, and copies the run directory back.
#
# Environment (all paths must be visible from the compute node unless noted):
#   TENNIS3D_ROOT         staged git checkout on shared storage (required). Must
#                         be clean and at a real commit; the tooling hashes it.
#   TENNIS3D_PYTHON       interpreter of the staged environment
#                         (default: $TENNIS3D_ROOT/.venv/bin/python)
#   TENNIS3D_DATA_MASTER  directory rsynced to node-local disk (required). It
#                         lands in $TENNIS3D_LOCAL/tennis-data/<its basename>.
#                         $TENNIS3D_ROOT/data must be a symlink to
#                         $TENNIS3D_LOCAL/tennis-data (created if missing), so
#                         the protocol's relative data paths resolve to the same
#                         absolute node-local path on every node.
#   TENNIS3D_LOCAL        node-local base directory (default: /tmp/$USER)
#   TENNIS3D_RUN          durable run directory on shared storage (required).
#                         Its node-local working copy is
#                         $TENNIS3D_LOCAL/runs/<basename of TENNIS3D_RUN>.
#   TENNIS3D_SEGMENT      comparison segment name. If the run has no segment
#                         of that name yet and --config is among the arguments,
#                         `tennis3d experiment-init` creates it first; the trial
#                         is passed `--segment`.
#   TENNIS3D_TRIAL        trial name (a child of the run directory). If unset,
#                         the basename of the --output argument is used.
#   TENNIS3D_LOGS         directory for the GPU utilization log
#                         (default: $SLURM_SUBMIT_DIR)
#   TENNIS3D_SYNC_SECONDS seconds between progress syncs (default: 10)
#   TENNIS3D_GPU_LOCK     passed through to scripts/experiment.local
#
# Arguments are those of `tennis3d experiment` (--config, --baseline,
# --hypothesis, ...). --output is rewritten to <node-local run>/<trial>, so a
# placeholder such as `--output x` is enough when TENNIS3D_TRIAL is set.
#
# Steps: copy data; copy an existing durable run directory to the node (a
# continuing campaign) and create the segment if it is missing; run the trial on local disk; every
# TENNIS3D_SYNC_SECONDS rsync the local run directory to the durable one without
# checkpoints (*.pt) for the dashboard; on exit, success or failure, rsync it
# once more in full. The job exits with the trial's status.
set -euo pipefail
: "${TENNIS3D_ROOT:?Set TENNIS3D_ROOT to the staged checkout on shared storage}"
: "${TENNIS3D_DATA_MASTER:?Set TENNIS3D_DATA_MASTER to the data directory to copy}"
: "${TENNIS3D_RUN:?Set TENNIS3D_RUN to the durable run directory}"
root="$TENNIS3D_ROOT"
python="${TENNIS3D_PYTHON:-$root/.venv/bin/python}"
local_base="${TENNIS3D_LOCAL:-/tmp/${USER:-$(id -un)}}"
durable="${TENNIS3D_RUN%/}"
local_run="$local_base/runs/$(basename -- "$durable")"
logs="${TENNIS3D_LOGS:-${SLURM_SUBMIT_DIR:-.}}"
sync_seconds="${TENNIS3D_SYNC_SECONDS:-10}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

# Rewrite --output, and find --config and --segment, in the argument list.
args=() config="" segment_given=""
trial="${TENNIS3D_TRIAL:-}"
while (($#)); do
    case "$1" in
        --output | --config | --segment)
            (($# >= 2)) || { echo "$1 needs a value" >&2; exit 2; }
            flag="$1" value="$2"
            shift 2
            ;;
        --output=* | --config=* | --segment=*)
            flag="${1%%=*}" value="${1#*=}"
            shift
            ;;
        *)
            args+=("$1")
            shift
            continue
            ;;
    esac
    case "$flag" in
        --output)
            [[ -n "$trial" ]] || trial="$(basename -- "$value")"
            ;;
        --config)
            config="$value"
            args+=(--config "$value")
            ;;
        --segment)
            segment_given=1
            args+=(--segment "$value")
            ;;
    esac
done
[[ -n "$trial" ]] || { echo "Set TENNIS3D_TRIAL or pass --output NAME" >&2; exit 2; }
args+=(--output "$local_run/$trial")
if [[ -z "$segment_given" && -n "${TENNIS3D_SEGMENT:-}" ]]; then
    args+=(--segment "$TENNIS3D_SEGMENT")
fi

gpu_pid="" sync_pid="" staged=""
background_sync() {
    # TERM is handled between transfers, so a running rsync always finishes.
    trap 'exit 0' TERM
    while :; do
        sleep "$sync_seconds" &
        wait $!
        rsync -a --exclude='*.pt' "$local_run/" "$durable/" 2>/dev/null || true
    done
}
finish() {
    local rc=$?
    trap - EXIT
    if [[ -n "$sync_pid" ]]; then
        # Stop the loop and let an active periodic rsync finish first.
        kill "$sync_pid" 2>/dev/null || true
        wait "$sync_pid" 2>/dev/null || true
    fi
    [[ -z "$gpu_pid" ]] || kill "$gpu_pid" 2>/dev/null || true
    # Only a successfully staged local run may overwrite the durable one.
    if [[ -n "$staged" && -d "$local_run" ]]; then
        # rsync publishes each file by renaming a temporary copy.
        rsync -a "$local_run/" "$durable/" || { echo "final rsync failed" >&2; ((rc)) || rc=1; }
    fi
    echo "job finished with status $rc"
    exit "$rc"
}
trap finish EXIT
# Slurm signals every process of the job, so the trial sees the signal too. The
# script keeps waiting for it to stop cleanly, then still runs the final rsync.
terminated=""
trap 'terminated=1' TERM INT

echo "host: $(hostname)"
echo "uptime: $(uptime)"
echo "commit: $(git -C "$root" log --oneline -1)"
echo "run: durable=$durable local=$local_run trial=$trial"

# Node-local data.
data_name="$(basename -- "${TENNIS3D_DATA_MASTER%/}")"
umask 077
mkdir -p "$local_base/tennis-data" "$local_base/runs"
chmod 700 "$local_base"
rsync -a --delete "${TENNIS3D_DATA_MASTER%/}/" "$local_base/tennis-data/$data_name/"
if [[ ! -e "$root/data" && ! -L "$root/data" ]]; then
    ln -s "$local_base/tennis-data" "$root/data"
elif [[ "$(readlink -f -- "$root/data")" != "$(readlink -f -- "$local_base/tennis-data")" ]]; then
    echo "$root/data must point to $local_base/tennis-data" >&2
    exit 2
fi

# Node-local run directory: continue the durable one, or start empty.
mkdir -p "$(dirname -- "$durable")"
if [[ -d "$durable" ]]; then
    rsync -a --delete "$durable/" "$local_run/"
else
    rm -rf -- "$local_run"
    mkdir -p "$local_run"
fi
staged=1

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 5 \
    >"$logs/gpu-${SLURM_JOB_ID:-local}.csv" &
gpu_pid=$!
background_sync &
sync_pid=$!

cd -- "$root"
run_child() {
    # Wait again if the shell's own signal trap interrupts the wait.
    local child rc
    "$@" &
    child=$!
    while :; do
        wait "$child" && rc=0 || rc=$?
        kill -0 "$child" 2>/dev/null || break
    done
    return "$rc"
}
if [[ -n "${TENNIS3D_SEGMENT:-}" && -n "$config" && ! -d "$local_run/segments/$TENNIS3D_SEGMENT" ]]; then
    run_child "$python" -m tennis3d experiment-init --config "$config" \
        --run "$local_run" --segment "$TENNIS3D_SEGMENT"
fi
status=0
run_child bash "$root/scripts/experiment.local" "${args[@]}" || status=$?
[[ -z "$terminated" ]] || ((status)) || status=143
exit "$status"
