#!/usr/bin/env bash
#SBATCH --job-name=tennis-train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120
# Full lifter training that survives preemption and the per-job time limit.
# Supply your site's --account, --partition, GPU request (--gpus=... or
# --gres=gpu:1) and --output (the Slurm log) with sbatch. Keep site values out
# of this file. Submit the same command again (or let
# Slurm requeue it) to continue: the job resumes from the durable checkpoint.
#
# How resume works (tennis3d/lifting/train.py): the trainer
# publishes checkpoints/last.pt atomically every `checkpoint_every` steps and at
# the end, or at validations when checkpoint_every is unset, with accumulated
# training seconds inside it and in budget.json. The
# trainer stops cleanly after the current step on SIGTERM or SIGINT, saves
# last.pt and summary.json, and `tennis3d train --resume` continues from there
# with the accumulated budgets. A hard kill loses only the work since the last
# published checkpoint. Choose `checkpoint_every` in the config so that interval
# is a few minutes of training.
#
# Where the output lives: training writes the output directory on the node's
# local disk because the trainer writes progress files at every step, which is
# slow over a network file system. A background loop copies it to the durable
# directory every TENNIS3D_SYNC_SECONDS (default 60), checkpoints included, so
# a preempted or killed job loses at most that interval plus the steps since the
# last checkpoint. rsync publishes every file by renaming a finished temporary
# copy, so the durable best.pt and last.pt are never half written, and the loop
# copies them in dependency order (run files, best.pt, last.pt, summary.json) so
# a kill between two copies still leaves a resumable set.
#
# Time limit: by default 120 seconds before the limit Slurm sends USR1 to this
# script. It sends the trainer SIGTERM, waits for last.pt, runs the final rsync
# and requeues the job with `scontrol requeue`. Preemption (SIGTERM from Slurm)
# reaches the trainer directly, the script syncs and exits, and Slurm requeues
# the job itself. Override --signal=B:USR1@SECONDS when validation plus checkpoint
# publication and sync need more warning; the Stage 4 launch uses 900 seconds.
# A job whose trainer finished for a real reason (train budget,
# max steps, plateau) is not requeued and exits 0.
#
# Environment (paths must be visible from the compute node unless noted):
#   TENNIS3D_ROOT         staged git checkout on shared storage (required)
#   TENNIS3D_PYTHON       interpreter of the staged environment
#                         (default: $TENNIS3D_ROOT/.venv/bin/python)
#   TENNIS3D_DATA_MASTER  optional directory rsynced to node-local disk. It
#                         lands in $TENNIS3D_LOCAL/tennis-data/<its basename>;
#                         $TENNIS3D_ROOT/data must be a symlink to
#                         $TENNIS3D_LOCAL/tennis-data (created if missing).
#   TENNIS3D_LOCAL        node-local base directory (default: /tmp/$USER)
#   TENNIS3D_OUTPUT       durable output directory (required); the node-local
#                         working copy is $TENNIS3D_LOCAL/runs/<its basename>
#   TENNIS3D_CONFIG       training YAML, required to start a new run; optional
#                         when resuming (the checkpoint carries its config)
#   TENNIS3D_FORK_PLAN    optional frozen continuation plan. A new run then forks
#                         the plan's finished source checkpoint under the plan's
#                         own run identity instead of starting from random
#                         weights; later allocations resume it like any run,
#                         and refuse a checkpoint forked from another plan.
#   TENNIS3D_LOGS         directory for the GPU utilization log
#                         (default: $SLURM_SUBMIT_DIR)
#   TENNIS3D_SYNC_SECONDS seconds between syncs (default: 60)
#   TENNIS3D_BACKUP_STORE optional separate durable snapshot store (for example
#                         $TENNIS3D_OUTPUT.backups); a separate login-host uploader
#                         must run `python -m tennis3d.lifting.backup upload --watch`
#   TENNIS3D_BACKUP_SECONDS snapshot interval (default: 3600); allocation ends
#                         always request publication. Backup failures are visible
#                         and do not affect trainer exit status or requeue.
# Script arguments are passed to `tennis3d train` (for example --max-train-seconds
# 172800 --device cuda); --output and --resume are managed by this script.
set -euo pipefail
: "${TENNIS3D_ROOT:?Set TENNIS3D_ROOT to the staged checkout on shared storage}"
: "${TENNIS3D_OUTPUT:?Set TENNIS3D_OUTPUT to the durable output directory}"
root="$TENNIS3D_ROOT"
python="${TENNIS3D_PYTHON:-$root/.venv/bin/python}"
local_base="${TENNIS3D_LOCAL:-/tmp/${USER:-$(id -un)}}"
durable="${TENNIS3D_OUTPUT%/}"
local_out="$local_base/runs/$(basename -- "$durable")"
logs="${TENNIS3D_LOGS:-${SLURM_SUBMIT_DIR:-.}}"
sync_seconds="${TENNIS3D_SYNC_SECONDS:-60}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# Queue wait precedes this clock. Charge actual wrapper staging on every resume.
wrapper_started="$("$python" -c 'import time; print(time.monotonic())')"
staging_seconds=0

sync_up() (
    # Order matters: run files first, then best.pt, last.pt, and summary.json.
    [[ -d "$local_out" ]] || return 0
    mkdir -p "$durable"
    # A manual publisher uses the same external lock. Keep it outside the run.
    exec 9>"$durable.sync.lock"
    flock -x 9
    rsync -a --exclude='*.pt' --exclude=summary.json "$local_out/" "$durable/" || return 1
    local f
    for f in best.pt last.pt; do
        if [[ -f "$local_out/checkpoints/$f" ]]; then
            rsync -a "$local_out/checkpoints/$f" "$durable/checkpoints/$f" || return 1
        fi
    done
    if [[ -f "$local_out/summary.json" ]]; then
        rsync -a "$local_out/summary.json" "$durable/summary.json" || return 1
    fi
)
backup_publish() {
    [[ -n "${TENNIS3D_BACKUP_STORE:-}" ]] || return 0
    # Local copy/verification only. Never put Hub uploads in this allocation.
    (cd -- "$root" && "$python" -m tennis3d.lifting.backup publish \
        --source "$durable" --store "$TENNIS3D_BACKUP_STORE" --interval "$1" "${@:2}") || \
        echo "backup publication failed; durable source preserved; uploader/publication retry required" >&2
}
background_sync() {
    # TERM is handled between transfers, so a running rsync always finishes.
    local sleep_pid=''
    trap '[[ -z "$sleep_pid" ]] || kill "$sleep_pid" 2>/dev/null || true; exit 0' TERM
    while :; do
        sleep "$sync_seconds" &
        sleep_pid=$!
        wait "$sleep_pid"
        sleep_pid=''
        if sync_up; then
            backup_publish "${TENNIS3D_BACKUP_SECONDS:-3600}"
        else
            echo "periodic scratch sync failed; retrying next interval" >&2
        fi
    done
}
stop_reason() {
    # Prints the stop reason of the last finished trainer in a directory.
    "$python" - "$1/summary.json" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1]))["stop_reason"])
except (OSError, KeyError, ValueError):
    print("")
PY
}

gpu_pid="" sync_pid="" staged="" child="" want_requeue="" usr1="" terminated=""
finish() {
    local rc=$? final_started=''
    trap - EXIT
    if [[ -n "$staged" && -n "${TENNIS3D_BACKUP_STORE:-}" ]]; then
        final_started="$("$python" -c 'import time; print(time.monotonic())')" || \
            echo "backup final accounting clock failed; retry required" >&2
    fi
    if [[ -n "$sync_pid" ]]; then
        # Stop the loop and let an active periodic rsync finish first.
        kill "$sync_pid" 2>/dev/null || true
        wait "$sync_pid" 2>/dev/null || true
    fi
    [[ -z "$gpu_pid" ]] || kill "$gpu_pid" 2>/dev/null || true
    # Only a successfully staged local output may overwrite the durable one.
    if [[ -n "$staged" ]]; then
        if sync_up; then
            if [[ -n "$final_started" ]]; then
                backup_publish 0 --started "$final_started"
            else
                backup_publish 0
            fi
        else
            echo "final rsync failed" >&2
            ((rc)) || rc=1
        fi
        if [[ -n "$final_started" ]]; then
            # This work happens after the trainer timer stops. Its mutable ledger
            # carries the final sync/publication cost into ordinary scratch resume.
            (cd -- "$root" && "$python" -m tennis3d.lifting.backup account \
                --source "$durable" --store "$TENNIS3D_BACKUP_STORE" \
                --event "${SLURM_JOB_ID:-local}.${SLURM_RESTART_COUNT:-0}.$$.$final_started" \
                --started "$final_started") || \
                echo "backup final accounting failed; pending evidence preserved; retry required" >&2
        fi
    fi
    if [[ -n "$want_requeue" && "$rc" -eq 0 ]]; then
        echo "requeueing job ${SLURM_JOB_ID:-} to continue training"
        scontrol requeue "${SLURM_JOB_ID:?}" || rc=1
    fi
    echo "job finished with status $rc"
    exit "$rc"
}
trap finish EXIT
# Time limit warning: ask the trainer to stop after its current step.
trap 'usr1=1; [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true' USR1
# Preemption or cancel: Slurm also signals the trainer, so only record it.
trap 'terminated=1' TERM INT

echo "host: $(hostname)"
echo "uptime: $(uptime)"
echo "commit: $(git -C "$root" log --oneline -1)"
echo "run: durable=$durable local=$local_out restart=${SLURM_RESTART_COUNT:-0}"

# Already finished? Then there is nothing to do (a duplicate submission).
if [[ -f "$durable/summary.json" ]]; then
    reason="$(stop_reason "$durable")"
    if [[ -n "$reason" && "$reason" != "requested" ]]; then
        echo "training already finished: $reason"
        exit 0
    fi
fi

# Node-local data.
umask 077
mkdir -p "$local_base/tennis-data" "$local_base/runs"
chmod 700 "$local_base"
if [[ -n "${TENNIS3D_DATA_MASTER:-}" ]]; then
    data_name="$(basename -- "${TENNIS3D_DATA_MASTER%/}")"
    rsync -a --delete "${TENNIS3D_DATA_MASTER%/}/" "$local_base/tennis-data/$data_name/"
    if [[ ! -e "$root/data" && ! -L "$root/data" ]]; then
        ln -s "$local_base/tennis-data" "$root/data"
    elif [[ "$(readlink -f -- "$root/data")" != "$(readlink -f -- "$local_base/tennis-data")" ]]; then
        echo "$root/data must point to $local_base/tennis-data" >&2
        exit 2
    fi
fi

# Reconcile a write-ahead final charge before copying the resume ledger.
if [[ -n "${TENNIS3D_BACKUP_STORE:-}" && -f "$durable/checkpoints/last.pt" ]]; then
    if ! (cd -- "$root" && "$python" -m tennis3d.lifting.backup account \
        --source "$durable" --store "$TENNIS3D_BACKUP_STORE"); then
        echo "backup accounting reconciliation failed; source preserved; refusing undercharged resume" >&2
        exit 2
    fi
fi

# Resume from the durable checkpoint if it exists, else start a new run.
train_args=()
rm -rf -- "$local_out"
if [[ -f "$durable/checkpoints/last.pt" ]]; then
    rsync -a "$durable/" "$local_out/"
    train_args+=(--resume "$local_out/checkpoints/last.pt")
    # A resumed fork must be the run this plan started, never another lineage.
    [[ -z "${TENNIS3D_FORK_PLAN:-}" ]] || train_args+=(--expect-plan "$TENNIS3D_FORK_PLAN")
    [[ -z "${TENNIS3D_CONFIG:-}" ]] || train_args+=(--config "$TENNIS3D_CONFIG")
    echo "resuming from $durable/checkpoints/last.pt"
else
    : "${TENNIS3D_CONFIG:?Set TENNIS3D_CONFIG to start a new run}"
    # No checkpoint yet: discard leftovers of an attempt of this script that
    # never saved, but never a directory that does not look like a training run.
    if [[ -d "$durable" && -n "$(ls -A -- "$durable")" && ! -f "$durable/run.json" ]]; then
        echo "$durable is not empty and is not a training run directory" >&2
        exit 2
    fi
    rm -rf -- "$durable"
    mkdir -p "$local_out"
    train_args+=(--config "$TENNIS3D_CONFIG")
    if [[ -n "${TENNIS3D_FORK_PLAN:-}" ]]; then
        train_args+=(--fork "$TENNIS3D_FORK_PLAN")
        echo "forking a new run from $TENNIS3D_FORK_PLAN"
    else
        echo "starting a new run"
    fi
fi
train_args+=(--output "$local_out")
staged=1

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 5 \
    >"$logs/gpu-${SLURM_JOB_ID:-local}.csv" &
gpu_pid=$!
background_sync &
sync_pid=$!

cd -- "$root"
staging_seconds="$("$python" -c 'import time,sys; print(time.monotonic()-float(sys.argv[1]))' "$wrapper_started")"
echo "charged wrapper staging seconds: $staging_seconds"
"$python" -m tennis3d train "${train_args[@]}" "$@" --job-setup-seconds "$staging_seconds" &
child=$!
status=0
while :; do
    wait "$child" && status=0 || status=$?
    kill -0 "$child" 2>/dev/null || break
done
child=""
if ((status == 0)); then
    reason="$(stop_reason "$local_out")"
    echo "trainer stopped: ${reason:-unknown}"
    if [[ "$reason" == "requested" ]]; then
        if [[ -n "$usr1" ]]; then
            want_requeue=1
        else
            status=143 # preempted or cancelled; Slurm decides about requeue
        fi
    fi
fi
exit "$status"
