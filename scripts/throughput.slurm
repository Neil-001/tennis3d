#!/usr/bin/env bash
#SBATCH --job-name=tennis-throughput
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@300
# Site allocation is supplied at submission.
# No data-master copy, validation, checkpoint warm start or full training.
set -euo pipefail
: "${TENNIS3D_ROOT:?isolated committed checkout}"
: "${TENNIS3D_THROUGHPUT_OUTPUT:?new durable qualification or existing campaign}"
python="${TENNIS3D_PYTHON:-$TENNIS3D_ROOT/.venv/bin/python}"
mode="${1:?qualify or trial}"
base="/tmp/${USER:?}/throughput-${SLURM_JOB_ID:?}"
[[ ! -e "$base" ]] || { echo 'owned job directory already exists' >&2; exit 2; }
umask 077
mkdir -p "$base"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONPATH="$TENNIS3D_ROOT"
local_run="$base/run"
durable="$TENNIS3D_THROUGHPUT_OUTPUT"
staged=0
evaluator_pid=""
finish() {
    rc=$?
    trap - EXIT
    if [[ "$staged" == 1 && -d "$local_run" ]]; then
        mkdir -p "$durable"
        rsync -a --exclude=/source-cache/ "$local_run/" "$durable/" || { echo 'final evidence sync failed' >&2; ((rc)) || rc=1; }
    fi
    # Retain partial local evidence on failure. Successful runs have durable copies.
    if ((rc == 0)); then rm -rf -- "$base"; fi
    exit "$rc"
}
trap finish EXIT
trap '[[ -z "$evaluator_pid" ]] || kill -TERM "$evaluator_pid" 2>/dev/null || true' USR1 TERM INT
run_evaluator() {
    local rc
    timeout --signal=TERM --kill-after=30s 3300s "$@" &
    evaluator_pid=$!
    while :; do
        wait "$evaluator_pid" && rc=0 || rc=$?
        kill -0 "$evaluator_pid" 2>/dev/null || break
    done
    evaluator_pid=""
    return "$rc"
}
case "$mode" in
    qualify)
        [[ ! -e "$durable" ]] || { echo 'qualification destination must be new' >&2; exit 2; }
        staged=1
        cd -- "$TENNIS3D_ROOT"
        run_evaluator "$python" -m tennis3d.throughput.evaluate qualify --source "$TENNIS3D_ROOT" --output "$local_run"
        ;;
    trial)
        trial="${2:?recorded campaign trial ID}"
        [[ "$trial" =~ ^[0-9]{6}$ ]] || exit 2
        [[ -d "$durable" ]] || exit 2
        rsync -a "$durable/" "$local_run/"
        staged=1
        # Run the immutable evaluator and libraries from the frozen oracle.
        export PYTHONPATH="$local_run/oracle"
        cd -- "$local_run/oracle"
        run_evaluator "$python" -m tennis3d.throughput.evaluate trial --run "$local_run" --trial "$trial" --cache-root "$base/cache"
        ;;
    *) exit 2 ;;
esac
