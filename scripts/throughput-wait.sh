#!/usr/bin/env bash
# Submits one throughput job on a Slurm login host and waits for it to finish.
# Arguments can also be supplied with THROUGHPUT_* variables, for runners that
# read this script through stdin and do not forward positional arguments.
set -euo pipefail
root="${THROUGHPUT_ROOT:?separate staged Git checkout}"
output="${THROUGHPUT_OUTPUT:?durable qualification or campaign path}"
record="${THROUGHPUT_JOB_RECORD:?new durable job ID file}"
mode="${THROUGHPUT_MODE:-qualify}"
trial="${THROUGHPUT_TRIAL:-}"
case "$mode" in
    qualify) [[ -z "$trial" ]] || { echo 'qualification does not accept a trial ID' >&2; exit 2; } ;;
    trial) [[ "$trial" =~ ^[0-9]{6}$ ]] || { echo 'trial mode needs a six-digit trial ID' >&2; exit 2; } ;;
    *) echo 'mode must be qualify or trial' >&2; exit 2 ;;
esac
mkdir -p "$(dirname -- "$record")"
exec 9>"$record.lock"
flock 9
exec 8>"$(dirname -- "$root")/one-gpu.lock"
flock 8
# A submission interruption never authorizes another job. Recover the ID, then
# reconcile that exact allocation. An empty record is uncertain and fails closed.
if [[ -e "$record" ]]; then
    [[ -s "$record" ]] || { echo 'uncertain submission; reconcile by unique job name' >&2; exit 2; }
    job="$(cat "$record")"
else
    : > "$record"
    status=0
    args=("$mode")
    [[ -z "$trial" ]] || args+=("$trial")
    export TENNIS3D_ROOT="$root" TENNIS3D_THROUGHPUT_OUTPUT="$output"
    sbatch --wait --parsable -A gpu-students -p studentkillable \
        --gpus=geforce_rtx_2080:1 --cpus-per-task=4 --mem=32G --time=01:00:00 \
        --no-requeue --job-name="throughput-$(basename -- "$record" .id)" \
        --output="$record.slurm.log" "$root/scripts/throughput.slurm" "${args[@]}" \
        | tee "$record" || status=$?
    [[ -s "$record" ]] || { echo 'submission ID unknown; reconcile before retry' >&2; exit 2; }
    job="$(cat "$record")"
fi
cluster=""
[[ "$job" =~ ^[0-9]+(\;[A-Za-z0-9_.-]+)?$ ]] || { echo 'invalid recorded Slurm ID' >&2; exit 2; }
IFS=';' read -r job cluster extra <<< "$job"
[[ -z "${extra:-}" && "$job" =~ ^[0-9]+$ && ( -z "$cluster" || "$cluster" =~ ^[A-Za-z0-9_.-]+$ ) ]] || { echo 'invalid recorded Slurm ID' >&2; exit 2; }
cluster_args=()
[[ -z "$cluster" ]] || cluster_args=(--clusters="$cluster")
# Recovery is deterministic waiting, never a second sbatch.
while [[ -n "$(squeue "${cluster_args[@]}" -h -j "$job")" ]]; do sleep 20; done
sacct "${cluster_args[@]}" -j "$job" --format=JobID,State,ExitCode,Elapsed,NodeList -n
state="$(sacct "${cluster_args[@]}" -j "$job" -X -n -o State | tr -d '[:space:]')"
accounting="$output/slurm-accounting.json"
[[ "$mode" != trial ]] || accounting="$output/trials/$trial/slurm-accounting.json"
mkdir -p "$(dirname -- "$accounting")"
sacct "${cluster_args[@]}" -j "$job" -X -n -P -o JobIDRaw,State,ExitCode,ElapsedRaw,Submit,Start,End,NodeList,AllocCPUS,ReqMem > "$record.accounting.txt"
python3 - "$job" "$record.accounting.txt" "$accounting" <<'PYACCOUNT'
import datetime,json,sys
from pathlib import Path
job,record,out=sys.argv[1:]
rows=[line.split('|') for line in Path(record).read_text().splitlines()]
rows=[row for row in rows if row[0]==job]
if len(rows)!=1: raise SystemExit('uncertain Slurm accounting; reconcile recorded job')
row=rows[0]
submit,start,end=(datetime.datetime.fromisoformat(value) for value in row[4:7])
value={'basis':'slurm','job_id':job,'state':row[1],'exit_code':row[2], 'compute_s':float(row[3]),'queue_s':(start-submit).total_seconds(),'submit':row[4],'start':row[5],'end':row[6],'node':row[7],'cpus':row[8],'requested_memory':row[9]}
Path(out).write_text(json.dumps(value,sort_keys=True)+'\n')
PYACCOUNT
[[ "$state" == COMPLETED ]] || { echo "recorded job $job ended $state" >&2; exit 1; }
