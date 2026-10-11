"""Small frozen envelopes only. No source trees or evaluator runs are needed."""

from pathlib import Path

from tennis3d.autoresearch.throughput_dashboard import canonical, digest
from tennis3d.throughput.campaign import _score


def create(run: Path):
    run.mkdir(parents=True)
    protocol = {
        "workloads": ["cold", "warm"],
        "min_paired_rounds": 2,
        "noise_threshold_fraction": 0.02,
        "sustained_noise_threshold_fraction": 0.02,
        "per_workload_total_noise_fraction": {"cold": 0.02, "warm": 0.03},
        "per_workload_sustained_noise_fraction": {"cold": 0.02, "warm": 0.03},
        "qualification": {
            "cold": [[100, 101], [100, 99]],
            "warm": [[10, 10], [10, 10]],
        },
        "benchmark": {
            "rules": {
                "steps": 128,
                "batch_size": 48,
                "gpu": "RTX 2080 Ti",
                "threads": 4,
                "memory_bytes": 34359738368,
                "max_frames": 250,
                "paired_rounds": 2,
                "steady_skip_steps": 8,
                "torch": "2.10.0+cu126",
                "cuda": "12.6",
                "revision": "main-data-pin",
                "descriptor_sha256": "d" * 64,
            }
        },
    }

    def write(name, value):
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical(value))
        return digest(path.read_bytes())

    write("protocol.json", protocol)
    initial = {
        "oracle_sha256": "a" * 64,
        "protocol_sha256": digest(canonical(protocol)),
    }
    entries = []

    def event(kind, payload):
        entry = {
            "sequence": len(entries),
            "previous": entries[-1]["sha256"] if entries else "0" * 64,
            "kind": kind,
            "payload": payload,
        }
        entry["sha256"] = digest(canonical(entry))
        entries.append(entry)

    event("initialize", initial)
    best = {
        "source_sha256": initial["oracle_sha256"],
        "trial_id": None,
        "result_sha256": None,
    }
    for index, factor, decision in [(1, 0.9, "retain"), (2, 0.95, "discard")]:
        identifier = f"{index:06d}"
        reservation = {
            "trial_id": identifier,
            "source_sha256": str(index) * 64,
            "incumbent_sha256": best["source_sha256"],
            "job_budget_s": 1800,
            "description": "Bounded batch prefetch",
        }
        write(f"trials/{identifier}/reservation.json", reservation)
        event("trial", reservation)
        identity = {
            **initial,
            **{
                k: reservation[k]
                for k in ("trial_id", "source_sha256", "incumbent_sha256")
            },
        }
        measurements = [
            {
                "workload": workload,
                "round": r,
                **{
                    f"{side}_{metric}": seconds * (factor if side == "candidate" else 1)
                    for side in ("candidate", "incumbent")
                    for metric, seconds in [
                        ("elapsed_s", 100 if index == 1 else 5),
                        ("stream_s", 80 if index == 1 else 4),
                        ("sustained_s", 60 if index == 1 else 3),
                    ]
                },
            }
            for workload in protocol["workloads"]
            for r in range(2)
        ]
        evidence = {
            **identity,
            "passed": True,
            "independent": True,
            "reports": [
                {
                    "workload": pair["workload"],
                    "round": pair["round"],
                    **{
                        side: {
                            **{
                                {
                                    "stream_s": "stream_elapsed_s",
                                    "sustained_s": "sustained_elapsed_s",
                                }.get(metric, metric): pair[f"{side}_{metric}"]
                                for metric in ("elapsed_s", "stream_s", "sustained_s")
                            },
                            "stall_mean_s": 0.01,
                            "stall_p95_s": 0.03,
                            "windows_per_second": 6144
                            / float(pair[f"{side}_elapsed_s"]),
                            "setup_s": 2 if index == 1 else 0.2,
                            "ram_peak_bytes": 10000,
                            "private_path": "/private/credential/location",
                        }
                        for side in ("candidate", "incumbent")
                    },
                }
                for pair in measurements
            ],
        }
        sha = write(f"trials/{identifier}/evidence.json", evidence)
        result = {
            **identity,
            "status": "completed",
            "gates": {
                "passed": True,
                "independent": True,
                "evidence_file": "evidence.json",
                "evidence_sha256": sha,
            },
            "measurements": measurements,
        }
        result["score"] = _score(protocol, result)
        result["recorded_at"] = "2026-10-06T15:00:00Z"
        result["job_accounting"] = {
            "basis": "slurm",
            "compute_s": 150,
            "queue_s": 25,
            "job_id": "123",
        }
        sha = write(f"trials/{identifier}/result.json", result)
        event("record", {"trial_id": identifier, "result_sha256": sha})
        event(decision, {"trial_id": identifier})
        if decision == "retain":
            best = {
                "trial_id": identifier,
                "source_sha256": reservation["source_sha256"],
                "result_sha256": sha,
            }
    write("best.json", best)
    (run / "journal.jsonl").write_bytes(b"".join(canonical(e) for e in entries))
    write(
        "cursor.json",
        {"sequence": entries[-1]["sequence"], "sha256": entries[-1]["sha256"]},
    )
    return entries


if __name__ == "__main__":
    import sys

    create(Path(sys.argv[1]))
