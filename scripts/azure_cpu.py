"""Bounded Azure CPU host lifecycle. No generation or publication implementation."""

import argparse
import base64
import datetime as dt
import hashlib
import ipaddress
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

REGIONS = {
    "switzerlandnorth",
    "denmarkeast",
    "polandcentral",
    "spaincentral",
    "israelcentral",
}
TYPES = {
    "vm": "Microsoft.Compute/virtualMachines",
    "os": "Microsoft.Compute/disks",
    "ip": "Microsoft.Network/publicIPAddresses",
    "nic": "Microsoft.Network/networkInterfaces",
    "vnet": "Microsoft.Network/virtualNetworks",
    "nsg": "Microsoft.Network/networkSecurityGroups",
}
API_VERSIONS = {"vm": "2024-03-01"}
RETRY_POLICY = {"retryCount": 3, "retryWindowInMinutes": 5}
ARM_TERMINAL = {"Succeeded", "Failed", "Canceled"}


def utc(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", value
    ):
        raise ValueError("deadline must use UTC ISO format ending in Z")
    value = dt.datetime.fromisoformat(value)
    if value.tzinfo is None or value.utcoffset() != dt.timedelta(0):
        raise ValueError("deadline must include UTC timezone")
    return value


def now():
    return dt.datetime.now(dt.UTC)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def call(argv, timeout=180):
    # Do not echo provider error bodies, which may include account/auth metadata.
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode:
        raise RuntimeError(f"{Path(argv[0]).name} failed, exit {result.returncode}")
    return json.loads(result.stdout) if result.stdout.strip() else None


def azure(executable, subscription):
    return lambda *args, timeout=180: call(
        [
            executable,
            *args,
            "--subscription",
            subscription,
            "--only-show-errors",
            "--output",
            "json",
        ],
        timeout=timeout,
    )


def identities(c):
    group = f"/subscriptions/{c['subscription']}/resourceGroups/{c['group']}"
    names = {key: f"cpu-{key}" for key in TYPES}
    ids = {key: f"{group}/providers/{kind}/{names[key]}" for key, kind in TYPES.items()}
    return group, names, ids


def tags(c):
    return {
        "project": "tennis3d",
        "run_id": c["run_id"],
        "source_commit": c["source_commit"],
        "expires_at": c["shutdown_at"],
    }


def tagged(resource, c):
    return all(resource.get("tags", {}).get(k) == v for k, v in tags(c).items())


def tag_args(c):
    return [f"{k}={v}" for k, v in tags(c).items()]


def unit(c):
    return "tennis-cpu-" + c["run_id"]


def require_timer(c):
    subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit(c) + ".timer"],
        check=True,
        timeout=30,
    )


def validate(c):
    uuid.UUID(c["subscription"])
    uuid.UUID(c["run_id"])
    if not re.fullmatch(r"tennis-(?:cpu|stage2)-[a-z0-9-]{1,50}", c["group"]):
        raise ValueError("use a dedicated tennis-cpu-* or tennis-stage2-* group")
    if c["region"] not in REGIONS:
        raise ValueError("region outside allowed policy")
    # These are the saved quota/catalog candidates, not a capacity guarantee.
    if c["region"] != "denmarkeast" or c["size"] not in {
        "Standard_F4s_v2",
        "Standard_F4",
        "Standard_F4s",
        "Standard_D3_v2",
        "Standard_DS3_v2",
    }:
        raise ValueError("size/region requires fresh admission evidence")
    if not re.fullmatch(r"[0-9a-f]{40}", c["source_commit"]):
        raise ValueError("pin source to full commit")
    if not re.fullmatch(r"Canonical:ubuntu-24_04-lts:server:[0-9.]+", c["image"]):
        raise ValueError("pin Ubuntu 24.04 x64 image version, never latest")
    control_network = ipaddress.ip_network(c["ssh_cidr"])
    if control_network.version != 4 or control_network.prefixlen != 32:
        raise ValueError("SSH requires control host IPv4 /32")
    if not c["ssh_public_key"].startswith("ssh-ed25519 "):
        raise ValueError("supply dedicated public SSH key")
    start, generate, drain, end = map(
        utc,
        [c[k] for k in ("created_at", "generate_until", "drain_until", "shutdown_at")],
    )
    if not start < generate < drain < end or (end - start).total_seconds() > 43200:
        raise ValueError("require ordered deadlines and <=12h resource session")
    if end.second or end.microsecond or (end - drain).total_seconds() < 300:
        raise ValueError(
            "Azure shutdown needs whole UTC minute and >=5min drain margin"
        )
    budget = c["budget"]
    quota = c["max_outbound_bytes"]
    if type(quota) is not int or quota <= 0:
        raise ValueError("positive outbound byte ceiling required before provisioning")
    # Every budget field is used below, so a missing one raises KeyError.
    if any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
        for v in budget.values()
    ):
        raise ValueError(
            "budget needs finite nonnegative measured quantities and quoted USD rates"
        )
    if quota / 1e9 > budget["egress_gb_with_retries"]:
        raise ValueError("outbound cap exceeds budgeted egress")
    hours = (
        (end - start).total_seconds() + RETRY_POLICY["retryWindowInMinutes"] * 60
    ) / 3600
    if (
        budget["compute_hours"] < hours
        or min(budget["disk_hours"], budget["ip_hours"]) < budget["compute_hours"] + 24
    ):
        raise ValueError(
            "forecast must include session and native retry window plus >=24h disk/IP recovery tail"
        )
    if (
        min(
            budget[k]
            for k in ("compute_usd_hour", "disk_usd_hour", "ip_usd_hour", "reserve_usd")
        )
        <= 0
    ):
        raise ValueError("missing actual rates or uncertainty reserve")
    total = (
        budget["compute_hours"] * budget["compute_usd_hour"]
        + budget["disk_hours"] * budget["disk_usd_hour"]
        + budget["ip_hours"] * budget["ip_usd_hour"]
        + budget["disk_transactions"] / 10000 * budget["disk_usd_10k_transactions"]
        + budget["egress_gb_with_retries"] * budget["egress_usd_gb"]
        + budget["other_usd"]
    )
    if (
        total + budget["reserve_usd"] + budget["pending_usd"]
        > budget["remaining_credit_usd"]
    ):
        raise ValueError(
            "whole-run forecast exceeds credit after reserve and pending charges"
        )
    for key in (
        "quote_evidence",
        "pilot_evidence",
        "whole_run_forecast_evidence",
        "admission_evidence",
    ):
        if not isinstance(c[key], str) or not c[key].strip():
            raise ValueError(f"missing {key}")
    return total


def template(c):
    """Explicit outbound through one Standard IPv4; no NAT or storage account."""
    _, names, ids = identities(c)
    resources = []

    def add(key, properties, dependencies=(), **extra):
        resources.append(
            {
                "type": TYPES[key],
                "apiVersion": API_VERSIONS.get(key, "2024-05-01"),
                "name": names[key],
                "location": c["region"],
                "tags": tags(c),
                "dependsOn": [ids[k] for k in dependencies],
                "properties": properties,
                **extra,
            }
        )

    add(
        "ip",
        {"publicIPAllocationMethod": "Static", "publicIPAddressVersion": "IPv4"},
        sku={"name": "Standard", "tier": "Regional"},
    )
    add(
        "nsg",
        {
            "securityRules": [
                {
                    "name": "control-ssh",
                    "properties": {
                        "priority": 100,
                        "access": "Allow",
                        "direction": "Inbound",
                        "protocol": "Tcp",
                        "sourceAddressPrefix": c["ssh_cidr"],
                        "sourcePortRange": "*",
                        "destinationAddressPrefix": "*",
                        "destinationPortRange": "22",
                    },
                }
            ]
        },
    )
    add(
        "vnet",
        {
            "addressSpace": {"addressPrefixes": ["10.42.0.0/16"]},
            "subnets": [
                {
                    "name": "cpu",
                    "properties": {
                        "addressPrefix": "10.42.0.0/24",
                        "defaultOutboundAccess": False,
                    },
                }
            ],
        },
    )
    add(
        "nic",
        {
            "networkSecurityGroup": {"id": ids["nsg"]},
            "ipConfigurations": [
                {
                    "name": "cpu",
                    "properties": {
                        "privateIPAllocationMethod": "Dynamic",
                        "subnet": {"id": ids["vnet"] + "/subnets/cpu"},
                        "publicIPAddress": {"id": ids["ip"]},
                    },
                }
            ],
        },
        ("ip", "nsg", "vnet"),
    )
    publisher, offer, sku, version = c["image"].split(":")
    add(
        "vm",
        {
            "hardwareProfile": {"vmSize": c["size"]},
            "priority": "Regular",
            "storageProfile": {
                "imageReference": {
                    "publisher": publisher,
                    "offer": offer,
                    "sku": sku,
                    "version": version,
                },
                "osDisk": {
                    "name": names["os"],
                    "createOption": "FromImage",
                    "diskSizeGB": 64,
                    "deleteOption": "Detach",
                    "managedDisk": {"storageAccountType": "StandardSSD_LRS"},
                },
            },
            "osProfile": {
                "computerName": "tennis-cpu",
                "adminUsername": "runner",
                "linuxConfiguration": {
                    "disablePasswordAuthentication": True,
                    "ssh": {
                        "publicKeys": [
                            {
                                "path": "/home/runner/.ssh/authorized_keys",
                                "keyData": c["ssh_public_key"],
                            }
                        ]
                    },
                },
            },
            "networkProfile": {
                "networkInterfaces": [
                    {
                        "id": ids["nic"],
                        "properties": {"primary": True, "deleteOption": "Detach"},
                    }
                ]
            },
            "diagnosticsProfile": {"bootDiagnostics": {"enabled": False}},
        },
        ("nic",),
    )
    return {
        "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
        "contentVersion": "1.0.0.0",
        "resources": resources,
    }


def owned(az, c, vm_only=False, untagged_disk=False):
    group, _, ids = identities(c)
    actual = az("group", "show", "--name", c["group"])
    if actual["id"].lower() != group.lower() or not tagged(actual, c):
        raise ValueError("resource group ownership mismatch")
    if vm_only:
        # Stop must work after an incomplete deployment, so check only the VM.
        vm = az("vm", "show", "--ids", ids["vm"])
        if vm["id"].lower() != ids["vm"].lower() or not tagged(vm, c):
            raise ValueError("VM ownership mismatch")
        return [vm]
    inventory = az("resource", "list", "--resource-group", c["group"])
    expected = {v.lower() for v in ids.values()}
    for resource in inventory:
        if resource["id"].lower() not in expected:
            raise ValueError("unexpected resource in group; refusing mutation")
        if (
            untagged_disk
            and resource["id"].lower() == ids["os"].lower()
            and not resource.get("tags")
        ):
            continue
        if not tagged(resource, c):
            raise ValueError("resource ownership tags mismatch")
    return inventory


def power(az, c):
    vm = az("vm", "get-instance-view", "--ids", identities(c)[2]["vm"])
    return [
        s["code"]
        for s in vm["instanceView"]["statuses"]
        if s["code"].startswith("PowerState/")
    ]


def deallocate_owned(az, c):
    # Include ownership reads and CLI latency in a wall-clock bound below the
    # controller service's RuntimeMaxSec=300, rather than just counting sleeps.
    until = time.monotonic() + 240
    failure = "deallocation not confirmed; retry stop and retain owned disk/IP budget"

    def bounded(*args):
        remaining = until - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(failure)
        return az(*args, timeout=min(30, remaining))

    owned(bounded, c, vm_only=True)
    bounded("vm", "deallocate", "--ids", identities(c)[2]["vm"], "--no-wait")
    for _ in range(48):
        if power(bounded, c) == ["PowerState/deallocated"]:
            return {"deallocation_requested": True, "confirmed": True}
        remaining = until - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(5, remaining))
    raise RuntimeError(failure)


def tag_attached_disk(az, c):
    vm = owned(az, c, vm_only=True)[0]
    owned(az, c, untagged_disk=True)
    ids = identities(c)[2]
    disk = az("disk", "show", "--ids", ids["os"])
    attached = vm["storageProfile"]["osDisk"]["managedDisk"]["id"]
    if (
        disk["id"].lower() != ids["os"].lower()
        or disk.get("managedBy", "").lower() != ids["vm"].lower()
        or attached.lower() != ids["os"].lower()
        or (disk.get("tags") and not tagged(disk, c))
    ):
        raise ValueError("implicit disk ownership/attachment mismatch")
    az("tag", "create", "--resource-id", ids["os"], "--tags", *tag_args(c))


def operation_endpoint(c, action):
    return (
        "https://management.azure.com/subscriptions/"
        + c["subscription"]
        + "/providers/Microsoft.ComputeSchedule/locations/"
        + c["region"]
        + "/virtualMachines"
        + action
        + "?api-version=2025-05-01"
    )


def operation_request(az, c, action, body):
    return az(
        "rest",
        "--method",
        "post",
        "--url",
        operation_endpoint(c, action),
        "--body",
        json.dumps(body),
    )


def operation_result(response, c, operation_id=None):
    results = response.get("results", [])
    if len(results) != 1:
        raise ValueError("expected exactly one scheduled-operation result")
    item = results[0]
    op = item.get("operation", {})
    vm_id = identities(c)[2]["vm"].lower()
    if (
        item.get("resourceId", "").lower() != vm_id
        or op.get("resourceId", "").lower() != vm_id
        or op.get("subscriptionId", "").lower() != c["subscription"].lower()
        or response.get("location", c["region"]).lower() != c["region"]
        or op.get("retryPolicy") != RETRY_POLICY
        or op.get("opType") != "Deallocate"
        or op.get("deadlineType") != "InitiateAt"
        or dt.datetime.fromisoformat(op.get("deadline", "")) != utc(c["shutdown_at"])
        or not op.get("operationId")
        or (operation_id is not None and op["operationId"] != operation_id)
    ):
        raise ValueError("scheduled-operation identity/deadline mismatch")
    return item, op


def operation_status(az, c, receipt):
    saved = receipt["operation"]
    response = operation_request(
        az,
        c,
        "GetOperationStatus",
        {"operationIds": [saved["id"]], "correlationid": saved["correlation_id"]},
    )
    return operation_result(response, c, saved["id"])


def arm_operation(az, c, receipt_path, receipt):
    correlation = str(uuid.uuid4())
    # Save correlation/endpoint before the request: a lost response must not be
    # treated as a confirmed refusal or retried by submitting a second operation.
    receipt["operation_submission"] = {
        "correlation_id": correlation,
        "endpoint": operation_endpoint(c, "SubmitDeallocate"),
        "target": identities(c)[2]["vm"],
        "deadline": c["shutdown_at"],
        "retry_policy": RETRY_POLICY,
    }
    save(receipt_path, receipt)
    response = operation_request(
        az,
        c,
        "SubmitDeallocate",
        {
            "schedule": {
                "deadline": c["shutdown_at"],
                "timezone": "UTC",
                "deadlineType": "InitiateAt",
            },
            "executionParameters": {"retryPolicy": RETRY_POLICY},
            "resources": {"ids": [identities(c)[2]["vm"]]},
            "correlationid": correlation,
        },
    )
    # Retain response before validation, so even an unexpected result is recoverable.
    receipt["operation_response"] = response
    save(receipt_path, receipt)
    item, op = operation_result(response, c)
    receipt["operation"] = {
        "id": op["operationId"],
        "correlation_id": correlation,
        "status_endpoint": operation_endpoint(c, "GetOperationStatus"),
    }
    save(receipt_path, receipt)
    if item.get("errorCode") or op.get("resourceOperationError"):
        raise ValueError("Azure submit returned an error; no deployment allowed")
    receipt["operation"]["accepted_status"] = require_pending_operation(az, c, receipt)
    save(receipt_path, receipt)


def require_pending_operation(az, c, receipt):
    # The no-spend absent-target probe stayed PendingScheduling until it failed
    # NotFound near InitiateAt. Do not require an unobserved intermediate state.
    item, op = operation_status(az, c, receipt)
    if (
        op.get("state") not in {"PendingScheduling", "Scheduled"}
        or item.get("errorCode")
        or op.get("resourceOperationError")
    ):
        raise ValueError("independent Azure operation is not pending without errors")
    if now() >= utc(c["generate_until"]):
        raise ValueError("generation deadline passed")
    return op


def release_deadlines(az, c, receipt_path, receipt):
    # Disarm only after the recorded native operation is confirmed terminal, so an
    # unconfirmed cancellation leaves both independent deadlines armed.
    if "operation" in receipt:
        cancel_operation(az, c, receipt_path, receipt)
    elif "operation_submission" in receipt:
        raise RuntimeError(
            "native submission unconfirmed; retain receipt and timer for reconciliation"
        )
    disarm_timer(c)


def cancel_operation(az, c, receipt_path, receipt):
    _, op = operation_status(az, c, receipt)
    if op.get("state") not in {"Cancelled", "Succeeded", "Failed"}:
        saved = receipt["operation"]
        operation_request(
            az,
            c,
            "CancelOperations",
            {"operationIds": [saved["id"]], "correlationid": saved["correlation_id"]},
        )
        for attempt in range(12):
            _, op = operation_status(az, c, receipt)
            if op.get("state") in {"Cancelled", "Succeeded", "Failed"}:
                break
            if attempt < 11:
                time.sleep(5)
        else:
            raise RuntimeError(
                "scheduled operation cancellation not confirmed; retain group/receipt"
            )
    receipt["operation"]["terminal_state"] = op["state"]
    save(receipt_path, receipt)


def start(az, c, manifest, receipt):
    validate(c)
    m = read(manifest)
    for key in ("run_id", "source_commit", "generate_until", "drain_until"):
        if m[key] != c[key]:
            raise ValueError("guest manifest differs from reviewed session")
    if not now() < utc(c["generate_until"]):
        raise ValueError("generation deadline passed")
    require_pending_operation(az, c, receipt)
    require_timer(c)
    # The persistent service owns the process cgroup even after SSH/controller loss.
    # Exact guest commands are supplied by the merged producer.
    encoded = base64.b64encode(Path(manifest).read_bytes()).decode()
    guest = str(Path(m["cwd"]) / "scripts/azure_guest.py")
    if not Path(m["cwd"]).is_absolute():
        raise ValueError("guest checkout must be absolute")
    digest = m["files"]["scripts/azure_guest.py"]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("pin guest runner hash")
    seconds = int((utc(c["drain_until"]) - now()).total_seconds())
    quota = c["max_outbound_bytes"]
    q = shlex.quote
    directory = "/var/lib/tennis-cpu/" + c["run_id"]
    script = f"""set -eu
# Refuse repeated starts: neither deadlines nor byte counters may be reset.
test ! -e {q(directory)}
test "$(sha256sum {q(guest)} | cut -d ' ' -f 1)" = {q(digest)}
command -v iptables >/dev/null
command -v ip6tables >/dev/null
mkdir -p {q(directory)}
echo {q(encoded)} | base64 -d > {q(directory + "/manifest.json")}
chown -R runner:runner {q(directory)}
# IPv4 quota counts every remote outbound packet, including transfer retries.
# IPv6 is blocked so it cannot bypass this ceiling. Reboot resets these rules;
# the service is transient and must never auto-restart after reboot.
iptables -N TENNIS_EGRESS
iptables -A TENNIS_EGRESS -o lo -j RETURN
iptables -A TENNIS_EGRESS -m quota --quota {quota} -j RETURN
iptables -A TENNIS_EGRESS -j REJECT
iptables -I OUTPUT 1 -j TENNIS_EGRESS
ip6tables -I OUTPUT 1 ! -o lo -j REJECT
systemd-run --unit={q(unit(c))} --property=User=runner --property=WorkingDirectory={q(m["cwd"])} --property=MemoryMax=6G --property=MemorySwapMax=0 --property=CPUQuota=400% --property=TasksMax=128 --property=RuntimeMaxSec={seconds} --property=TimeoutStopSec=10 --property=KillMode=control-group /usr/bin/python3 {q(guest)} --manifest {q(directory + "/manifest.json")} --result {q(directory + "/result.json")}
systemctl is-active --quiet {q(unit(c))}
echo TENNIS_STARTED_{c["run_id"]}
"""
    result = az(
        "vm",
        "run-command",
        "invoke",
        "--ids",
        identities(c)[2]["vm"],
        "--command-id",
        "RunShellScript",
        "--scripts",
        script,
        timeout=max(1, int((utc(c["generate_until"]) - now()).total_seconds())),
    )
    # RunCommand returns ARM success even when the shell failed. Inspect guest code.
    statuses = result.get("value", [])
    if not statuses or not any(
        "TENNIS_STARTED_" + c["run_id"] in s.get("message", "") for s in statuses
    ):
        raise RuntimeError("guest launch not confirmed; inspect RunCommand on owned VM")
    return {
        "start_requested": True,
        "unit": unit(c),
        "verify": "inspect systemctl status and guest result; do not infer completion from ARM success",
    }


def disarm_timer(c):
    for suffix in (".timer", ".service"):
        name = unit(c) + suffix
        stopped = subprocess.run(
            ["systemctl", "--user", "stop", name],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        shown = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                name,
                "--property=LoadState,ActiveState,SubState",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        state = dict(line.split("=", 1) for line in shown.stdout.splitlines())
        # A never-run transient service disappears with its timer, so stopping it
        # exits 5. That is harmless only with a successful not-found readback.
        accepted = {"loaded": {0}, "not-found": {0, 5}}.get(state.get("LoadState"), ())
        if (
            stopped.returncode not in accepted
            or state.get("ActiveState") != "inactive"
            or state.get("SubState") != "dead"
        ):
            raise ValueError(
                f"{name} not confirmed inactive, stop exit {stopped.returncode}"
            )


def require_linger():
    result = subprocess.run(
        ["loginctl", "show-user", str(os.getuid()), "-p", "Linger", "--value"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    if result.stdout.strip() != "yes":
        raise ValueError("control-host user manager must already have Linger=yes")


class DeploymentUnsettled(RuntimeError):
    """ARM may still be creating resources; retain independent stop controls."""


def deployment_state(az, c):
    status = az(
        "deployment", "group", "show", "--resource-group", c["group"], "--name", "cpu"
    )
    return status["properties"]["provisioningState"]


def cancel_deployment(az, c):
    try:
        owned(az, c, untagged_disk=True)
        az(
            "deployment",
            "group",
            "cancel",
            "--resource-group",
            c["group"],
            "--name",
            "cpu",
        )
        for _ in range(12):
            if deployment_state(az, c) in ARM_TERMINAL:
                return
            time.sleep(5)
        raise RuntimeError("ARM deployment still running")
    except Exception as exc:
        raise DeploymentUnsettled(
            "ARM cancellation unconfirmed; retain independent deadlines and reconcile deployment"
        ) from exc


def deploy(az, c, arm):
    terminal = False
    try:
        az(
            "deployment",
            "group",
            "create",
            "--resource-group",
            c["group"],
            "--name",
            "cpu",
            "--template-file",
            str(arm),
            "--no-wait",
        )
        while now() < utc(c["generate_until"]):
            state = deployment_state(az, c)
            terminal = state in ARM_TERMINAL
            if state == "Succeeded":
                return
            if terminal:
                raise RuntimeError(
                    "Azure deployment failed; independent deadline remains armed"
                )
            time.sleep(5)
        raise RuntimeError("provisioning deadline exceeded")
    except Exception:
        # A CLI timeout does not stop ARM. Confirm terminal cancellation before
        # recovery can deallocate; otherwise leave both independent controls armed.
        if not terminal:
            cancel_deployment(az, c)
        raise


def lifecycle(action, path, az_path="az", recovery=None, manifest=None):
    path = Path(path).resolve()
    c = read(path)
    az = azure(az_path, c["subscription"])
    group_id, _, ids = identities(c)
    receipt_path = path.with_suffix(".receipt.json")
    if action == "prepare":
        total = validate(c)
        save(path.with_suffix(".arm.json"), template(c))
        return {
            "forecast_usd": total,
            "resource_ids": ids,
            "spending_authorized": False,
        }
    if action == "provision":
        validate(c)
        if not now() < utc(c["generate_until"]) or utc(c["created_at"]) > now():
            raise ValueError("session is not current")
        if az("group", "exists", "--name", c["group"]):
            raise ValueError("group already exists; never adopt or overwrite")
        if receipt_path.exists():
            raise ValueError(
                "ownership receipt already exists; reconcile its operation before retry"
            )
        require_linger()
        # Arm the persistent timer before the native operation and resource creation.
        # No workload is installed or started by ARM.
        subprocess.run(
            [
                "systemd-run",
                "--user",
                "--unit",
                unit(c),
                "--on-calendar",
                utc(c["shutdown_at"]).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "--timer-property=Persistent=true",
                "--property=Restart=on-failure",
                "--property=RestartSec=30",
                "--property=StartLimitIntervalSec=86400",
                "--property=StartLimitBurst=12",
                "--property=RuntimeMaxSec=300",
                *(
                    ["--setenv=AZURE_CONFIG_DIR=" + os.environ["AZURE_CONFIG_DIR"]]
                    if "AZURE_CONFIG_DIR" in os.environ
                    else []
                ),
                sys.executable,
                str(Path(__file__).resolve()),
                "stop",
                str(path),
                "--az",
                str(Path(shutil.which(az_path) or az_path).resolve()),
            ],
            check=True,
            timeout=30,
        )
        require_timer(c)
        # Persist ownership and the cloud operation before any billable create.
        receipt = {
            "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "group_id": group_id,
            "resource_ids": ids,
        }
        save(receipt_path, receipt)
        try:
            arm_operation(az, c, receipt_path, receipt)
            if az("group", "exists", "--name", c["group"]):
                raise ValueError(
                    "group appeared while arming deadline; refusing adoption"
                )
            az(
                "group",
                "create",
                "--name",
                c["group"],
                "--location",
                c["region"],
                "--tags",
                *tag_args(c),
            )
            owned(az, c)
            arm = path.with_suffix(".arm.json")
            save(arm, template(c))
            # Recheck after group creation as scheduling/provisioning can take time.
            require_pending_operation(az, c, receipt)
        except Exception:
            if "operation" in receipt or "operation_submission" not in receipt:
                release_deadlines(az, c, receipt_path, receipt)
            # Unknown submission has no trusted operation ID to cancel. Keep its
            # receipt and local timer; never proceed to any resource creation.
            raise
        try:
            receipt["deployment_submitted"] = True
            save(receipt_path, receipt)
            deploy(az, c, arm)
            tag_attached_disk(az, c)
            owned(az, c)
        except DeploymentUnsettled:
            # Deallocation could race still-running ARM work. Keep Azure and local
            # deadlines, report uncertainty, and require deployment reconciliation.
            raise
        except Exception:
            # The native operation is already armed if the controller disappears.
            # Recheck group+VM identity before even attempting failure recovery.
            deallocate_owned(az, c)
            raise
        return {"provisioned": True, "generation_started": False}
    receipt = read(receipt_path)
    if receipt["config_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("config changed since ownership receipt")
    if action == "cleanup" and az("group", "exists", "--name", c["group"]) is False:
        release_deadlines(az, c, receipt_path, receipt)
        return {"group_absent": True, "confirmed": True}
    if action == "stop":
        return deallocate_owned(az, c)
    inventory = owned(az, c)
    if action == "cleanup" and receipt.get("deployment_submitted"):
        deployments = az("deployment", "group", "list", "--resource-group", c["group"])
        expected = group_id.lower() + "/providers/microsoft.resources/deployments/cpu"
        if (
            not isinstance(deployments, list)
            or len(deployments) > 1
            or any(
                not isinstance(item, dict) or item.get("id", "").lower() != expected
                for item in deployments
            )
        ):
            raise ValueError(
                "unexpected or unreadable deployment inventory; retain controls"
            )
        if not deployments:
            if owned(az, c):
                raise ValueError(
                    "deployment absent but resources remain; retain controls"
                )
            # Creation may have been rejected before recording a deployment.
            # Retain both controls through deletion; only confirmed group absence
            # on a later cleanup can resolve any late/ambiguous create request.
            az("group", "delete", "--name", c["group"], "--yes", "--no-wait")
            return {
                "empty_owned_group_deletion_requested": True,
                "shutdown_controls_retained": True,
                "confirmed": False,
                "verify": "repeat cleanup after group absence is confirmed",
            }
        if deployment_state(az, c) not in ARM_TERMINAL:
            raise ValueError("ARM deployment is not terminal; retain shutdown controls")
        # ARM may have created a VM/disk between the first inventory read and
        # terminal confirmation. All cleanup decisions need the current inventory.
        inventory = owned(az, c)
    if action == "start":
        return start(az, c, manifest, receipt)
    present = {resource["id"].lower() for resource in inventory}
    has_vm = ids["vm"].lower() in present
    has_disk = ids["os"].lower() in present
    if action == "status":
        return {
            "resources": [r["id"] for r in inventory],
            "power": power(az, c) if has_vm else [],
            "shutdown_at": c["shutdown_at"],
            "independent_operation": operation_status(az, c, receipt)[1],
        }
    if action == "cleanup" and not has_vm and not has_disk:
        release_deadlines(az, c, receipt_path, receipt)
        az("group", "delete", "--name", c["group"], "--yes", "--no-wait")
        return {"owned_group_without_data_deletion_requested": True, "confirmed": False}
    if action == "cleanup":
        if recovery is None:
            raise ValueError("cleanup needs local recovery evidence")
        if has_vm and power(az, c) != ["PowerState/deallocated"]:
            raise ValueError("confirm deallocation before cleanup")
        evidence = read(recovery)
        if evidence["run_id"] != c["run_id"] or evidence.get("complete") is not True:
            raise ValueError("recovery receipt is incomplete or belongs to another run")
        for key in ("state", "publication_receipts", "logs"):
            item = evidence[key]
            file = Path(recovery).resolve().parent / item["path"]
            with file.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise ValueError("recovery copy hash mismatch")
        release_deadlines(az, c, receipt_path, receipt)
        az("group", "delete", "--name", c["group"], "--yes", "--no-wait")
        return {
            "deletion_requested": True,
            "confirmed": False,
            "verify": "az group exists must return false; disk/IP bill until deletion",
        }
    raise ValueError("unknown action")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["prepare", "provision", "start", "status", "stop", "cleanup"]
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--az", default="az")
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.action == "start" and args.manifest is None:
        parser.error("start requires --manifest")
    try:
        print(
            json.dumps(
                lifecycle(
                    args.action, args.config, args.az, args.recovery, args.manifest
                ),
                indent=2,
            )
        )
    except (
        ValueError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        parser.exit(2, f"Refused/failed: {exc}\n")


if __name__ == "__main__":
    main()
