"""No-spend lifecycle tests. The executable fake records all Azure requests."""

import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "azure_cpu", Path(__file__).parents[1] / "scripts/azure_cpu.py"
)
assert SPEC and SPEC.loader
cpu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cpu)


def config():
    start = dt.datetime.now(dt.UTC).replace(second=0, microsecond=0)
    stamp = lambda seconds: (
        (start + dt.timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")
    )
    return {
        "subscription": "00000000-0000-4000-8000-000000000000",
        "run_id": "00000000-0000-4000-8000-000000000001",
        "group": "tennis-cpu-test",
        "region": "denmarkeast",
        "size": "Standard_F4s_v2",
        "source_commit": "a" * 40,
        "image": "Canonical:ubuntu-24_04-lts:server:24.04.202601010",
        "ssh_cidr": "192.0.2.1/32",
        "ssh_public_key": "ssh-ed25519 public-test-key",
        "created_at": stamp(-60),
        "generate_until": stamp(1200),
        "drain_until": stamp(1500),
        "shutdown_at": stamp(1800),
        "max_outbound_bytes": 1_000_000_000,
        "quote_evidence": "TEST ONLY",
        "pilot_evidence": "TEST ONLY",
        "whole_run_forecast_evidence": "TEST ONLY",
        "admission_evidence": "TEST ONLY",
        "budget": {
            "compute_hours": 1,
            "disk_hours": 25,
            "ip_hours": 25,
            "disk_transactions": 10000,
            "egress_gb_with_retries": 1,
            "compute_usd_hour": 1,
            "disk_usd_hour": 0.1,
            "ip_usd_hour": 0.1,
            "disk_usd_10k_transactions": 0.01,
            "egress_usd_gb": 0.1,
            "other_usd": 1,
            "reserve_usd": 2,
            "remaining_credit_usd": 100,
            "pending_usd": 0,
        },
    }


OPERATION_ID = "00000000-0000-4000-8000-000000000002"
CORRELATION_ID = "00000000-0000-4000-8000-000000000003"


def operation(c, state="Scheduled", operation_id=OPERATION_ID):
    vm = cpu.identities(c)[2]["vm"]
    return {
        "results": [
            {
                "resourceId": vm,
                "operation": {
                    "resourceId": vm,
                    "operationId": operation_id,
                    "opType": "Deallocate",
                    "subscriptionId": c["subscription"],
                    "retryPolicy": {"retryCount": 3, "retryWindowInMinutes": 5},
                    "deadlineType": "InitiateAt",
                    "deadline": c["shutdown_at"],
                    "state": state,
                },
            }
        ]
    }


def operation_action(args):
    if args[:2] in (["rest", "--method"], ("rest", "--method")):
        return (
            args[args.index("--url") + 1]
            .split("virtualMachines", 1)[1]
            .split("?", 1)[0]
        )
    return None


def request_body(args):
    return json.loads(args[args.index("--body") + 1])


@pytest.fixture
def setup(tmp_path):
    c = config()
    group, _, ids = cpu.identities(c)
    responses = {
        "group show": {"id": group, "tags": cpu.tags(c)},
        "group exists": True,
        "resource list": [{"id": v, "tags": cpu.tags(c)} for v in ids.values()],
        "vm show": {
            "id": ids["vm"],
            "tags": cpu.tags(c),
            "storageProfile": {"osDisk": {"managedDisk": {"id": ids["os"]}}},
        },
        "disk show": {"id": ids["os"], "tags": {}, "managedBy": ids["vm"]},
        "vm get-instance-view": {
            "instanceView": {"statuses": [{"code": "PowerState/deallocated"}]}
        },
        "vm deallocate": None,
        "group delete": None,
        "SubmitDeallocate": operation(c, "PendingScheduling"),
        "GetOperationStatus": operation(c),
        "CancelOperations": {},
        "deployment group show": {"properties": {"provisioningState": "Succeeded"}},
        "deployment group list": [
            {"id": group + "/providers/Microsoft.Resources/deployments/cpu"}
        ],
    }
    config_path = tmp_path / "session.json"
    cpu.save(config_path, c)
    cpu.save(
        config_path.with_suffix(".receipt.json"),
        {
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "group_id": group,
            "resource_ids": ids,
            "deployment_submitted": True,
            "operation": {"id": OPERATION_ID, "correlation_id": CORRELATION_ID},
        },
    )
    fake = tmp_path / "az"
    fake.write_text(f"""#!{sys.executable}
import json, pathlib, sys
base = pathlib.Path(__file__).parent
args = sys.argv[1:]
with (base/'calls.jsonl').open('a') as stream:
    stream.write(json.dumps(args)+'\\n')
responses = json.loads((base/'responses.json').read_text())
key = ' '.join(args[:2])
if key == 'deployment group':
    key = ' '.join(args[:3])
if key == 'rest --method':
    key = args[args.index('--url')+1].split('virtualMachines', 1)[1].split('?', 1)[0]
if key not in responses: sys.exit(7)
if key == 'CancelOperations' and 'sequence' not in responses['GetOperationStatus']:
    responses['GetOperationStatus']['results'][0]['operation']['state'] = 'Cancelled'
    (base/'responses.json').write_text(json.dumps(responses))
response = responses[key]
if isinstance(response, dict) and 'sequence' in response:
    response = response['sequence'].pop(0) if len(response['sequence']) > 1 else response['sequence'][0]
    (base/'responses.json').write_text(json.dumps(responses))
print(json.dumps(response))
""")
    fake.chmod(0o755)
    cpu.save(tmp_path / "responses.json", responses)
    return c, config_path, str(fake), responses


@pytest.fixture
def provision_setup(setup):
    _, path, _, _ = setup
    path.with_suffix(".receipt.json").unlink()
    return setup


def requests(path):
    return [
        json.loads(line)
        for line in (path.parent / "calls.jsonl").read_text().splitlines()
    ]


def test_prepare_no_azure_and_conservative_budget(setup):
    _, path, _, _ = setup
    result = cpu.lifecycle("prepare", path, "/does/not/exist")
    assert result["forecast_usd"] == pytest.approx(7.11)
    t = cpu.read(path.with_suffix(".arm.json"))
    assert len(t["resources"]) == 5
    vm = next(r for r in t["resources"] if r["type"] == cpu.TYPES["vm"])
    assert vm["properties"]["priority"] == "Regular"
    assert (
        vm["properties"]["storageProfile"]["osDisk"]["managedDisk"][
            "storageAccountType"
        ]
        == "StandardSSD_LRS"
    )
    assert not (path.parent / "calls.jsonl").exists()


@pytest.mark.parametrize(
    "edit",
    [
        {"region": "eastus"},
        {"size": "Standard_F8s_v2"},
        {"image": "Ubuntu2404"},
        {"ssh_cidr": "0.0.0.0/0"},
        {"ssh_cidr": "2001:db8::/32"},
        {"source_commit": "main"},
        {"quote_evidence": ""},
    ],
)
def test_refuse_unreviewed_configuration(edit):
    c = config()
    c.update(edit)
    with pytest.raises(ValueError):
        cpu.validate(c)


def test_refuse_forecast_without_credit_or_recovery_tail():
    c = config()
    c["budget"]["remaining_credit_usd"] = 3
    with pytest.raises(ValueError, match="credit"):
        cpu.validate(c)
    c = config()
    c["budget"]["disk_hours"] = 1
    with pytest.raises(ValueError, match="tail"):
        cpu.validate(c)


def test_deadline_stop_uses_receipt_and_owns_vm_even_if_deploy_incomplete(setup):
    c, path, fake, responses = setup
    # Deployment or disk tagging can be incomplete when the deadline fires.
    responses["resource list"] = []
    cpu.save(path.parent / "responses.json", responses)
    result = cpu.lifecycle("stop", path, fake)
    assert result == {"deallocation_requested": True, "confirmed": True}
    calls = requests(path)
    assert calls[-2][:2] == ["vm", "deallocate"]
    assert cpu.identities(c)[2]["vm"] in calls[-2]
    assert all(
        call[call.index("--subscription") + 1] == c["subscription"] for call in calls
    )


@pytest.mark.parametrize("target", ["group show", "vm show"])
def test_stop_refuses_other_owner(setup, target):
    _, path, fake, responses = setup
    responses[target]["tags"]["run_id"] = "another"
    cpu.save(path.parent / "responses.json", responses)
    with pytest.raises(ValueError, match="ownership"):
        cpu.lifecycle("stop", path, fake)
    assert not any(call[:2] == ["vm", "deallocate"] for call in requests(path))


def recovery(path, c):
    evidence = {"run_id": c["run_id"], "complete": True}
    for name in ("state", "publication_receipts", "logs"):
        file = path.parent / name
        file.write_text("preserved")
        evidence[name] = {
            "path": name,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        }
    location = path.parent / "recovery.json"
    cpu.save(location, evidence)
    return location


def test_cleanup_preserves_local_recovery_and_removes_whole_owned_group(
    setup, monkeypatch
):
    c, path, fake, responses = setup
    responses["GetOperationStatus"] = {
        "sequence": [operation(c), operation(c, "Cancelled")]
    }
    cpu.save(path.parent / "responses.json", responses)
    saved = recovery(path, c)
    timer_stub(monkeypatch, path)
    assert cpu.lifecycle("cleanup", path, fake, saved)["deletion_requested"]
    assert (path.parent / "state").read_text() == "preserved"
    assert requests(path)[-1][:2] == ["group", "delete"]


@pytest.mark.parametrize(
    "fault", ["unknown_resource", "running", "tampered_recovery", "changed_config"]
)
def test_cleanup_refusal_paths(setup, fault):
    c, path, fake, responses = setup
    saved = recovery(path, c)
    if fault == "unknown_resource":
        responses["resource list"].append({"id": "foreign", "tags": cpu.tags(c)})
    elif fault == "running":
        responses["vm get-instance-view"]["instanceView"]["statuses"][0]["code"] = (
            "PowerState/running"
        )
    elif fault == "tampered_recovery":
        (path.parent / "state").write_text("changed")
    else:
        c["group"] = "tennis-cpu-someone-else"
        cpu.save(path, c)
    cpu.save(path.parent / "responses.json", responses)
    with pytest.raises(ValueError):
        cpu.lifecycle("cleanup", path, fake, saved)
    if (path.parent / "calls.jsonl").exists():
        assert not any(call[:2] == ["group", "delete"] for call in requests(path))


def test_start_missing_azure_deadline_never_launches(setup, monkeypatch):
    c, path, fake, responses = setup
    responses["GetOperationStatus"] = operation(c, "Failed")
    cpu.save(path.parent / "responses.json", responses)
    manifest = path.parent / "manifest.json"
    cpu.save(
        manifest,
        {
            key: c[key]
            for key in ("run_id", "source_commit", "generate_until", "drain_until")
        },
    )
    with pytest.raises(ValueError, match="not pending"):
        cpu.lifecycle("start", path, fake, manifest=manifest)
    assert not any(call[:2] == ["vm", "run-command"] for call in requests(path))


def test_no_create_if_existing_group(provision_setup):
    _, path, fake, responses = provision_setup
    responses["group exists"] = True
    cpu.save(path.parent / "responses.json", responses)
    with pytest.raises(ValueError, match="already exists"):
        cpu.lifecycle("provision", path, fake)
    assert requests(path)[-1][:2] == ["group", "exists"]


def timer_stub(monkeypatch, path, fail=False):
    original = cpu.subprocess.run

    def run(argv, **kwargs):
        if argv[0] in {"systemd-run", "systemctl", "loginctl"}:
            with (path.parent / "calls.jsonl").open("a") as stream:
                stream.write(json.dumps(argv) + "\n")
            if fail:
                raise cpu.subprocess.CalledProcessError(1, argv)
            stdout = (
                "LoadState=not-found\nActiveState=inactive\nSubState=dead\n"
                if argv[:3] == ["systemctl", "--user", "show"]
                else "yes\n"
            )
            return cpu.subprocess.CompletedProcess(argv, 0, stdout=stdout)
        return original(argv, **kwargs)

    monkeypatch.setattr(cpu.subprocess, "run", run)


@pytest.mark.parametrize("native_state", ["PendingScheduling", "Scheduled"])
def test_provision_arms_both_deadlines_before_any_create(
    provision_setup, monkeypatch, native_state
):
    c, path, fake, responses = provision_setup
    responses.update(
        {
            "group exists": False,
            "group create": responses["group show"],
            "SubmitDeallocate": operation(c, native_state),
            "GetOperationStatus": operation(c, native_state),
            "deployment group create": {},
            "deployment group show": {"properties": {"provisioningState": "Succeeded"}},
            "tag create": {},
        }
    )
    cpu.save(path.parent / "responses.json", responses)
    timer_stub(monkeypatch, path)
    assert cpu.lifecycle("provision", path, fake)["generation_started"] is False
    calls = requests(path)
    assert "Microsoft.DevTestLab" not in json.dumps(calls)
    arm = next(i for i, call in enumerate(calls) if call[0] == "systemd-run")
    create = next(i for i, call in enumerate(calls) if call[:2] == ["group", "create"])
    assert arm < create
    assert "--property=Restart=on-failure" in calls[arm]
    assert "--property=StartLimitBurst=12" in calls[arm]
    assert "--property=RuntimeMaxSec=300" in calls[arm]
    assert "stop" in calls[arm]
    submit = next(
        i
        for i, call in enumerate(calls)
        if operation_action(call) == "SubmitDeallocate"
    )
    ready = next(
        i
        for i, call in enumerate(calls)
        if operation_action(call) == "GetOperationStatus"
    )
    deploy = next(
        i for i, call in enumerate(calls) if call[:2] == ["deployment", "group"]
    )
    assert arm < submit < ready < create < deploy
    assert "--no-wait" in calls[deploy]
    assert any(
        call[:3] == ["deployment", "group", "show"] for call in calls[deploy + 1 :]
    )
    assert (
        sum(operation_action(call) == "GetOperationStatus" for call in calls[:deploy])
        >= 2
    )
    submitted = request_body(calls[submit])
    assert submitted["resources"]["ids"] == [cpu.identities(c)[2]["vm"]]
    assert submitted["schedule"] == {
        "deadline": c["shutdown_at"],
        "timezone": "UTC",
        "deadlineType": "InitiateAt",
    }
    url = calls[submit][calls[submit].index("--url") + 1]
    assert (
        f"/subscriptions/{c['subscription']}/providers/Microsoft.ComputeSchedule/locations/{c['region']}/"
        in url
    )
    assert "/resourceGroups/" not in url
    receipt = cpu.read(path.with_suffix(".receipt.json"))
    assert receipt["operation"]["id"] == OPERATION_ID
    assert receipt["operation"]["correlation_id"] == submitted["correlationid"]
    assert receipt["operation_response"] == responses["SubmitDeallocate"]


def test_no_provision_if_controller_deadline_cannot_be_armed(
    provision_setup, monkeypatch
):
    _, path, fake, responses = provision_setup
    responses["group exists"] = False
    cpu.save(path.parent / "responses.json", responses)
    timer_stub(monkeypatch, path, fail=True)
    with pytest.raises(cpu.subprocess.CalledProcessError):
        cpu.lifecycle("provision", path, fake)
    assert not any(call[:2] == ["group", "create"] for call in requests(path))


def test_start_bounds_service_and_outbound_then_checks_guest_success(
    setup, monkeypatch
):
    c, path, fake, responses = setup
    m = {
        key: c[key]
        for key in ("run_id", "source_commit", "generate_until", "drain_until")
    }
    m.update(
        {"cwd": "/home/runner/tennis3d", "files": {"scripts/azure_guest.py": "a" * 64}}
    )
    manifest = path.parent / "guest.json"
    cpu.save(manifest, m)
    responses["vm run-command"] = {
        "value": [{"message": "TENNIS_STARTED_" + c["run_id"]}]
    }
    cpu.save(path.parent / "responses.json", responses)
    timer_stub(monkeypatch, path)
    assert cpu.lifecycle("start", path, fake, manifest=manifest)["start_requested"]
    call = requests(path)[-1]
    shell = call[call.index("--scripts") + 1]
    assert "Microsoft.DevTestLab" not in json.dumps(requests(path))
    assert "--property=MemoryMax=6G" in shell
    assert "--property=MemorySwapMax=0" in shell
    assert "--property=KillMode=control-group" in shell
    assert "--quota 1000000000" in shell
    assert "ip6tables -I OUTPUT" in shell
    assert shell.index("iptables -I OUTPUT") < shell.index("systemd-run")
    # Azure's transport success is not enough if the guest shell failed.
    responses["vm run-command"] = {"value": [{"message": "[stdout]\n[stderr]failed"}]}
    cpu.save(path.parent / "responses.json", responses)
    with pytest.raises(RuntimeError, match="not confirmed"):
        cpu.lifecycle("start", path, fake, manifest=manifest)


def test_stop_does_not_claim_success_on_unconfirmed_deallocation(setup, monkeypatch):
    _, path, fake, responses = setup
    responses["vm get-instance-view"]["instanceView"]["statuses"][0]["code"] = (
        "PowerState/running"
    )
    cpu.save(path.parent / "responses.json", responses)
    monkeypatch.setattr(cpu.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="not confirmed"):
        cpu.lifecycle("stop", path, fake)


def test_disk_and_ip_forecast_covers_all_compute_sessions():
    c = config()
    c["budget"]["compute_hours"] = 10
    with pytest.raises(ValueError, match="tail"):
        cpu.validate(c)


@pytest.mark.parametrize(
    "field", ["created_at", "generate_until", "drain_until", "shutdown_at"]
)
def test_controller_rejects_non_guest_utc_format_before_provision(setup, field):
    c, path, fake, _ = setup
    c[field] = c[field].replace("Z", "+00:00")
    cpu.save(path, c)
    with pytest.raises(ValueError, match="ending in Z"):
        cpu.lifecycle("provision", path, fake)
    assert not (path.parent / "calls.jsonl").exists()


@pytest.mark.parametrize("target", ["group show", "vm show"])
def test_tag_disk_refuses_foreign_group_or_vm_without_mutation(setup, target):
    c, path, fake, responses = setup
    responses[target]["tags"]["run_id"] = "foreign"
    cpu.save(path.parent / "responses.json", responses)
    with pytest.raises(ValueError, match="ownership"):
        cpu.tag_attached_disk(cpu.azure(fake, c["subscription"]), c)
    assert not any(
        call[:2] in (["tag", "create"], ["vm", "deallocate"], ["group", "delete"])
        for call in requests(path)
    )


def test_implicit_disk_requires_both_attachment_directions(setup):
    c, path, fake, responses = setup
    responses["disk show"]["managedBy"] = "different-vm"
    cpu.save(path.parent / "responses.json", responses)
    with pytest.raises(ValueError, match="attachment"):
        cpu.tag_attached_disk(cpu.azure(fake, c["subscription"]), c)
    assert not any(call[:2] == ["tag", "create"] for call in requests(path))


@pytest.mark.parametrize(
    "fault",
    [
        "controller_loss",
        "foreign_group",
        "foreign_vm",
        "schedule_refused",
        "unknown_state",
        "ready_then_failed",
    ],
)
def test_partial_provision_keeps_native_deadline_and_refuses_unsafe_mutation(
    provision_setup, monkeypatch, fault
):
    c, path, _, _ = provision_setup
    group_id, _, ids = cpu.identities(c)
    state: dict[str, Any] = {
        "group": None,
        "vm": None,
        "operation": None,
        "calls": [],
        "ready": False,
        "cancelled": False,
    }

    def az(*args):
        state["calls"].append(args)
        key = args[:2]
        action = operation_action(args)
        if key == ("group", "exists"):
            return False
        if action == "SubmitDeallocate":
            assert state["group"] is None
            assert state["vm"] is None
            assert request_body(args)["resources"]["ids"] == [ids["vm"]]
            if fault == "schedule_refused":
                raise RuntimeError("missing target rejected")
            state["operation"] = operation(c, "PendingScheduling")
            return state["operation"]
        if action == "CancelOperations":
            state["cancelled"] = True
            return {}
        if action == "GetOperationStatus":
            assert request_body(args)["operationIds"] == [OPERATION_ID]
            if state["cancelled"]:
                return operation(c, "Cancelled")
            if fault == "unknown_state" or (
                fault == "ready_then_failed" and state["group"]
            ):
                return operation(c, "Unknown")
            state["operation"] = operation(c)
            state["ready"] = True
            return state["operation"]
        if key == ("group", "create"):
            assert state["ready"]
            receipt = cpu.read(path.with_suffix(".receipt.json"))
            assert receipt["operation"]["id"] == OPERATION_ID
            state["group"] = {"id": group_id, "tags": cpu.tags(c)}
            return state["group"]
        if key == ("group", "show"):
            return state["group"]
        if key == ("resource", "list"):
            return [state["vm"]] if state["vm"] else []
        if key == ("deployment", "group"):
            assert state["operation"]["results"][0]["operation"]["state"] == "Scheduled"
            state["vm"] = {"id": ids["vm"], "tags": cpu.tags(c)}
            if fault == "controller_loss":
                # Process loss skips the controller's exception recovery.
                raise KeyboardInterrupt("controller vanished after Azure created VM")
            if fault == "foreign_group":
                state["group"]["tags"]["run_id"] = "foreign"
            if fault == "foreign_vm":
                state["vm"]["tags"]["run_id"] = "foreign"
            raise RuntimeError("partial deployment failed")
        if key == ("vm", "show"):
            return state["vm"]
        raise AssertionError(f"unexpected request: {args}")

    monkeypatch.setattr(cpu, "azure", lambda *_: az)
    monkeypatch.setattr(cpu.time, "sleep", lambda _: None)
    timer_stub(monkeypatch, path)
    with pytest.raises((KeyboardInterrupt, RuntimeError, ValueError)):
        cpu.lifecycle("provision", path, "/fake/az")
    assert not any(
        call[:2] in (("vm", "deallocate"), ("group", "delete"))
        for call in state["calls"]
    )
    assert state["cancelled"] == (fault in {"unknown_state", "ready_then_failed"})
    disarmed = any(
        call[:3] == ["systemctl", "--user", "stop"] for call in requests(path)
    )
    assert disarmed == (fault in {"unknown_state", "ready_then_failed"})
    if fault in {"schedule_refused", "unknown_state", "ready_then_failed"}:
        assert state["vm"] is None
        assert not any(call[:2] == ("deployment", "group") for call in state["calls"])
        if fault != "ready_then_failed":
            assert state["group"] is None
    else:
        assert state["vm"] is not None
        assert state["operation"]["results"][0]["operation"]["state"] == "Scheduled"
        assert (
            cpu.read(path.with_suffix(".receipt.json"))["operation"]["id"]
            == OPERATION_ID
        )
    if fault == "unknown_state":
        assert (
            sum(
                operation_action(call) == "GetOperationStatus"
                for call in state["calls"]
            )
            >= 2
        )


def test_template_and_id_allowlist_exclude_devtestlab():
    c = config()
    cpu.validate(c)  # No daily-schedule region is needed.
    template = cpu.template(c)
    ids = cpu.identities(c)[2]
    assert set(ids) == {"vm", "os", "ip", "nic", "vnet", "nsg"}
    assert all(r["location"] == c["region"] for r in template["resources"])
    assert "Microsoft.DevTestLab" not in json.dumps(template)
    assert "Microsoft.DevTestLab" not in json.dumps(ids)


def test_old_daily_schedule_is_unexpected_inventory(setup):
    c, path, fake, responses = setup
    group = cpu.identities(c)[0]
    responses["resource list"].append(
        {
            "id": group
            + "/providers/Microsoft.DevTestLab/schedules/shutdown-computevm-cpu-vm",
            "tags": cpu.tags(c),
        }
    )
    cpu.save(path.parent / "responses.json", responses)
    with pytest.raises(ValueError, match="unexpected"):
        cpu.lifecycle("cleanup", path, fake, recovery(path, c))
    assert not any(call[:2] == ["group", "delete"] for call in requests(path))


@pytest.mark.parametrize("stage", ["SubmitDeallocate", "GetOperationStatus"])
@pytest.mark.parametrize(
    "fault",
    [
        "item_error",
        "operation_error",
        "item_identity",
        "operation_identity",
        "deadline",
        "operation_type",
        "deadline_type",
        "subscription",
        "region",
        "retry_policy",
        "missing_id",
        "multiple_results",
    ],
)
def test_native_operation_refuses_service_errors_and_mismatches_before_create(
    provision_setup, monkeypatch, stage, fault
):
    c, path, fake, responses = provision_setup
    response = operation(c)
    item = response["results"][0]
    op = item["operation"]
    if fault == "item_error":
        item["errorCode"] = "ServiceFailure"
    elif fault == "operation_error":
        op["resourceOperationError"] = {"code": "ServiceFailure"}
    elif fault == "item_identity":
        item["resourceId"] += "-foreign"
    elif fault == "operation_identity":
        op["resourceId"] += "-foreign"
    elif fault == "deadline":
        op["deadline"] = c["drain_until"]
    elif fault == "operation_type":
        op["opType"] = "Start"
    elif fault == "deadline_type":
        op["deadlineType"] = "CompleteBy"
    elif fault == "subscription":
        op["subscriptionId"] = "foreign"
    elif fault == "region":
        response["location"] = "polandcentral"
    elif fault == "retry_policy":
        op["retryPolicy"]["retryWindowInMinutes"] = 90
    elif fault == "missing_id":
        del op["operationId"]
    else:
        response["results"].append(item.copy())
    responses[stage] = response
    responses["group exists"] = False
    cpu.save(path.parent / "responses.json", responses)
    timer_stub(monkeypatch, path)
    with pytest.raises(ValueError):
        cpu.lifecycle("provision", path, fake)
    calls = requests(path)
    assert not any(
        call[:2] in (["group", "create"], ["deployment", "group"]) for call in calls
    )
    receipt = cpu.read(path.with_suffix(".receipt.json"))
    assert receipt["operation_response"] == responses["SubmitDeallocate"]


def test_native_status_refuses_different_operation_id(provision_setup, monkeypatch):
    c, path, fake, responses = provision_setup
    responses["group exists"] = False
    responses["GetOperationStatus"] = operation(c, operation_id="foreign-operation")
    cpu.save(path.parent / "responses.json", responses)
    timer_stub(monkeypatch, path)
    with pytest.raises(ValueError, match="identity/deadline"):
        cpu.lifecycle("provision", path, fake)
    assert not any(call[:2] == ["group", "create"] for call in requests(path))


def test_provision_refuses_existing_operation_receipt(setup, monkeypatch):
    _, path, fake, responses = setup
    responses["group exists"] = False
    cpu.save(path.parent / "responses.json", responses)
    timer_stub(monkeypatch, path)
    before = path.with_suffix(".receipt.json").read_bytes()
    with pytest.raises(ValueError, match="receipt already exists"):
        cpu.lifecycle("provision", path, fake)
    assert path.with_suffix(".receipt.json").read_bytes() == before
    assert not any(
        operation_action(call) == "SubmitDeallocate" or call[:2] == ["group", "create"]
        for call in requests(path)
    )


@pytest.mark.parametrize("state", ["Unknown", "Cancelled", "Failed", "Executing"])
def test_start_requires_live_error_free_native_operation(setup, monkeypatch, state):
    c, path, fake, responses = setup
    responses["GetOperationStatus"] = operation(c, state)
    cpu.save(path.parent / "responses.json", responses)
    manifest = path.parent / "manifest.json"
    cpu.save(
        manifest,
        {
            key: c[key]
            for key in ("run_id", "source_commit", "generate_until", "drain_until")
        },
    )
    timer_stub(monkeypatch, path)
    with pytest.raises(ValueError, match="not pending"):
        cpu.lifecycle("start", path, fake, manifest=manifest)
    assert not any(call[:2] == ["vm", "run-command"] for call in requests(path))


def test_start_requires_local_timer_even_with_native_deadline(setup, monkeypatch):
    c, path, fake, responses = setup
    cpu.save(path.parent / "responses.json", responses)
    manifest = path.parent / "manifest.json"
    cpu.save(
        manifest,
        {
            key: c[key]
            for key in ("run_id", "source_commit", "generate_until", "drain_until")
        },
    )
    timer_stub(monkeypatch, path, fail=True)
    with pytest.raises(cpu.subprocess.CalledProcessError):
        cpu.lifecycle("start", path, fake, manifest=manifest)
    assert not any(call[:2] == ["vm", "run-command"] for call in requests(path))


@pytest.mark.parametrize("terminal", ["Cancelled", "Succeeded", "Failed"])
def test_cleanup_cancels_only_recorded_operation_and_confirms_terminal(
    setup, monkeypatch, terminal
):
    c, path, fake, responses = setup
    responses["GetOperationStatus"] = {
        "sequence": [operation(c), operation(c, "Cancelling"), operation(c, terminal)]
    }
    cpu.save(path.parent / "responses.json", responses)
    sleeps = []
    # Keep subprocess polling on real time; only stub the controller's retry wait.
    monkeypatch.setattr(
        cpu,
        "time",
        SimpleNamespace(monotonic=cpu.time.monotonic, sleep=sleeps.append),
    )
    timer_stub(monkeypatch, path)
    cpu.lifecycle("cleanup", path, fake, recovery(path, c))
    calls = requests(path)
    native = [call for call in calls if operation_action(call)]
    assert [operation_action(call) for call in native] == [
        "GetOperationStatus",
        "CancelOperations",
        "GetOperationStatus",
        "GetOperationStatus",
    ]
    for call in native:
        assert request_body(call) == {
            "operationIds": [OPERATION_ID],
            "correlationid": CORRELATION_ID,
        }
    assert calls[-1][:2] == ["group", "delete"]
    assert sleeps == [5]
    assert (
        cpu.read(path.with_suffix(".receipt.json"))["operation"]["terminal_state"]
        == terminal
    )


@pytest.mark.parametrize("fault", ["never_terminal", "wrong_id"])
def test_cleanup_refuses_delete_without_confirmed_owned_operation_terminal(
    setup, monkeypatch, fault
):
    c, path, fake, responses = setup
    responses["GetOperationStatus"] = {"sequence": [operation(c)]}
    if fault == "wrong_id":
        responses["GetOperationStatus"] = {
            "sequence": [operation(c), operation(c, "Cancelled", "foreign")]
        }
    cpu.save(path.parent / "responses.json", responses)
    monkeypatch.setattr(cpu.time, "sleep", lambda _: None)
    with pytest.raises(
        (RuntimeError, ValueError), match="not confirmed|identity/deadline"
    ):
        cpu.lifecycle("cleanup", path, fake, recovery(path, c))
    assert not any(call[:2] == ["group", "delete"] for call in requests(path))
    assert (
        "terminal_state" not in cpu.read(path.with_suffix(".receipt.json"))["operation"]
    )


@pytest.mark.parametrize("quota", [0, -1, 1.5, True, 2_000_000_000])
def test_invalid_outbound_quota_refused_before_any_azure_request(
    provision_setup, quota
):
    c, path, fake, _ = provision_setup
    c["max_outbound_bytes"] = quota
    cpu.save(path, c)
    with pytest.raises(ValueError):
        cpu.lifecycle("provision", path, fake)
    assert not (path.parent / "calls.jsonl").exists()


def test_provision_requires_user_linger_before_arming_or_creating(
    provision_setup, monkeypatch
):
    _, path, fake, responses = provision_setup
    responses["group exists"] = False
    cpu.save(path.parent / "responses.json", responses)
    timer_stub(monkeypatch, path)
    original = cpu.subprocess.run

    def run(argv, **kwargs):
        if argv[0] == "loginctl":
            return cpu.subprocess.CompletedProcess(argv, 0, stdout="no\n")
        return original(argv, **kwargs)

    monkeypatch.setattr(cpu.subprocess, "run", run)
    with pytest.raises(ValueError, match="linger|Linger"):
        cpu.lifecycle("provision", path, fake)
    if (path.parent / "calls.jsonl").exists():
        assert not any(
            call[0] == "systemd-run"
            or operation_action(call) == "SubmitDeallocate"
            or call[:2] == ["group", "create"]
            for call in requests(path)
        )


@pytest.mark.parametrize(
    "end_state", ["Succeeded", "Failed", "Canceled", "timeout", "unsettled"]
)
def test_async_deployment_waits_for_terminal_or_cancels_at_deadline(
    monkeypatch, tmp_path, end_state
):
    c = config()
    current = cpu.utc(c["generate_until"]) - dt.timedelta(seconds=10)
    calls = []
    polls = []

    def az(*args):
        calls.append(args)
        if args[:3] == ("deployment", "group", "show"):
            polls.append(args)
            state = (
                "Running"
                if len(polls) == 1 or end_state in {"timeout", "unsettled"}
                else end_state
            )
            if end_state == "timeout" and len(polls) > 2:
                state = "Canceled"
            return {"properties": {"provisioningState": state}}
        return {}

    monkeypatch.setattr(cpu, "owned", lambda *args, **kwargs: [])

    def sleep(seconds):
        nonlocal current
        current += dt.timedelta(seconds=seconds)

    monkeypatch.setattr(cpu, "now", lambda: current)
    monkeypatch.setattr(cpu.time, "sleep", sleep)
    if end_state == "Succeeded":
        cpu.deploy(az, c, tmp_path / "arm.json")
    else:
        with pytest.raises(
            RuntimeError, match="failed|deadline exceeded|cancellation unconfirmed"
        ):
            cpu.deploy(az, c, tmp_path / "arm.json")
    assert calls[0][:3] == ("deployment", "group", "create")
    assert "--no-wait" in calls[0]
    assert len(polls) == ({"timeout": 3, "unsettled": 14}.get(end_state, 2))
    assert any(call[:3] == ("deployment", "group", "cancel") for call in calls) == (
        end_state in {"timeout", "unsettled"}
    )


@pytest.mark.parametrize("group_exists", [False, True])
def test_cleanup_without_billable_resources_cancels_operation_and_disarms_timer(
    setup, monkeypatch, group_exists
):
    _, path, fake, responses = setup
    responses["group exists"] = group_exists
    responses["resource list"] = []
    cpu.save(path.parent / "responses.json", responses)
    timer_stub(monkeypatch, path)
    result = cpu.lifecycle("cleanup", path, fake)
    calls = requests(path)
    assert any(operation_action(call) == "CancelOperations" for call in calls)
    assert any(call[:3] == ["systemctl", "--user", "stop"] for call in calls)
    assert not any(call[:2] == ["vm", "deallocate"] for call in calls)
    assert any(call[:2] == ["group", "delete"] for call in calls) == group_exists
    assert (
        cpu.read(path.with_suffix(".receipt.json"))["operation"]["terminal_state"]
        == "Cancelled"
    )
    if not group_exists:
        assert result == {"group_absent": True, "confirmed": True}


def test_operation_accepts_equivalent_azure_utc_offset():
    c = config()
    response = operation(c)
    response["results"][0]["operation"]["deadline"] = c["shutdown_at"].replace(
        "Z", "+00:00"
    )
    _, op = cpu.operation_result(response, c, OPERATION_ID)
    assert op["operationId"] == OPERATION_ID


@pytest.mark.parametrize("empty", [False, True])
def test_cleanup_keeps_all_controls_while_arm_still_runs(setup, monkeypatch, empty):
    c, path, fake, responses = setup
    responses["deployment group show"] = {
        "properties": {"provisioningState": "Running"}
    }
    if empty:
        responses["resource list"] = []
    cpu.save(path.parent / "responses.json", responses)
    timer_stub(monkeypatch, path)
    with pytest.raises(ValueError, match="ARM deployment is not terminal"):
        cpu.lifecycle("cleanup", path, fake, recovery(path, c))
    assert not any(
        operation_action(call) == "CancelOperations"
        or call[:3] == ["systemctl", "--user", "stop"]
        or call[:2] == ["group", "delete"]
        for call in requests(path)
    )


def test_partial_network_only_cleanup_requires_terminal_arm_but_no_data_copy(
    setup, monkeypatch
):
    c, path, fake, responses = setup
    ids = cpu.identities(c)[2]
    responses["resource list"] = [{"id": ids["ip"], "tags": cpu.tags(c)}]
    cpu.save(path.parent / "responses.json", responses)
    timer_stub(monkeypatch, path)
    assert cpu.lifecycle("cleanup", path, fake)[
        "owned_group_without_data_deletion_requested"
    ]
    assert not any(call[:2] == ["vm", "get-instance-view"] for call in requests(path))
    assert requests(path)[-1][:2] == ["group", "delete"]


@pytest.mark.parametrize("short_by_seconds", [1, 0])
def test_compute_minimum_includes_native_retry_window(short_by_seconds):
    c = config()
    seconds = (cpu.utc(c["shutdown_at"]) - cpu.utc(c["created_at"])).total_seconds()
    c["budget"]["compute_hours"] = (seconds + 300 - short_by_seconds) / 3600
    if short_by_seconds:
        with pytest.raises(ValueError, match="native retry window"):
            cpu.validate(c)
    else:
        assert cpu.validate(c) > 0


def test_cleanup_refreshes_inventory_after_arm_creates_vm(setup, monkeypatch):
    c, path, fake, responses = setup
    final_inventory = responses["resource list"]
    responses["resource list"] = []
    responses["vm get-instance-view"]["instanceView"]["statuses"] = [
        {"code": "PowerState/running"}
    ]
    cpu.save(path.parent / "responses.json", responses)
    original = cpu.azure(fake, c["subscription"])

    def az(*args):
        if args[:3] == ("deployment", "group", "show"):
            # VM/disk appear after the original inventory, before terminal readback.
            responses["resource list"] = final_inventory
            cpu.save(path.parent / "responses.json", responses)
        return original(*args)

    monkeypatch.setattr(cpu, "azure", lambda *_: az)
    timer_stub(monkeypatch, path)
    with pytest.raises(ValueError, match="confirm deallocation"):
        cpu.lifecycle("cleanup", path, fake, recovery(path, c))
    calls = requests(path)
    assert sum(call[:2] == ["resource", "list"] for call in calls) == 2
    assert not any(
        operation_action(call) == "CancelOperations"
        or call[:3] == ["systemctl", "--user", "stop"]
        or call[:2] == ["group", "delete"]
        for call in calls
    )


def test_stop_waits_for_normal_deallocation_progress(setup, monkeypatch):
    _, path, fake, responses = setup
    responses["vm get-instance-view"] = {
        "sequence": [
            {"instanceView": {"statuses": [{"code": state}]}}
            for state in [
                "PowerState/running",
                "PowerState/deallocating",
                "PowerState/deallocating",
                "PowerState/deallocated",
            ]
        ]
    }
    cpu.save(path.parent / "responses.json", responses)
    sleeps = []
    monkeypatch.setattr(
        cpu,
        "time",
        SimpleNamespace(monotonic=cpu.time.monotonic, sleep=sleeps.append),
    )
    assert cpu.lifecycle("stop", path, fake)["confirmed"]
    assert sleeps == [5, 5, 5]


def test_stop_wall_clock_bound_includes_cli_latency(setup, monkeypatch):
    c, _, _, responses = setup
    current = 0.0
    calls = []

    def az(*args, timeout):
        nonlocal current
        calls.append((current, timeout))
        current += timeout
        if args[:2] == ("vm", "get-instance-view"):
            return {"instanceView": {"statuses": [{"code": "PowerState/deallocating"}]}}
        return responses[" ".join(args[:2])]

    def sleep(seconds):
        nonlocal current
        current += seconds

    monkeypatch.setattr(cpu.time, "monotonic", lambda: current)
    monkeypatch.setattr(cpu.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="not confirmed"):
        cpu.deallocate_owned(az, c)
    assert current <= 240
    assert all(start + timeout <= 240 and timeout <= 30 for start, timeout in calls)


@pytest.mark.parametrize("delete_fails", [False, True])
def test_rejected_deployment_empty_group_retains_controls_until_absent(
    setup, monkeypatch, delete_fails
):
    _, path, fake, responses = setup
    responses["deployment group list"] = []
    responses["resource list"] = []
    del responses["deployment group show"]
    if delete_fails:
        del responses["group delete"]
    cpu.save(path.parent / "responses.json", responses)
    timer_stub(monkeypatch, path)
    if delete_fails:
        with pytest.raises(RuntimeError, match="failed"):
            cpu.lifecycle("cleanup", path, fake)
    else:
        result = cpu.lifecycle("cleanup", path, fake)
        assert result["shutdown_controls_retained"] and not result["confirmed"]
    calls = requests(path)
    assert not any(
        operation_action(call) == "CancelOperations"
        or call[:3] == ["systemctl", "--user", "stop"]
        for call in calls
    )
    assert calls[-1][:2] == ["group", "delete"]
    if not delete_fails:
        responses["group exists"] = False
        cpu.save(path.parent / "responses.json", responses)
        assert cpu.lifecycle("cleanup", path, fake)["group_absent"]
        assert any(
            operation_action(call) == "CancelOperations" for call in requests(path)
        )
        assert any(
            call[:3] == ["systemctl", "--user", "stop"] for call in requests(path)
        )


@pytest.mark.parametrize(
    "fault", ["list_failure", "malformed", "unexpected", "new_resource"]
)
def test_missing_deployment_recovery_refuses_uncertain_or_nonempty_group(
    setup, monkeypatch, fault
):
    c, path, fake, responses = setup
    responses["resource list"] = []
    responses["deployment group list"] = []
    if fault == "list_failure":
        del responses["deployment group list"]
    elif fault == "malformed":
        responses["deployment group list"] = {}
    elif fault == "unexpected":
        responses["deployment group list"] = [{"id": "foreign-deployment"}]
    else:
        responses["resource list"] = {
            "sequence": [[], [{"id": cpu.identities(c)[2]["ip"], "tags": cpu.tags(c)}]]
        }
    cpu.save(path.parent / "responses.json", responses)
    timer_stub(monkeypatch, path)
    with pytest.raises((RuntimeError, ValueError)):
        cpu.lifecycle("cleanup", path, fake)
    assert not any(
        operation_action(call) == "CancelOperations"
        or call[:3] == ["systemctl", "--user", "stop"]
        or call[:2] == ["group", "delete"]
        for call in requests(path)
    )


def test_cleanup_releases_never_run_service_and_repeated_absent_units(
    setup, monkeypatch
):
    c, path, fake, _ = setup
    saved = recovery(path, c)
    original = cpu.subprocess.run
    stops = []

    def run(argv, **kwargs):
        if argv[:2] != ["systemctl", "--user"]:
            return original(argv, **kwargs)
        name = argv[3]
        assert name in {cpu.unit(c) + ".timer", cpu.unit(c) + ".service"}
        if argv[2] == "stop":
            # First cleanup stops the timer but the never-run service is absent.
            # Subsequent cleanup finds both absent, as in a saved readback.
            code = 0 if not stops else 5
            stops.append(name)
            return cpu.subprocess.CompletedProcess(argv, code)
        assert argv[2] == "show" and kwargs["check"] is True
        return cpu.subprocess.CompletedProcess(
            argv, 0, stdout="LoadState=not-found\nActiveState=inactive\nSubState=dead\n"
        )

    monkeypatch.setattr(cpu.subprocess, "run", run)
    for _ in range(2):
        assert cpu.lifecycle("cleanup", path, fake, saved)["deletion_requested"]
    assert stops == [cpu.unit(c) + suffix for suffix in (".timer", ".service") * 2]
    assert (
        cpu.read(path.with_suffix(".receipt.json"))["operation"]["terminal_state"]
        == "Cancelled"
    )
    assert sum(call[:2] == ["group", "delete"] for call in requests(path)) == 2


@pytest.mark.parametrize(
    "fault",
    [
        "stop_error",
        "read_error",
        "active",
        "failed",
        "missing_state",
        "loaded_stop_error",
    ],
)
def test_cleanup_refuses_uncertain_or_active_local_controls(setup, monkeypatch, fault):
    c, path, fake, _ = setup
    original = cpu.subprocess.run

    def run(argv, **kwargs):
        if argv[:2] != ["systemctl", "--user"]:
            return original(argv, **kwargs)
        assert argv[3] == cpu.unit(c) + ".timer"
        if argv[2] == "stop":
            code = 1 if fault == "stop_error" else 5
            return cpu.subprocess.CompletedProcess(argv, code)
        if fault == "read_error":
            raise cpu.subprocess.CalledProcessError(1, argv)
        states = {
            "active": "LoadState=not-found\nActiveState=active\nSubState=running\n",
            "failed": "LoadState=error\nActiveState=inactive\nSubState=dead\n",
            "missing_state": "LoadState=not-found\nActiveState=inactive\n",
            "loaded_stop_error": "LoadState=loaded\nActiveState=inactive\nSubState=dead\n",
        }
        return cpu.subprocess.CompletedProcess(
            argv,
            0,
            stdout=states.get(
                fault, "LoadState=not-found\nActiveState=inactive\nSubState=dead\n"
            ),
        )

    monkeypatch.setattr(cpu.subprocess, "run", run)
    with pytest.raises((ValueError, cpu.subprocess.CalledProcessError)):
        cpu.lifecycle("cleanup", path, fake, recovery(path, c))
    assert not any(call[:2] == ["group", "delete"] for call in requests(path))


def test_disarm_loaded_units_requires_success_and_inactive_readback(monkeypatch):
    c = config()
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return cpu.subprocess.CompletedProcess(
            argv, 0, stdout="LoadState=loaded\nActiveState=inactive\nSubState=dead\n"
        )

    monkeypatch.setattr(cpu.subprocess, "run", run)
    cpu.disarm_timer(c)
    assert [call[2] for call in calls] == ["stop", "show", "stop", "show"]
