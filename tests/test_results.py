from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from tennis3d.artifacts import read_json
from tennis3d.schemas import Result, read_result, write_result

FIXTURE = Path(__file__).parent / "fixtures" / "trajectory.json"


def test_fixture_round_trip_preserves_timestamps_and_gaps(tmp_path):
    result = read_result(FIXTURE)
    output = tmp_path / "result.json"
    write_result(output, result)
    assert read_result(output) == result
    assert [f.t_s for f in result.frames] == [0, 0.04, 0.08, 0.12, 0.16, 0.2]
    assert result.frames[3].xyz_m is None
    assert result.frames[2].duplicate_of == 1
    with pytest.raises(FileExistsError):
        write_result(output, result)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema_version="2.0"),
        lambda d: d.update(extra="no"),
        lambda d: d["frames"][0].update(xyz_m=[1, 2, float("nan")]),
        lambda d: d["frames"][0].update(xyz_m=[1, 2, float("inf")]),
        lambda d: d["frames"][0].update(xyz_m=[1, 2]),
        lambda d: d["frames"][0].update(xyz_m=None),
        lambda d: d["frames"][3].update(xyz_m=[1, 2, 3]),
        lambda d: d["frames"][0].update(camera_valid=False),
        lambda d: d["frames"][0].update(reconstruction="inferred"),
        lambda d: d["frames"][1].update(frame_index=0),
        lambda d: d["frames"][1].update(t_s=0.0),
        lambda d: d["frames"][0].update(t_s=0.001),
        lambda d: d["frames"][2].update(duplicate_of=2),
        lambda d: d["frames"][2].update(duplicate_of=None),
        lambda d: d["frames"][3].update(reason_codes=[]),
        lambda d: d["frames"][0].update(frame_index=True),
        lambda d: d["source"].update(frame_count=99),
        lambda d: d["source"].update(sha256="recorded-at-runtime"),
        lambda d: d["source"].update(time_base=[1, 0]),
        lambda d: d["summary"].update(valid_frame_fraction=1.0),
        lambda d: d.update(status="ok"),
        lambda d: d["coordinates"].update(units="mm"),
    ],
)
def test_reject_invalid_results(change):
    data = deepcopy(read_json(FIXTURE))
    change(data)
    with pytest.raises(ValidationError):
        Result.model_validate(data)


def test_failed_input_and_empty_decoder_report():
    data = read_json(FIXTURE)
    data.update(status="failed", failure_reasons=["unsupported_shot"])
    assert Result.model_validate(data).status == "failed"
    data["frames"] = []
    data["source"]["frame_count"] = 0
    data["summary"] = {"valid_frame_fraction": 0.0, "detected_frame_fraction": 0.0}
    data["failure_reasons"] = ["decoder_error"]
    assert Result.model_validate(data).status == "failed"
