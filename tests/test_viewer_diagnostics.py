import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tennis3d.schemas import Result, read_result

EXAMPLE = Path(__file__).resolve().parents[1] / "viewer/example.json"


def test_example_and_roundtrip():
    result = read_result(EXAMPLE)
    assert Result.model_validate_json(result.model_dump_json()) == result
    assert result.synthetic_diagnostics is not None
    assert len(result.synthetic_diagnostics.frames) == result.source.frame_count


@pytest.mark.parametrize("change", ["alignment", "point", "event", "provenance"])
def test_reject_bad_diagnostic(change):
    data = copy.deepcopy(json.loads(EXAMPLE.read_text()))
    diagnostic = data["synthetic_diagnostics"]
    if change == "alignment":
        diagnostic["frames"][1]["t_s"] += 0.001
    elif change == "point":
        diagnostic["frames"][0]["ball_clean_uv"] = [float("nan"), 1.0]
    elif change == "event":
        diagnostic["events"][0]["t_s"] = 3.0
    else:
        data["provenance"]["trajectory_kind"] = "model_estimate"
    with pytest.raises(ValidationError):
        Result.model_validate(data)
