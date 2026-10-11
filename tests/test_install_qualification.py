"""Qualification rejects dropped rows, state changes and mismatched source PTS."""

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import qualify_install
from tennis3d.artifacts import file_sha256
from tennis3d.schemas import read_result

RESULT = Path(__file__).parent / "fixtures" / "trajectory.json"


def test_parity_checks_rows_observations_and_gaps_before_xyz():
    result = read_result(RESULT)
    altered = result.model_copy(deep=True)
    assert qualify_install.result_parity(result, altered) == (True, 0)
    altered.frames.pop()
    with pytest.raises(ValueError, match="row counts"):
        qualify_install.result_parity(result, altered)
    altered = result.model_copy(deep=True)
    altered.frames[0] = altered.frames[0].model_copy(update={"observation": "missing"})
    assert not qualify_install.result_parity(result, altered)[0]
    altered = result.model_copy(deep=True)
    altered.frames[0] = altered.frames[0].model_copy(update={"xyz_m": None})
    assert not qualify_install.result_parity(result, altered)[0]
    altered = result.model_copy(deep=True)
    altered.frames[0] = altered.frames[0].model_copy(update={"t_s": 0.001})
    assert not qualify_install.result_parity(result, altered)[0]


def test_source_alignment_rejects_changed_pts_even_with_same_export_times(
    tmp_path, monkeypatch
):
    from tennis3d.video import decode as decoder

    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture source")
    result = read_result(RESULT)
    result = result.model_copy(
        update={
            "source": result.source.model_copy(update={"sha256": file_sha256(source)})
        }
    )
    rows = [
        SimpleNamespace(
            index=f.frame_index, pts=i * 100, time_base=Fraction(1, 1000), t_s=f.t_s
        )
        for i, f in enumerate(result.frames)
    ]
    arrays = {
        "frame_index": np.array([f.index for f in rows]),
        "pts": np.array([f.pts for f in rows]),
        "time_base": np.array([f.time_base.as_integer_ratio() for f in rows]),
        "t_s": np.array([f.t_s for f in rows]),
    }
    monkeypatch.setattr(decoder, "decode", lambda path: iter(rows))
    monkeypatch.setattr(qualify_install, "read_npz", lambda path: (arrays, {}))
    assert qualify_install.source_alignment(source, tmp_path, result)["rows"] == len(
        rows
    )
    arrays["pts"][1] += 1
    with pytest.raises(ValueError, match="source pts"):
        qualify_install.source_alignment(source, tmp_path, result)
