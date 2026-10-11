"""Secondary archive failures must not replace the producer outcome."""

from argparse import Namespace

import pytest

from tennis3d.synthetic import cli, producer, producer_bundle
from tennis3d.synthetic.producer_parallel import Stopped


@pytest.mark.parametrize(
    "primary,code,message",
    [
        (Stopped("requested stop"), 4, "Simulation stopped"),
        (ValueError("original generation failure"), 2, "original generation failure"),
        (None, 2, "archive failed"),
    ],
)
def test_archive_failure_preserves_primary_outcome(
    tmp_path, monkeypatch, capsys, primary, code, message
):
    output = tmp_path / "output"
    (output / "control").mkdir(parents=True)
    (output / "control/plan.json").write_text("{}")

    def generate(*args, **kwargs):
        if primary is not None:
            raise primary
        return {"state": "complete"}

    def archive(*args, **kwargs):
        raise ValueError("archive failed")

    monkeypatch.setattr(producer, "generate_main", generate)
    monkeypatch.setattr(producer_bundle, "recovery_bundle", archive)
    args = Namespace(
        config=None,
        seed=None,
        format="parquet",
        execution_config=None,
        restore_bundle=None,
        resume=False,
        output=output,
        pool_cache=None,
        recovery_bundle=tmp_path / "recovery.zip",
    )
    assert cli.run_command(args) == code
    text = capsys.readouterr().out
    assert message in text
    if primary is not None:
        assert "Recovery bundle failed: archive failed" in text
