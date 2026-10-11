import json
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
from pydantic import Field, ValidationError

from tennis3d.artifacts import (
    ArtifactProvenance,
    atomic_file,
    file_sha256,
    json_sha256,
    read_json,
    read_npz,
    write_json,
    write_npz,
)
from tennis3d.config import load_config, record_config
from tennis3d.schemas import StrictModel


class Nested(StrictModel):
    size: int = Field(default=4, gt=0)
    enabled: bool = True


class StageConfig(StrictModel):
    seed: int = 1
    nested: Nested = Field(default_factory=Nested)


def test_config_precedence_rejection_and_record(tmp_path):
    path = tmp_path / "stage.yaml"
    path.write_text("seed: 12\nnested:\n  size: 8\n")
    config = load_config(StageConfig, path, overrides={"nested": {"enabled": False}})
    assert config == StageConfig(seed=12, nested=Nested(size=8, enabled=False))
    output = tmp_path / "resolved.json"
    digest = record_config(output, config)
    assert digest == json_sha256(read_json(output))
    assert load_config(StageConfig, output) == config
    for text in (
        "seed: 1\nseed: 2",
        "nested: {typo: 4}",
        "seed: '4'",
        "[]",
        "seed: .nan",
    ):
        path.write_text(text)
        with pytest.raises((ValueError, ValidationError)):
            load_config(StageConfig, path)


def test_atomic_failure_no_clobber_and_no_temporary_files(tmp_path):
    path = tmp_path / "value.json"
    write_json(path, {"original": True})
    with pytest.raises(RuntimeError), atomic_file(path, overwrite=True) as stream:
        stream.write(b"incomplete")
        raise RuntimeError("interrupted")
    assert read_json(path) == {"original": True}
    with pytest.raises(FileExistsError):
        write_json(path, {"replacement": True})
    assert list(tmp_path.iterdir()) == [path]
    write_json(path, {"replacement": True}, overwrite=True)
    assert read_json(path) == {"replacement": True}
    assert len(file_sha256(path)) == 64


def test_numeric_npz_provenance_and_pickle_rejection(tmp_path):
    path = tmp_path / "frames.npz"
    meta = ArtifactProvenance(
        source_sha256="a" * 64, config_sha256="b" * 64, code_revision="test"
    )
    write_npz(
        path, {"t_s": np.array([0, 0.04]), "valid": np.array([True, False])}, meta
    )
    arrays, actual = read_npz(path, expected=meta)
    assert actual == meta
    np.testing.assert_array_equal(arrays["valid"], [True, False])
    with pytest.raises(ValueError, match="reuse"):
        read_npz(path, expected=meta.model_copy(update={"source_sha256": "c" * 64}))
    for invalid in (np.array([object()]), np.array(["text"])):
        with pytest.raises(ValueError, match="numeric"):
            write_npz(tmp_path / "bad.npz", {"unsafe": invalid}, meta)
    np.savez(
        tmp_path / "unsafe.npz",
        __provenance__=np.frombuffer(
            json.dumps(meta.model_dump()).encode(), dtype=np.uint8
        ),
        unsafe=np.array([object()]),
    )
    with pytest.raises(ValueError, match="Object arrays"):
        read_npz(tmp_path / "unsafe.npz")


def test_json_rejects_duplicate_keys_and_nonfinite(tmp_path):
    path = tmp_path / "bad.json"
    for text in ('{"a":1,"a":2}', '{"a":NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_json(path)


class FloatConfig(StrictModel):
    learning_rate: float = 1e-5


def test_json_config_exponents_round_trip(tmp_path):
    path = tmp_path / "resolved.json"
    record_config(path, FloatConfig())
    assert load_config(FloatConfig, path) == FloatConfig()
    path.write_text('{"learning_rate":2e-5}')
    assert load_config(FloatConfig, path).learning_rate == 2e-5


@pytest.mark.parametrize("key", ["file", "allow_pickle"])
def test_npz_reserved_names_rejected_before_replacement(tmp_path, key):
    path = tmp_path / "artifact.npz"
    meta = ArtifactProvenance(
        source_sha256="a" * 64, config_sha256="b" * 64, code_revision="test"
    )
    write_npz(path, {"values": np.array([1, 2, 3])}, meta)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="reserved"):
        write_npz(path, {key: np.array([4, 5, 6])}, meta, overwrite=True)
    assert path.read_bytes() == original
    np.testing.assert_array_equal(read_npz(path)[0]["values"], [1, 2, 3])


@pytest.mark.parametrize(
    "payload", ['{"value":1e999}', '{"value":-1e999}', '{"nested":[{"value":1e400}]}']
)
def test_json_rejects_overflow(tmp_path, payload):
    path = tmp_path / "overflow.json"
    path.write_text(payload)
    with pytest.raises(ValueError, match="non-finite"):
        read_json(path)
    path.write_text('{"value":1e-50}')
    assert read_json(path)["value"] == 1e-50


class Mode(Enum):
    LOCAL = "local"


class TypedConfig(StrictModel):
    source: Path
    shape: tuple[int, int]
    mode: Mode


def test_typed_config_round_trip_from_yaml_json_and_overrides(tmp_path):
    path = tmp_path / "typed.yaml"
    path.write_text("source: clip.mp4\nshape: [640, 360]\nmode: local\n")
    expected = TypedConfig(source=Path("clip.mp4"), shape=(640, 360), mode=Mode.LOCAL)
    config = load_config(TypedConfig, path)
    assert config == expected
    resolved = tmp_path / "resolved.json"
    record_config(resolved, config)
    assert load_config(TypedConfig, resolved) == expected
    assert load_config(
        TypedConfig, resolved, overrides={"shape": [320, 180]}
    ).shape == (320, 180)
    with pytest.raises(ValidationError):
        load_config(TypedConfig, resolved, overrides={"shape": ["320", 180]})


class InvalidRangeDefault(StrictModel):
    count: int = Field(default=-1, gt=0)


class InvalidFiniteDefault(StrictModel):
    scale: float = float("nan")


@pytest.mark.parametrize("model", [InvalidRangeDefault, InvalidFiniteDefault])
def test_config_rejects_invalid_defaults(model):
    with pytest.raises(ValidationError):
        load_config(model)


class NativeDefaults(StrictModel):
    source: Path = Path("runs/test")
    shape: tuple[int, int] = (640, 360)
    mode: Mode = Mode.LOCAL


class NestedInvalidDefault(StrictModel):
    nested: InvalidRangeDefault = Field(default_factory=InvalidRangeDefault)


def test_config_validates_native_and_nested_defaults(tmp_path):
    config = load_config(NativeDefaults)
    assert config == NativeDefaults()
    output = tmp_path / "resolved.json"
    record_config(output, config)
    assert load_config(NativeDefaults, output) == config
    with pytest.raises(ValidationError):
        load_config(NestedInvalidDefault)
    assert load_config(InvalidRangeDefault, overrides={"count": 2}).count == 2
