"""Stage configs subclass StrictModel; load defaults, file, then explicit overrides."""

from pathlib import Path
from typing import TypeVar

import yaml
from yaml.resolver import BaseResolver

from tennis3d.artifacts import canonical_json, json_sha256, read_json, write_json
from tennis3d.schemas import StrictModel

ConfigT = TypeVar("ConfigT", bound=StrictModel)


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in result:
            raise ValueError(f"duplicate config key: {key}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    model: type[ConfigT],
    path: str | Path | None = None,
    *,
    overrides: dict | None = None,
) -> ConfigT:
    """Load a YAML/JSON mapping. CLI overrides contain only explicitly set values.

    File values and overrides must be JSON-serializable. Strict JSON validation
    supports typed paths, tuples and enums without numeric-string coercion.
    Nested models must also inherit StrictModel. Paths remain as written; stage
    code defines their base. No environment expansion or hidden global settings.
    """
    values = {}
    if path is not None:
        path = Path(path)
        values = (
            read_json(path)
            if path.suffix.lower() == ".json"
            else yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
        )
        if not isinstance(values, dict):
            raise ValueError("config must contain a mapping")
    resolved = model.model_validate_json(
        canonical_json(_merge(values, overrides or {}))
    )
    # JSON mode parses typed file values; Python mode also checks native defaults.
    return model.model_validate(resolved.model_dump())


def record_config(
    path: str | Path, config: StrictModel, *, overwrite: bool = False
) -> str:
    """Write the complete resolved config atomically and return its canonical hash."""
    values = type(config).model_validate(config.model_dump()).model_dump(mode="json")
    write_json(path, values, overwrite=overwrite)
    return json_sha256(values)
