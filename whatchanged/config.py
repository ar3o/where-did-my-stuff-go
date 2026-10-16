"""Stage settings: one YAML file per stage in configs/, read into nested dataclasses.

Values written as `$NAME` are taken from the environment variable NAME; when it is unset, the code default is used, as
for keys missing from the file. The defaults in code and in configs/ are the same settings.
"""

from __future__ import annotations

import dataclasses
import os
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

from whatchanged import paths

T = TypeVar("T")


_UNSET = object()


def _expand(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        raw = os.environ.get(value[1:])
        return _UNSET if raw is None else yaml.safe_load(raw) if raw else raw
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def _build(cls: type[T], raw: dict) -> T:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown settings {sorted(unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in raw:
            continue
        value = _expand(raw[f.name])
        if value is _UNSET:
            continue
        hint = hints[f.name]
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            value = _build(hint, value)
        kwargs[f.name] = value
    return cls(**kwargs)


def load(cls: type[T], stage: str, path: str | Path | None = None) -> T:
    """Settings of `stage` from `path` (default configs/<stage>.yaml)."""
    path = Path(path) if path else paths.CONFIGS / f"{stage}.yaml"
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    return _build(cls, raw or {})
