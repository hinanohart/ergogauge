"""The ergodicity certificate: a JSON-serializable, byte-stable result object."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

SCHEMA_VERSION = "ergogauge/0.1"
_ROUND_NDIGITS = 8


def _stabilize(obj: Any) -> Any:
    """Recursively round floats so JSON is byte-identical across reruns and platforms."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        if obj in (float("inf"), float("-inf")):
            return None
        return round(obj, _ROUND_NDIGITS)
    if isinstance(obj, dict):
        return {k: _stabilize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stabilize(v) for v in obj]
    return obj


@dataclass
class Certificate:
    """Result of an ergogauge analysis."""

    gate: dict[str, Any]
    levels: list[dict[str, Any]]
    aggregate: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "schema_version": self.schema_version,
            "gate": self.gate,
            "levels": self.levels,
            "aggregate": self.aggregate,
            "meta": self.meta,
        }
        return cast("dict[str, Any]", _stabilize(raw))

    def to_json(self, path: str | Path | None = None) -> str:
        s = json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(s + "\n", encoding="utf-8")
        return s

    def to_html(self, path: str | Path) -> None:
        from .viz import render_html

        Path(path).write_text(render_html(self), encoding="utf-8")
