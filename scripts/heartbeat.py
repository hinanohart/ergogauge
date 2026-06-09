#!/usr/bin/env python3
"""Update the autonomous-build heartbeat in .ergogauge-progress.json (build bookkeeping)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROGRESS = Path(__file__).resolve().parent.parent / ".ergogauge-progress.json"


def main() -> int:
    if not PROGRESS.exists():
        return 0
    stamp = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    data = json.loads(PROGRESS.read_text(encoding="utf-8"))
    data.setdefault("session_lock", {})["last_heartbeat_utc"] = stamp
    data["session_lock"]["pid"] = os.getpid()
    tmp = PROGRESS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, PROGRESS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
