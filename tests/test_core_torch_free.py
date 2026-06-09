"""The core library must import without pulling torch into sys.modules."""

from __future__ import annotations

import subprocess
import sys


def test_core_import_is_torch_free() -> None:
    code = (
        "import sys, ergogauge; "
        "assert 'torch' not in sys.modules, 'core import pulled torch'; "
        "print(ergogauge.__version__)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
