"""Self-contained HTML report. Pure stdlib by default; matplotlib only if [viz] present.

No external URLs, scripts, or fonts are referenced (CI greps for that).
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .certificate import Certificate

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;color:#1a1a1a;max-width:64rem}
h1{font-size:1.4rem} h2{font-size:1.05rem;margin-top:1.6rem;border-bottom:1px solid #ddd}
table{border-collapse:collapse;margin:.5rem 0} td,th{border:1px solid #ccc;padding:.25rem .6rem;font-size:.85rem}
.flag{display:inline-block;padding:.1rem .5rem;border-radius:.3rem;font-weight:600;font-size:.8rem}
.HEALTHY{background:#d8f5d8} .REPETITIVE{background:#ffe0b3} .LOCKED{background:#ffd0d0}
.COLLAPSED{background:#ffc0c0} .OVER_RANDOM{background:#e0e0ff} .ABSTAIN{background:#eee}
.bar{height:.7rem;background:#4a7} .barbg{background:#eee;width:14rem;display:inline-block}
pre{background:#f6f6f6;padding:.8rem;overflow:auto;font-size:.78rem}
small{color:#666}
"""


def _flag_span(flag: str) -> str:
    return f'<span class="flag {html.escape(flag)}">{html.escape(flag)}</span>'


def _occupancy_svg(pi_top: list[tuple[int, float]]) -> str:
    rows = []
    for state, mass in pi_top:
        w = max(0.2, mass * 14.0)
        rows.append(
            f'<tr><td>{state}</td><td><span class="barbg">'
            f'<span class="bar" style="width:{w:.2f}rem"></span></span> {mass:.4f}</td></tr>'
        )
    return "<table><tr><th>state</th><th>stationary mass</th></tr>" + "".join(rows) + "</table>"


def render_html(cert: Certificate) -> str:
    d: dict[str, Any] = cert.to_dict()
    agg = d.get("aggregate", {})
    agg_flag = (agg.get("flags") or ["ABSTAIN"])[0]
    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>ergogauge certificate</title><style>{_CSS}</style></head><body>",
        "<h1>ergogauge ergodicity certificate</h1>",
        f"<p>aggregate flag: {_flag_span(str(agg_flag))} "
        f"&nbsp; confidence: <b>{html.escape(str(agg.get('confidence', 'low')))}</b> "
        f"&nbsp; status: <b>{html.escape(str(agg.get('status', '')))}</b></p>",
        "<p><small>Calibrated heuristic, not a theorem. CPU-only, reference-free, "
        "decode-free. Flags carry bootstrap CIs and fail closed to ABSTAIN.</small></p>",
    ]
    gate = d.get("gate", {})
    parts.append("<h2>identifiability gate</h2>")
    parts.append(
        f"<p>status <b>{html.escape(str(gate.get('status', '')))}</b>, "
        f"effective pairs {gate.get('effective_n_pairs', '?')}, "
        f"support coverage {gate.get('support_coverage', '?')}, "
        f"sparsity {gate.get('sparsity', '?')}</p>"
    )
    if gate.get("reasons"):
        parts.append(
            "<ul>" + "".join(f"<li>{html.escape(str(r))}</li>" for r in gate["reasons"]) + "</ul>"
        )

    for lvl in d.get("levels", []):
        m = lvl.get("metrics", {})
        parts.append(f"<h2>codebook level {lvl.get('level', '?')}</h2>")
        flags = " ".join(_flag_span(str(f)) for f in lvl.get("flags", []))
        parts.append(
            f"<p>{flags} &nbsp; confidence {html.escape(str(lvl.get('flag_confidence', '')))}</p>"
        )
        rowsout = []
        for name, key in (
            ("spectral gap", "spectral_gap"),
            ("Kemeny", "kemeny"),
            ("Cheeger phi", "cheeger"),
            ("entropy ratio", "over_random_discriminator"),
        ):
            blk = m.get(key, {})
            val = blk.get("value", blk.get("conductance_phi", blk.get("ratio", "?")))
            ci = blk.get("ci95", blk.get("ci95_phi", blk.get("ci95_ratio", [])))
            rowsout.append(f"<tr><td>{name}</td><td>{val}</td><td>{ci}</td></tr>")
        parts.append(
            "<table><tr><th>metric</th><th>value</th><th>CI95</th></tr>"
            + "".join(rowsout)
            + "</table>"
        )
        if "occupancy_top" in lvl:
            parts.append(_occupancy_svg([(int(s), float(mm)) for s, mm in lvl["occupancy_top"]]))

    parts.append("<h2>raw certificate</h2>")
    parts.append("<pre>" + html.escape(json.dumps(d, indent=2, sort_keys=True)) + "</pre>")
    parts.append("</body></html>")
    return "".join(parts)
