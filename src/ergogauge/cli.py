"""Command-line interface for ergogauge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .api import DISCLAIMER, certify, certify_corpus
from .baseline_vendi import vendi_from_tokens
from .io import coerce_utterance, load_file


def _load_any(path: str) -> Any:
    return load_file(path)


def _cmd_certify(args: argparse.Namespace) -> int:
    obj = _load_any(args.path)
    is_corpus = isinstance(obj, list) and len(obj) > 0 and not isinstance(obj[0], int)
    cert = certify_corpus(obj, seed=args.seed) if is_corpus else certify(obj, seed=args.seed)
    out = cert.to_json()
    if args.output:
        Path(args.output).write_text(out + "\n", encoding="utf-8")
    else:
        print(out)
    return 0 if cert.aggregate.get("status") != "ABSTAIN" else 0


def _single_metric(args: argparse.Namespace, key: str) -> int:
    obj = _load_any(args.path)
    cert = certify(obj, seed=args.seed)
    lvl0 = cert.to_dict()["levels"][0]["metrics"]
    print(json.dumps(lvl0.get(key, {}), indent=2, sort_keys=True))
    return 0


def _cmd_gap(args: argparse.Namespace) -> int:
    return _single_metric(args, "spectral_gap")


def _cmd_kemeny(args: argparse.Namespace) -> int:
    return _single_metric(args, "kemeny")


def _cmd_cheeger(args: argparse.Namespace) -> int:
    return _single_metric(args, "cheeger")


def _cmd_vendi_compare(args: argparse.Namespace) -> int:
    obj = _load_any(args.path)
    per_level = coerce_utterance(
        obj if not (isinstance(obj, list) and obj and isinstance(obj[0], list)) else obj
    )
    pooled = [t for lvl in per_level for t in lvl]
    cert = certify(obj, seed=args.seed)
    gap = cert.to_dict()["levels"][0]["metrics"]["spectral_gap"]["value"]
    rev = list(reversed(pooled))
    print(
        json.dumps(
            {
                "note": "Vendi is order-independent (a sanity check, not a baseline to beat).",
                "vendi_original": vendi_from_tokens(pooled),
                "vendi_reversed": vendi_from_tokens(rev),
                "ergogauge_spectral_gap": gap,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    obj = _load_any(args.path)
    is_corpus = isinstance(obj, list) and len(obj) > 0 and not isinstance(obj[0], int)
    cert = certify_corpus(obj, seed=args.seed) if is_corpus else certify(obj, seed=args.seed)
    d = cert.to_dict()
    print("ergogauge doctor")
    print("  mode:", "corpus" if is_corpus else "single-utterance")
    print("  gate:", d["gate"]["status"], d["gate"].get("reasons", []))
    for lvl in d["levels"]:
        print(
            f"  level {lvl['level']}: flags={lvl['flags']} confidence={lvl['flag_confidence']} "
            f"reversible={lvl['reversible']} states={lvl['n_states_after_collapse']}"
        )
    print("  aggregate:", d["aggregate"])
    print("  note:", DISCLAIMER)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    obj = _load_any(args.path)
    is_corpus = isinstance(obj, list) and len(obj) > 0 and not isinstance(obj[0], int)
    cert = certify_corpus(obj, seed=args.seed) if is_corpus else certify(obj, seed=args.seed)
    out = args.output or "ergogauge_report.html"
    cert.to_html(out)
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ergogauge",
        description="Reference-free ergodicity certificate for codec-LM token streams.",
    )
    p.add_argument("--version", action="version", version=f"ergogauge {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("path", help="path to tokens .json/.npz")
        sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("certify", help="full certificate (JSON)")
    add_common(sp)
    sp.add_argument("-o", "--output", help="write JSON to file")
    sp.set_defaults(func=_cmd_certify)

    for name, fn, helptext in (
        ("gap", _cmd_gap, "spectral gap only"),
        ("kemeny", _cmd_kemeny, "Kemeny constant only"),
        ("cheeger", _cmd_cheeger, "Cheeger / Fiedler only"),
        ("vendi-compare", _cmd_vendi_compare, "order-sensitivity sanity check vs in-repo Vendi"),
        ("doctor", _cmd_doctor, "identifiability / ABSTAIN diagnosis"),
    ):
        sp = sub.add_parser(name, help=helptext)
        add_common(sp)
        sp.set_defaults(func=fn)

    sp = sub.add_parser("report", help="self-contained HTML report")
    add_common(sp)
    sp.add_argument("-o", "--output", help="output html path")
    sp.set_defaults(func=_cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    func = args.func
    result: int = func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
