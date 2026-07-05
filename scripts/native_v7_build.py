#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.abg_aog_v7.grammar_builder import build_native_grammar_from_terminal_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a native ABG-HKG-AOG v7 grammar from a terminal cache.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-part-support", type=float, default=0.10)
    parser.add_argument("--score-tau", type=float, default=0.10)
    args = parser.parse_args()
    grammar = build_native_grammar_from_terminal_cache(
        args.cache,
        out=args.out,
        min_part_support=float(args.min_part_support),
        score_tau=float(args.score_tau),
    )
    print(f"saved native v7 grammar: {args.out}")
    print(f"nodes={len(grammar.nodes)} rules={len(grammar.rules)} relations={len(grammar.relations)}")


if __name__ == "__main__":
    main()
