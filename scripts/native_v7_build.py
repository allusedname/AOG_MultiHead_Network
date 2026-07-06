#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.abg_aog_v7.grammar_builder_v2 import build_native_grammar_from_terminal_cache_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a calibrated native ABG-HKG-AOG v7 grammar from a terminal cache.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-part-support", type=float, default=0.10)
    parser.add_argument("--score-tau", type=float, default=0.10)
    parser.add_argument("--max-templates-per-part", type=int, default=4)
    parser.add_argument("--min-template-support", type=int, default=3)
    parser.add_argument("--min-relation-support", type=int, default=6)
    args = parser.parse_args()
    grammar = build_native_grammar_from_terminal_cache_v2(
        args.cache,
        out=args.out,
        min_part_support=float(args.min_part_support),
        score_tau=float(args.score_tau),
        max_templates_per_part=int(args.max_templates_per_part),
        min_template_support=int(args.min_template_support),
        min_relation_support=int(args.min_relation_support),
    )
    print(f"saved native v7 grammar: {args.out}")
    print(f"nodes={len(grammar.nodes)} rules={len(grammar.rules)} relations={len(grammar.relations)}")


if __name__ == "__main__":
    main()
