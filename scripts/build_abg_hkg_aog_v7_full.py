#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.abg_aog import BlockPursuitConfig, EMBlockPursuitLearner, response_from_terminal_records
from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.strict_aog.terminals import load_terminal_cache


def _strip_flag(argv: list[str], flag: str) -> tuple[list[str], str | None]:
    out: list[str] = []
    value: str | None = None
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            value = argv[i + 1] if i + 1 < len(argv) else None
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out, value


def main() -> None:
    argv, block_out = _strip_flag(sys.argv[1:], "--block-bank-out")
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--cache", required=True)
    probe.add_argument("--out", required=True)
    probe.add_argument("--block-max-blocks", type=int, default=32)
    known, _ = probe.parse_known_args(argv)
    # Remove block-only flags before delegating to the fixed v6/v7 builder.
    clean: list[str] = []
    skip = False
    block_flags = {"--block-max-blocks"}
    for i, item in enumerate(argv):
        if skip:
            skip = False
            continue
        if item in block_flags:
            skip = True
            continue
        clean.append(item)
    sys.argv = [str(ROOT / "scripts" / "build_hier_pra_aog_v6.py")] + clean
    runpy.run_path(str(ROOT / "scripts" / "build_hier_pra_aog_v6.py"), run_name="__main__")

    payload = load_terminal_cache(known.cache, map_location="cpu", materialize=True)
    schema = RoleSchema.from_payload(payload["schema"])
    records = payload.get("records", [])
    response, names = response_from_terminal_records(records, num_parts=schema.num_parts)
    bank = EMBlockPursuitLearner(BlockPursuitConfig(max_blocks=int(known.block_max_blocks))).fit(response, feature_names=list(getattr(schema, "part_names", names)))
    path = Path(block_out) if block_out else Path(known.out).with_suffix(Path(known.out).suffix + ".block_bank.pt")
    bank.save(path)
    print(f"saved v7 full block-pursuit bank: {path}")
    print(bank.summary(limit=30))


if __name__ == "__main__":
    main()
