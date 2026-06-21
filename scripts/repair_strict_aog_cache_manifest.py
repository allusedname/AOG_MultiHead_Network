#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.strict_aog.terminals import _resolve_shard_path


def _repair_one(path: Path, *, dry_run: bool = False) -> None:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("kind") != "strict_aog_terminal_cache":
        raise ValueError(f"Expected strict_aog_terminal_cache manifest: {path}")
    if not payload.get("sharded"):
        print(f"[repair] {path}: not sharded; nothing to repair")
        return
    new_refs: list[str] = []
    missing: list[str] = []
    for ref in payload.get("shards", []):
        resolved = _resolve_shard_path(path, ref)
        if not resolved.exists():
            missing.append(str(ref))
            new_refs.append(str(ref))
            continue
        try:
            new_refs.append(str(resolved.resolve().relative_to(path.parent.resolve())))
        except ValueError:
            new_refs.append(str(resolved.resolve()))
    print(f"[repair] {path}: shards={len(new_refs)} missing={len(missing)}")
    if missing:
        print("[repair] unresolved shard refs:")
        for m in missing[:20]:
            print("  ", m)
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")
        raise FileNotFoundError(f"Could not resolve {len(missing)} shard refs in {path}")
    changed = list(payload.get("shards", [])) != new_refs
    if not changed:
        print(f"[repair] {path}: already canonical")
        return
    if dry_run:
        print(f"[repair] {path}: would rewrite shard refs; first old={payload['shards'][0]} new={new_refs[0]}")
        return
    payload = dict(payload)
    payload["shards"] = new_refs
    torch.save(payload, path)
    print(f"[repair] {path}: rewritten; first ref now {new_refs[0] if new_refs else '<none>'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Repair sharded strict-AOG terminal-cache manifests that contain cwd-relative shard paths.")
    p.add_argument("caches", nargs="+", help="train/val strict AOG terminal-cache manifest .pt files")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for cache in args.caches:
        _repair_one(Path(cache), dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
