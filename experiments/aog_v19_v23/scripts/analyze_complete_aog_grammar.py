#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from collections import Counter, defaultdict

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.complete_aog.grammar import load_complete_aog


def main() -> None:
    p = argparse.ArgumentParser(description="Audit a Complete Neural Spatial AOG grammar.")
    p.add_argument("--grammar", required=True)
    p.add_argument("--out-dir", default="")
    args = p.parse_args()
    g = load_complete_aog(args.grammar)
    rows = []
    edge_rows = []
    for c, cname in enumerate(g.schema.obj_names):
        for a in range(g.num_templates):
            slot_ids = (g.slot_valid[c, a] > 0.5).nonzero(as_tuple=False).flatten().tolist()
            eidx = ((g.edges[:, 0] == c) & (g.edges[:, 1] == a)).nonzero(as_tuple=False).flatten().tolist() if g.edges.numel() else []
            parts = [g.schema.part_names[int(g.slot_part[c, a, s])] for s in slot_ids if int(g.slot_part[c, a, s]) >= 0]
            cnt = Counter(parts)
            rows.append({
                "class": cname,
                "class_idx": c,
                "template": a,
                "valid": float(g.template_valid[c, a].item()),
                "kind": g.template_kind[c][a] if c < len(g.template_kind) and a < len(g.template_kind[c]) else "template",
                "num_slots": len(slot_ids),
                "num_required_slots": int((g.slot_required[c, a] > 0.5).sum().item()),
                "num_edges": len(eidx),
                "slot_parts": ",".join(parts),
                "slot_part_counts": ";".join(f"{k}:{v}" for k, v in cnt.items()),
            })
            for e in eidx:
                si, sj = int(g.edges[e, 2]), int(g.edges[e, 3])
                pi = g.schema.part_names[int(g.slot_part[c, a, si])] if int(g.slot_part[c, a, si]) >= 0 else "?"
                pj = g.schema.part_names[int(g.slot_part[c, a, sj])] if int(g.slot_part[c, a, sj]) >= 0 else "?"
                edge_rows.append({
                    "class": cname,
                    "class_idx": c,
                    "template": a,
                    "edge": int(e),
                    "slot_i": si,
                    "slot_j": sj,
                    "part_i": pi,
                    "part_j": pj,
                    "type": g.edge_type[e] if e < len(g.edge_type) else "relation",
                    "support": float(g.edge_support[e].item()),
                    "required": float(g.edge_required[e].item()) if g.edge_required.numel() else 0.0,
                    "mean_rel_ll_scale_var_mean": float(g.edge_rel_var[e].mean().item()) if g.edge_rel_var.numel() else float('nan'),
                })
    df = pd.DataFrame(rows)
    edf = pd.DataFrame(edge_rows)
    print("[grammar-audit]", args.grammar)
    print(f"classes={g.num_classes} templates/class={g.num_templates} max_slots={g.max_slots}")
    print(f"valid_templates={int(g.template_valid.sum().item())} valid_slots={int(g.slot_valid.sum().item())} edges={int(g.edges.shape[0])}")
    print("templates edges summary:")
    print(df[["class", "template", "valid", "num_slots", "num_required_slots", "num_edges", "slot_part_counts"]].to_string(index=False, max_colwidth=100))
    print("\nedge count distribution:")
    print(df["num_edges"].describe().to_string())
    if not edf.empty:
        print("\nedge type counts:")
        print(edf["type"].value_counts().to_string())
        print("\nedge part-pair counts:")
        print((edf["part_i"] + "-" + edf["part_j"]).value_counts().head(40).to_string())
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "grammar_templates.csv", index=False)
        edf.to_csv(out / "grammar_edges.csv", index=False)
        print(f"saved CSVs to {out}")


if __name__ == "__main__":
    main()
