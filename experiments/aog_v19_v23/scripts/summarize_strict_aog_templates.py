#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.strict_aog.grammar import load_strict_aog


EDGE_TYPE_NAMES = {
    0: "anchor",
    1: "repeat",
    2: "generic",
    3: "structural",
    4: "semantic",
    5: "ATTACH_BOND",
    6: "HINGE",
    7: "BUTTING",
    8: "CONCENTRIC",
    9: "BAR_CIRCLE",
    10: "AXIAL_ALIGN",
    11: "BILATERAL_SYMMETRY",
    12: "CONTAINS",
    13: "SUPPORTS",
    14: "OCCLUDES",
}

SINGLETON_PARTS = {"body", "frame", "torso", "head", "tail", "seat", "beak", "mouth"}


def _part_key(name: str) -> str:
    return str(name).lower().replace("-", "_").replace("/", "_")


def template_summary_tables(grammar) -> tuple[pd.DataFrame, pd.DataFrame]:
    template_rows = []
    edge_rows = []
    edges = grammar.edges.detach().cpu().long()
    edge_types = grammar.edge_type.detach().cpu().long()
    edge_support = grammar.edge_support.detach().cpu().float()
    edge_info_gain = grammar.edge_info_gain.detach().cpu().float()
    for c, cls_name in enumerate(grammar.class_names):
        for a in range(int(grammar.num_templates)):
            if float(grammar.template_valid[c, a].item()) <= 0.5:
                continue
            slot_parts: list[str] = []
            slot_layout: list[str] = []
            valid_slots: list[int] = []
            for s in range(int(grammar.max_slots)):
                if float(grammar.slot_valid[c, a, s].item()) <= 0.5:
                    continue
                valid_slots.append(s)
                part_id = int(grammar.slot_part[c, a, s].item())
                part_name = grammar.part_names[part_id] if 0 <= part_id < len(grammar.part_names) else str(part_id)
                cx, cy, *_ = [float(x) for x in grammar.slot_geom_mean[c, a, s].detach().cpu().tolist()]
                slot_parts.append(part_name)
                slot_layout.append(f"s{s}:{part_name}@({cx:.2f},{cy:.2f})")
            edge_idx = ((edges[:, 0] == c) & (edges[:, 1] == a)).nonzero(as_tuple=False).flatten().tolist() if edges.numel() else []
            endpoint_slots = set()
            type_counts = Counter(int(edge_types[i].item()) for i in edge_idx)
            pair_type_counts: dict[tuple[int, int], int] = defaultdict(int)
            for i in edge_idx:
                _, _, si, sj = [int(x) for x in edges[i].tolist()]
                endpoint_slots.update([si, sj])
                pair_type_counts[tuple(sorted((si, sj)))] += 1
                pi = int(grammar.slot_part[c, a, si].item())
                pj = int(grammar.slot_part[c, a, sj].item())
                edge_rows.append({
                    "class_id": c,
                    "class_name": cls_name,
                    "template_id": a,
                    "slot_i": si,
                    "slot_j": sj,
                    "part_i": grammar.part_names[pi] if 0 <= pi < len(grammar.part_names) else str(pi),
                    "part_j": grammar.part_names[pj] if 0 <= pj < len(grammar.part_names) else str(pj),
                    "edge_type": int(edge_types[i].item()),
                    "edge_type_name": EDGE_TYPE_NAMES.get(int(edge_types[i].item()), str(int(edge_types[i].item()))),
                    "edge_support": float(edge_support[i].item()),
                    "edge_info_gain": float(edge_info_gain[i].item()),
                })
            part_counts = Counter(slot_parts)
            singleton_duplicates = {
                p: n for p, n in part_counts.items()
                if n > 1 and _part_key(p) in SINGLETON_PARTS
            }
            num_slots = len(valid_slots)
            num_edges = len(edge_idx)
            coverage = len(endpoint_slots & set(valid_slots)) / float(max(num_slots, 1))
            max_pair_bundle = max(pair_type_counts.values(), default=0)
            issues = []
            if num_slots < 2:
                issues.append("degenerate_one_slot")
            if num_edges == 0:
                issues.append("no_relation_edges")
            if num_edges > max(12, 2 * num_slots):
                issues.append("over_dense_edges")
            if coverage < 0.75 and num_slots >= 3:
                issues.append("low_edge_endpoint_coverage")
            if max_pair_bundle > 2:
                issues.append("multi_relation_bundle_per_pair")
            if singleton_duplicates:
                issues.append("singleton_duplicate:" + ",".join(f"{p}x{n}" for p, n in singleton_duplicates.items()))
            template_rows.append({
                "class_id": c,
                "class_name": cls_name,
                "template_id": a,
                "template_prior": float(grammar.template_prior[c, a].item()),
                "num_slots": num_slots,
                "num_edges": num_edges,
                "edge_endpoint_coverage": coverage,
                "max_pair_relation_bundle": max_pair_bundle,
                "slot_parts": ", ".join(slot_parts),
                "slot_layout": "; ".join(slot_layout),
                "edge_types": ", ".join(f"{EDGE_TYPE_NAMES.get(t, str(t))}:{n}" for t, n in sorted(type_counts.items())),
                "template_issues": ";".join(issues),
            })
    return pd.DataFrame(template_rows), pd.DataFrame(edge_rows)


def plot_class_templates(grammar, out_path: Path) -> None:
    num_classes = int(grammar.num_classes)
    num_templates = int(grammar.num_templates)
    fig, axes = plt.subplots(num_classes, num_templates, figsize=(4.8 * num_templates, 2.8 * num_classes), constrained_layout=True)
    if num_classes == 1:
        axes = np.array([axes])
    if num_templates == 1:
        axes = axes[:, None]
    edges = grammar.edges.detach().cpu().long()
    edge_types = grammar.edge_type.detach().cpu().long()
    palette = list(plt.cm.tab20(np.linspace(0, 1, 20)))
    for c, cls_name in enumerate(grammar.class_names):
        for a in range(num_templates):
            ax = axes[c, a]
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            if a == 0:
                ax.set_ylabel(cls_name, fontsize=10, rotation=0, ha="right", va="center")
            if float(grammar.template_valid[c, a].item()) <= 0.5:
                ax.set_title(f"T{a}: invalid", fontsize=9)
                ax.set_facecolor("#f4f4f5")
                continue
            valid_slots = [s for s in range(int(grammar.max_slots)) if float(grammar.slot_valid[c, a, s].item()) > 0.5]
            slot_pos = {}
            for s in valid_slots:
                cx, cy, *_ = [float(x) for x in grammar.slot_geom_mean[c, a, s].detach().cpu().tolist()]
                slot_pos[s] = (min(max(cx, 0.03), 0.97), min(max(cy, 0.03), 0.97))
            edge_idx = ((edges[:, 0] == c) & (edges[:, 1] == a)).nonzero(as_tuple=False).flatten().tolist() if edges.numel() else []
            pair_types = defaultdict(list)
            for i in edge_idx:
                _, _, si, sj = [int(x) for x in edges[i].tolist()]
                if si in slot_pos and sj in slot_pos:
                    pair_types[tuple(sorted((si, sj)))].append(EDGE_TYPE_NAMES.get(int(edge_types[i].item()), str(int(edge_types[i].item()))))
            for (si, sj), types in pair_types.items():
                x0, y0 = slot_pos[si]
                x1, y1 = slot_pos[sj]
                ax.plot([x0, x1], [y0, y1], color="#64748b", alpha=0.28, linewidth=0.7 + 0.25 * min(len(types), 6), zorder=1)
            for s in valid_slots:
                part_id = int(grammar.slot_part[c, a, s].item())
                part_name = grammar.part_names[part_id] if 0 <= part_id < len(grammar.part_names) else str(part_id)
                cx, cy, w, h, *_ = [float(x) for x in grammar.slot_geom_mean[c, a, s].detach().cpu().tolist()]
                cx = min(max(cx, 0.03), 0.97)
                cy = min(max(cy, 0.03), 0.97)
                w = min(max(w, 0.035), 0.28)
                h = min(max(h, 0.035), 0.28)
                color = palette[part_id % len(palette)]
                ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=color, edgecolor="black", linewidth=0.7, alpha=0.38, zorder=2))
                ax.text(cx, cy, f"{s}:{part_name}", fontsize=6.5, ha="center", va="center", bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=0.8), zorder=4)
            ax.set_title(f"T{a} prior={float(grammar.template_prior[c, a].item()):.2f} slots={len(valid_slots)} edges={len(edge_idx)}", fontsize=9)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grammar = load_strict_aog(args.grammar)
    summary, edges = template_summary_tables(grammar)
    summary_path = out_dir / f"{args.prefix}_class_template_summary.csv"
    edge_path = out_dir / f"{args.prefix}_class_template_edges.csv"
    fig_path = out_dir / f"{args.prefix}_class_templates.png"
    summary.to_csv(summary_path, index=False)
    edges.to_csv(edge_path, index=False)
    plot_class_templates(grammar, fig_path)
    issue_counts = summary["template_issues"].str.get_dummies(sep=";").sum().sort_values(ascending=False)
    issue_path = out_dir / f"{args.prefix}_class_template_issues.md"
    lines = [
        f"# Template Audit: {args.prefix}",
        "",
        f"- grammar: `{args.grammar}`",
        f"- templates: {len(summary)}",
        f"- mean slots: {summary['num_slots'].mean():.2f}",
        f"- mean edges: {summary['num_edges'].mean():.2f}",
        f"- mean endpoint coverage: {summary['edge_endpoint_coverage'].mean():.3f}",
        "",
        "## Issue Counts",
        "",
    ]
    if issue_counts.empty:
        lines.append("No template issues flagged.")
    else:
        for issue, count in issue_counts.items():
            if issue:
                lines.append(f"- {issue}: {int(count)}")
    lines.extend([
        "",
        "## Files",
        "",
        f"- summary: `{summary_path}`",
        f"- edges: `{edge_path}`",
        f"- figure: `{fig_path}`",
    ])
    issue_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"summary={summary_path}")
    print(f"edges={edge_path}")
    print(f"figure={fig_path}")
    print(f"issues={issue_path}")
    print(issue_counts.to_string())


if __name__ == "__main__":
    main()
