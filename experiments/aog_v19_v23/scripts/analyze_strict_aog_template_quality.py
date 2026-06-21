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


def part_key(name: str) -> str:
    return str(name).lower().replace("-", "_").replace("/", "_")


def entropy_effective_count(fracs: np.ndarray) -> float:
    fracs = np.asarray(fracs, dtype=float)
    fracs = fracs[fracs > 0]
    if fracs.size == 0:
        return 0.0
    return float(np.exp(-(fracs * np.log(fracs)).sum()))


def summarize_templates(grammar, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = grammar.edges.detach().cpu().long()
    edge_types = grammar.edge_type.detach().cpu().long()
    template_rows = []

    true_total = predictions.groupby("true").size().to_dict()
    true_usage = predictions.groupby(["true", "true_template"]).size().to_dict()
    true_correct = predictions[predictions["correct"].astype(bool)].groupby(["true", "true_template"]).size().to_dict()
    pred_usage = predictions.groupby(["pred", "pred_template"]).size().to_dict()

    for c, cls_name in enumerate(grammar.class_names):
        class_n = int(true_total.get(c, 0))
        for a in range(int(grammar.num_templates)):
            if float(grammar.template_valid[c, a].item()) <= 0.5:
                continue
            valid_slots = [s for s in range(int(grammar.max_slots)) if float(grammar.slot_valid[c, a, s].item()) > 0.5]
            slot_parts = []
            optional_slots = 0
            required_slots = 0
            position_stds = []
            size_stds = []
            area_stds = []
            for s in valid_slots:
                part_id = int(grammar.slot_part[c, a, s].item())
                part_name = grammar.part_names[part_id] if 0 <= part_id < len(grammar.part_names) else str(part_id)
                slot_parts.append(part_name)
                if float(grammar.slot_required[c, a, s].item()) > 0.5:
                    required_slots += 1
                else:
                    optional_slots += 1
                geom_var = grammar.slot_geom_var[c, a, s].detach().cpu().float().clamp_min(0)
                position_stds.append(float(torch.sqrt(geom_var[:2]).mean().item()))
                size_stds.append(float(torch.sqrt(geom_var[2:4]).mean().item()))
                if geom_var.numel() > 4:
                    area_stds.append(float(torch.sqrt(geom_var[4]).item()))

            edge_idx = ((edges[:, 0] == c) & (edges[:, 1] == a)).nonzero(as_tuple=False).flatten().tolist() if edges.numel() else []
            endpoint_degree = Counter()
            endpoint_slots = set()
            edge_type_counts = Counter()
            pair_type_counts: dict[tuple[int, int], int] = defaultdict(int)
            for i in edge_idx:
                _, _, si, sj = [int(x) for x in edges[i].tolist()]
                endpoint_degree[si] += 1
                endpoint_degree[sj] += 1
                endpoint_slots.update([si, sj])
                edge_type_counts[int(edge_types[i].item())] += 1
                pair_type_counts[tuple(sorted((si, sj)))] += 1

            part_counts = Counter(slot_parts)
            singleton_duplicates = {
                p: n for p, n in part_counts.items()
                if n > 1 and part_key(p) in SINGLETON_PARTS
            }
            num_slots = len(valid_slots)
            max_edges = num_slots * (num_slots - 1) / 2 if num_slots > 1 else 1.0
            num_edges = len(edge_idx)
            usage_n = int(true_usage.get((c, a), 0))
            correct_n = int(true_correct.get((c, a), 0))
            pred_n = int(pred_usage.get((c, a), 0))
            usage_frac = usage_n / class_n if class_n else 0.0
            acc_when_true_template = correct_n / usage_n if usage_n else np.nan
            max_degree = max(endpoint_degree.values(), default=0)
            endpoint_coverage = len(endpoint_slots & set(valid_slots)) / max(num_slots, 1)
            mean_degree = (2 * num_edges / num_slots) if num_slots else 0.0
            duplicate_slot_count = sum(n - 1 for n in part_counts.values() if n > 1)

            flags = []
            if usage_frac < 0.05:
                flags.append("low_usage")
            if num_slots > 8:
                flags.append("many_slots")
            if num_edges > max(12, 1.75 * num_slots):
                flags.append("dense_edges")
            if max_degree > 3:
                flags.append("high_slot_degree")
            if singleton_duplicates:
                flags.append("singleton_duplicate")
            if usage_frac > 0.35 and num_slots > 8:
                flags.append("high_use_complex_template")
            if usage_frac > 0.35 and float(np.mean(position_stds or [0.0])) < 0.04:
                flags.append("high_use_rigid_layout")

            template_rows.append({
                "class_id": c,
                "class_name": cls_name,
                "template_id": a,
                "template_prior": float(grammar.template_prior[c, a].item()),
                "true_usage_n": usage_n,
                "true_usage_frac": usage_frac,
                "pred_usage_n": pred_n,
                "accuracy_when_true_template": acc_when_true_template,
                "num_slots": num_slots,
                "num_unique_parts": len(part_counts),
                "duplicate_slot_count": duplicate_slot_count,
                "required_slots": required_slots,
                "optional_slots": optional_slots,
                "num_edges": num_edges,
                "edge_density": num_edges / max_edges,
                "mean_degree": mean_degree,
                "max_degree": max_degree,
                "edge_endpoint_coverage": endpoint_coverage,
                "max_pair_relation_bundle": max(pair_type_counts.values(), default=0),
                "position_std_mean": float(np.mean(position_stds or [0.0])),
                "size_std_mean": float(np.mean(size_stds or [0.0])),
                "area_std_mean": float(np.mean(area_stds or [0.0])),
                "slot_parts": ", ".join(slot_parts),
                "edge_types": ", ".join(f"{EDGE_TYPE_NAMES.get(t, str(t))}:{n}" for t, n in sorted(edge_type_counts.items())),
                "quality_flags": ";".join(flags),
            })

    template_df = pd.DataFrame(template_rows)
    class_rows = []
    for c, cls_name in enumerate(grammar.class_names):
        sub = template_df[template_df["class_id"] == c].copy()
        usage = sub["true_usage_frac"].to_numpy(float)
        prior = sub["template_prior"].to_numpy(float)
        prior = prior / prior.sum() if prior.sum() > 0 else np.zeros_like(prior)
        class_rows.append({
            "class_id": c,
            "class_name": cls_name,
            "n_val": int(true_total.get(c, 0)),
            "accuracy": float(predictions[predictions["true"] == c]["correct"].mean()),
            "effective_templates_used": entropy_effective_count(usage),
            "effective_templates_prior": entropy_effective_count(prior),
            "max_usage_frac": float(usage.max()) if usage.size else 0.0,
            "min_usage_frac": float(usage.min()) if usage.size else 0.0,
            "usage_prior_l1": float(np.abs(usage - prior).sum()) if usage.size == prior.size else np.nan,
            "mean_slots": float(sub["num_slots"].mean()),
            "mean_edges": float(sub["num_edges"].mean()),
            "mean_position_std": float(sub["position_std_mean"].mean()),
            "num_low_usage_templates": int((sub["true_usage_frac"] < 0.05).sum()),
            "num_high_use_complex_templates": int(sub["quality_flags"].str.contains("high_use_complex_template", regex=False).sum()),
            "num_high_degree_templates": int(sub["quality_flags"].str.contains("high_slot_degree", regex=False).sum()),
        })
    class_df = pd.DataFrame(class_rows)
    return template_df, class_df


def plot_dashboard(template_df: pd.DataFrame, class_df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    class_order = list(class_df["class_name"])
    pivot_usage = template_df.pivot(index="class_name", columns="template_id", values="true_usage_frac").reindex(class_order)
    pivot_acc = template_df.pivot(index="class_name", columns="template_id", values="accuracy_when_true_template").reindex(class_order)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    im = axes[0, 0].imshow(pivot_usage.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=max(0.01, float(pivot_usage.max().max())))
    axes[0, 0].set_title("True-class template usage fraction")
    axes[0, 0].set_yticks(range(len(class_order)), class_order)
    axes[0, 0].set_xticks(range(pivot_usage.shape[1]), [f"T{c}" for c in pivot_usage.columns])
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im2 = axes[0, 1].imshow(pivot_acc.fillna(0).to_numpy(), aspect="auto", cmap="RdYlGn", vmin=0.8, vmax=1.0)
    axes[0, 1].set_title("Accuracy when true class uses template")
    axes[0, 1].set_yticks(range(len(class_order)), class_order)
    axes[0, 1].set_xticks(range(pivot_acc.shape[1]), [f"T{c}" for c in pivot_acc.columns])
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    x = np.arange(len(class_df))
    axes[1, 0].bar(x - 0.18, class_df["effective_templates_used"], width=0.36, label="used")
    axes[1, 0].bar(x + 0.18, class_df["effective_templates_prior"], width=0.36, label="prior")
    axes[1, 0].set_title("Effective number of templates per class")
    axes[1, 0].set_xticks(x, class_df["class_name"], rotation=45, ha="right")
    axes[1, 0].set_ylim(0, max(3.2, float(class_df[["effective_templates_used", "effective_templates_prior"]].max().max()) + 0.2))
    axes[1, 0].legend()

    sc = axes[1, 1].scatter(
        template_df["num_slots"],
        template_df["true_usage_frac"],
        s=35 + 45 * template_df["num_edges"],
        c=template_df["position_std_mean"],
        cmap="viridis",
        alpha=0.78,
        edgecolor="black",
        linewidth=0.4,
    )
    for _, row in template_df.iterrows():
        if row["true_usage_frac"] >= 0.25 or row["num_slots"] >= 9:
            axes[1, 1].text(row["num_slots"] + 0.05, row["true_usage_frac"], f"{row['class_name']} T{int(row['template_id'])}", fontsize=7)
    axes[1, 1].set_xlabel("Number of slots (simplicity)")
    axes[1, 1].set_ylabel("Validation usage fraction")
    axes[1, 1].set_title("Template simplicity vs representativeness")
    fig.colorbar(sc, ax=axes[1, 1], fraction=0.046, label="mean slot position std")

    fig.savefig(out_dir / f"{prefix}_template_quality_dashboard.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = template_df["accuracy_when_true_template"].fillna(0.0)
    sc = ax.scatter(
        template_df["num_edges"] / template_df["num_slots"].clip(lower=1),
        template_df["position_std_mean"],
        s=80 + 550 * template_df["true_usage_frac"],
        c=colors,
        cmap="RdYlGn",
        vmin=0.8,
        vmax=1.0,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )
    for _, row in template_df.iterrows():
        if row["true_usage_frac"] >= 0.20 or row["quality_flags"]:
            ax.text(row["num_edges"] / max(row["num_slots"], 1) + 0.02, row["position_std_mean"], f"{row['class_name']} T{int(row['template_id'])}", fontsize=7)
    ax.set_xlabel("Edges per slot (lower is simpler)")
    ax.set_ylabel("Mean slot position std (higher covers more layout variation)")
    ax.set_title("Template simplicity/flexibility/accuracy")
    fig.colorbar(sc, ax=ax, label="accuracy when true template")
    fig.savefig(out_dir / f"{prefix}_simplicity_flexibility_scatter.png", dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "(empty)"
    rows: list[list[str]] = []
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                if np.isnan(float(val)):
                    vals.append("")
                else:
                    vals.append(format(float(val), floatfmt))
            else:
                vals.append(str(val))
        rows.append(vals)
    header = [str(c) for c in df.columns]
    widths = [len(h) for h in header]
    for vals in rows:
        widths = [max(w, len(v)) for w, v in zip(widths, vals)]
    def fmt(vals: list[str]) -> str:
        return "| " + " | ".join(v.ljust(w) for v, w in zip(vals, widths)) + " |"
    lines = [fmt(header), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(fmt(vals) for vals in rows)
    return "\n".join(lines)


def write_report(template_df: pd.DataFrame, class_df: pd.DataFrame, out_path: Path, prefix: str) -> None:
    lines = [
        f"# Template Quality Diagnostic: {prefix}",
        "",
        "This diagnostic asks whether AOG templates are both simple and representative.",
        "",
        "## What Is Measured",
        "",
        "- usage: how often each template is selected for validation examples of its true class",
        "- simplicity: number of slots, number of edges, degree, density, duplicate slots",
        "- representativeness: effective number of templates used per class and slot geometry variance",
        "- risk flags: low-use templates, high-use complex templates, high-degree templates, rigid high-use layouts",
        "",
        "## Class-Level Summary",
        "",
        markdown_table(class_df, floatfmt=".3f"),
        "",
        "## Flagged Templates",
        "",
    ]
    flagged = template_df[template_df["quality_flags"].fillna("") != ""].copy()
    if flagged.empty:
        lines.append("No templates flagged by the usage/simplicity diagnostic.")
    else:
        cols = [
            "class_name", "template_id", "true_usage_frac", "accuracy_when_true_template",
            "num_slots", "num_edges", "max_degree", "position_std_mean", "quality_flags",
        ]
        lines.append(markdown_table(flagged[cols], floatfmt=".3f"))
    lines.extend([
        "",
        "## Most-Used Templates",
        "",
        markdown_table(template_df.sort_values(["true_usage_frac", "num_slots"], ascending=[False, False]).head(12)[
            [
                "class_name", "template_id", "true_usage_frac", "accuracy_when_true_template",
                "num_slots", "num_edges", "position_std_mean", "slot_parts",
            ]
        ], floatfmt=".3f"),
        "",
        "## Interpretation Guide",
        "",
        "- A good template set should not use all samples through one complex template unless the class truly has one stable layout.",
        "- High-use templates with many slots are candidates for simplification.",
        "- Low-use templates are candidates for merging, pruning, or rebuilding around a missing layout mode.",
        "- Very low slot position variance in a high-use template can mean the template is too rigid.",
        "- Very high variance can mean the slot is too vague; inspect the corresponding class template image.",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grammar = load_strict_aog(args.grammar)
    predictions = pd.read_csv(args.predictions)

    template_df, class_df = summarize_templates(grammar, predictions)
    template_path = out_dir / f"{args.prefix}_template_quality_by_template.csv"
    class_path = out_dir / f"{args.prefix}_template_quality_by_class.csv"
    report_path = out_dir / f"{args.prefix}_template_quality_report.md"
    template_df.to_csv(template_path, index=False)
    class_df.to_csv(class_path, index=False)
    plot_dashboard(template_df, class_df, out_dir, args.prefix)
    write_report(template_df, class_df, report_path, args.prefix)
    print(f"template={template_path}")
    print(f"class={class_path}")
    print(f"report={report_path}")
    print(f"dashboard={out_dir / f'{args.prefix}_template_quality_dashboard.png'}")
    print(f"scatter={out_dir / f'{args.prefix}_simplicity_flexibility_scatter.png'}")


if __name__ == "__main__":
    main()
