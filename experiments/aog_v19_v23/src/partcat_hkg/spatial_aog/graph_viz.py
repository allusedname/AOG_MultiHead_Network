from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader

from .grammar import SpatialAOGGrammar, load_spatial_aog
from .parser import ParserConfig, SpatialAOGParser
from .terminals import AOGTerminalDataset, TerminalRecord, collate_terminal_batch


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _safe_name(x: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(x))


def _part_name(grammar: SpatialAOGGrammar, k: int) -> str:
    if 0 <= int(k) < len(grammar.schema.part_names):
        return grammar.schema.part_names[int(k)]
    return f"part{int(k)}"


def _class_name(grammar: SpatialAOGGrammar, c: int) -> str:
    if 0 <= int(c) < len(grammar.schema.obj_names):
        return grammar.schema.obj_names[int(c)]
    return f"class{int(c)}"


def _resolve_class(grammar: SpatialAOGGrammar, class_name: str | None = None, class_idx: int | None = None) -> int:
    if class_idx is not None and int(class_idx) >= 0:
        c = int(class_idx)
        if c < 0 or c >= grammar.num_classes:
            raise ValueError(f"class_idx={c} out of range [0,{grammar.num_classes})")
        return c
    if class_name:
        names = {n: i for i, n in enumerate(grammar.schema.obj_names)}
        if class_name not in names:
            raise ValueError(f"Unknown class_name={class_name!r}. Available: {grammar.schema.obj_names}")
        return int(names[class_name])
    return 0


def _write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _render_dot(dot_path: Path, *, formats: tuple[str, ...] = ("png", "svg")) -> list[Path]:
    """Render a DOT file if graphviz is available; otherwise do nothing."""
    out: list[Path] = []
    exe = shutil.which("dot")
    if exe is None:
        return out
    for fmt in formats:
        target = dot_path.with_suffix(f".{fmt}")
        try:
            subprocess.run([exe, f"-T{fmt}", str(dot_path), "-o", str(target)], check=True, timeout=60)
            out.append(target)
        except Exception:
            # Keep DOT as the guaranteed artifact; rendering is optional.
            continue
    return out


def _csv_write(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _to_image(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        return None
    img = x.detach().float().cpu()
    # New caches store image_raw in [0,1]. Older caches accidentally stored the
    # ImageNet-normalized network input. Detect that case and de-normalize so the
    # overlay looks like a normal image instead of a high-contrast blue/yellow map.
    if img.ndim == 3 and img.shape[0] == 3:
        if float(img.min()) < -0.05 or float(img.max()) > 1.05:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype).view(3, 1, 1)
            img = img * std + mean
        img = img.permute(1, 2, 0)
    return img.clamp(0, 1)


def _color_for_index(i: int) -> tuple[float, float, float, float]:
    cmap = plt.get_cmap("tab20")
    return cmap(int(i) % 20)


# -----------------------------------------------------------------------------
# Global and class-level grammar visualization
# -----------------------------------------------------------------------------


def grammar_template_rows(grammar: SpatialAOGGrammar) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edges = grammar.edges.detach().cpu()
    for c in range(grammar.num_classes):
        for a in range(grammar.num_templates):
            slot_ids = (grammar.slot_valid[c, a].detach().cpu() > 0).nonzero(as_tuple=False).flatten().tolist()
            edge_mask = (edges[:, 0] == c) & (edges[:, 1] == a) if edges.numel() else torch.zeros(0, dtype=torch.bool)
            part_counts: dict[str, int] = {}
            for s in slot_ids:
                k = int(grammar.slot_part[c, a, s].detach().cpu().item())
                pn = _part_name(grammar, k)
                part_counts[pn] = part_counts.get(pn, 0) + 1
            rows.append({
                "class_idx": int(c),
                "class": _class_name(grammar, c),
                "template": int(a),
                "valid": float(grammar.template_valid[c, a].detach().cpu().item()),
                "prior": float(grammar.template_prior[c, a].detach().cpu().item()),
                "num_slots": int(len(slot_ids)),
                "num_required_slots": int(((grammar.slot_required[c, a].detach().cpu() > 0) & (grammar.slot_valid[c, a].detach().cpu() > 0)).sum().item()),
                "num_edges": int(edge_mask.sum().item()) if edges.numel() else 0,
                "part_counts": json.dumps(part_counts, sort_keys=True),
            })
    return rows


def write_global_aog_dot(grammar: SpatialAOGGrammar, out_dir: str | Path, *, render: bool = True) -> dict[str, Path | list[Path]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "digraph AOG_Global {",
        "  rankdir=LR;",
        "  graph [fontsize=10, labelloc=t, label=\"Global Spatial AOG: S -> class Or-nodes -> template And-nodes\"];",
        "  node [shape=box, style=rounded, fontsize=9];",
        "  S [label=\"S\\nroot\", shape=oval, style=filled, fillcolor=\"#eeeeee\"];",
    ]
    for c in range(grammar.num_classes):
        cn = _class_name(grammar, c)
        lines.append(f'  C{c} [label="O_{c}: {cn}\\nclass prior={float(grammar.class_prior[c].detach().cpu()):.3f}", shape=oval, style=filled, fillcolor="#dbeafe"];')
        lines.append(f"  S -> C{c};")
        for a in range(grammar.num_templates):
            if float(grammar.template_valid[c, a].detach().cpu().item()) <= 0:
                continue
            n_slots = int((grammar.slot_valid[c, a].detach().cpu() > 0).sum().item())
            edge_mask = (grammar.edges[:, 0] == c) & (grammar.edges[:, 1] == a) if grammar.edges.numel() else torch.zeros(0, dtype=torch.bool)
            n_edges = int(edge_mask.sum().item()) if grammar.edges.numel() else 0
            label = f"A_{{{c},{a}}}: template {a}\\nprior={float(grammar.template_prior[c,a].detach().cpu()):.3f} slots={n_slots} edges={n_edges}"
            lines.append(f'  T{c}_{a} [label="{label}", shape=box, style=filled, fillcolor="#dcfce7"];')
            lines.append(f"  C{c} -> T{c}_{a} [label=\"switch\"];")
    lines.append("}")
    dot_path = _write_text(out_dir / "global_aog_overview.dot", "\n".join(lines))
    rendered = _render_dot(dot_path) if render else []
    return {"dot": dot_path, "rendered": rendered}


def plot_global_summary(grammar: SpatialAOGGrammar, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = grammar_template_rows(grammar)
    rows_csv = _csv_write(out_dir / "global_template_summary.csv", rows)

    # Template slot/edge count heatmaps.
    valid_rows = [r for r in rows if r["valid"] > 0]
    labels = [f"{r['class']}:{r['template']}" for r in valid_rows]
    mat = torch.tensor([[r["num_slots"], r["num_edges"], r["num_required_slots"]] for r in valid_rows], dtype=torch.float32)
    fig_h = max(4.0, min(18.0, 0.28 * max(1, len(labels))))
    fig, ax = plt.subplots(figsize=(7, fig_h))
    im = ax.imshow(mat.numpy(), aspect="auto")
    ax.set_xticks([0, 1, 2], ["slots", "edges", "required slots"])
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_title("Valid template structure counts")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(int(mat[i, j].item())), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.65)
    fig.tight_layout()
    heat_path = out_dir / "global_template_slot_edge_counts.png"
    fig.savefig(heat_path, dpi=180)
    plt.close(fig)

    # Global part-slot counts.
    part_counts = torch.zeros(grammar.num_parts, dtype=torch.float32)
    for c in range(grammar.num_classes):
        for a in range(grammar.num_templates):
            for s in (grammar.slot_valid[c, a].detach().cpu() > 0).nonzero(as_tuple=False).flatten().tolist():
                k = int(grammar.slot_part[c, a, s].detach().cpu().item())
                if 0 <= k < grammar.num_parts:
                    part_counts[k] += 1
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * grammar.num_parts), 4.5))
    ax.bar(range(grammar.num_parts), part_counts.numpy())
    ax.set_xticks(range(grammar.num_parts), grammar.schema.part_names, rotation=60, ha="right")
    ax.set_ylabel("slot count across all templates")
    ax.set_title("Global terminal slot vocabulary usage")
    fig.tight_layout()
    part_path = out_dir / "global_part_slot_counts.png"
    fig.savefig(part_path, dpi=180)
    plt.close(fig)

    # Global part-pair edge matrix.
    edge_mat = torch.zeros(grammar.num_parts, grammar.num_parts, dtype=torch.float32)
    for e, row in enumerate(grammar.edges.detach().cpu().tolist()):
        c, a, si, sj = [int(x) for x in row]
        ki = int(grammar.slot_part[c, a, si].detach().cpu().item())
        kj = int(grammar.slot_part[c, a, sj].detach().cpu().item())
        if 0 <= ki < grammar.num_parts and 0 <= kj < grammar.num_parts:
            edge_mat[ki, kj] += 1
            if ki != kj:
                edge_mat[kj, ki] += 1
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * grammar.num_parts), max(5, 0.45 * grammar.num_parts)))
    im = ax.imshow(edge_mat.numpy())
    ax.set_xticks(range(grammar.num_parts), grammar.schema.part_names, rotation=60, ha="right")
    ax.set_yticks(range(grammar.num_parts), grammar.schema.part_names)
    ax.set_title("Global part-pair relation counts")
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    edge_path = out_dir / "global_part_edge_matrix.png"
    fig.savefig(edge_path, dpi=180)
    plt.close(fig)

    return {"template_csv": rows_csv, "template_counts": heat_path, "part_slots": part_path, "part_edge_matrix": edge_path}


def write_category_aog_dot(grammar: SpatialAOGGrammar, c: int, out_dir: str | Path, *, render: bool = True) -> dict[str, Path | list[Path]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cname = _class_name(grammar, c)
    lines = [
        f"digraph AOG_Category_{_safe_name(cname)} {{",
        "  rankdir=LR;",
        f"  graph [fontsize=10, labelloc=t, label=\"Category AOG: {cname}\"];",
        "  node [fontsize=9];",
        f'  C [label="O: {cname}\\nOr-node", shape=oval, style=filled, fillcolor="#dbeafe"];',
    ]
    for a in range(grammar.num_templates):
        if float(grammar.template_valid[c, a].detach().cpu().item()) <= 0:
            continue
        lines.append(f"  subgraph cluster_T{a} {{")
        lines.append(f'    label="A_{{{a}}}: template And-node prior={float(grammar.template_prior[c,a].detach().cpu()):.3f}";')
        lines.append("    color=\"#86efac\";")
        lines.append(f'    T{a} [label="template {a}", shape=box, style=filled, fillcolor="#dcfce7"];')
        lines.append(f"    C -> T{a} [label=\"switch\", ltail=cluster_T{a}];")
        slot_ids = (grammar.slot_valid[c, a].detach().cpu() > 0).nonzero(as_tuple=False).flatten().tolist()
        for s in slot_ids:
            k = int(grammar.slot_part[c, a, s].detach().cpu().item())
            req = float(grammar.slot_required[c, a, s].detach().cpu().item())
            sup = float(grammar.slot_support[c, a, s].detach().cpu().item())
            fill = "#fee2e2" if req > 0 else "#fef9c3"
            lines.append(f'    S{a}_{s} [label="s{s}: {_part_name(grammar,k)}\\nsupp={sup:.2f} req={req:.0f}", shape=box, style=filled, fillcolor="{fill}"];')
            lines.append(f"    T{a} -> S{a}_{s} [style=dotted];")
        for e, row in enumerate(grammar.edges.detach().cpu().tolist()):
            ce, ae, si, sj = [int(x) for x in row]
            if ce != c or ae != a:
                continue
            etype = grammar.edge_type_names[e] if e < len(grammar.edge_type_names) else "rel"
            support = float(grammar.edge_support[e].detach().cpu().item())
            req = float(grammar.edge_required[e].detach().cpu().item())
            color = "#dc2626" if req > 0 else "#2563eb"
            lines.append(f'    S{a}_{si} -> S{a}_{sj} [dir=none, color="{color}", penwidth=1.8, label="{etype} {support:.2f}"];')
        lines.append("  }")
    lines.append("}")
    dot_path = _write_text(out_dir / f"category_{_safe_name(cname)}.dot", "\n".join(lines))
    rendered = _render_dot(dot_path) if render else []
    return {"dot": dot_path, "rendered": rendered}


def plot_category_templates(grammar: SpatialAOGGrammar, c: int, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cname = _class_name(grammar, c)
    template_paths: dict[str, Path] = {}
    for a in range(grammar.num_templates):
        if float(grammar.template_valid[c, a].detach().cpu().item()) <= 0:
            continue
        slot_ids = (grammar.slot_valid[c, a].detach().cpu() > 0).nonzero(as_tuple=False).flatten().tolist()
        if not slot_ids:
            continue
        # Use slot mean geometry when available; otherwise circle layout.
        xy = {}
        for idx, s in enumerate(slot_ids):
            gx = float(grammar.slot_geom_mean[c, a, s, 0].detach().cpu().item())
            gy = float(grammar.slot_geom_mean[c, a, s, 1].detach().cpu().item())
            if not math.isfinite(gx) or not math.isfinite(gy) or (gx == 0 and gy == 0):
                theta = 2 * math.pi * idx / max(1, len(slot_ids))
                gx, gy = 0.5 + 0.35 * math.cos(theta), 0.5 + 0.35 * math.sin(theta)
            xy[s] = (gx, gy)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(f"{cname}: template {a} slots and relation edges")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(1.05, -0.05)
        ax.set_xlabel("normalized x"); ax.set_ylabel("normalized y")
        ax.grid(True, alpha=0.2)
        # edges first
        for e, row in enumerate(grammar.edges.detach().cpu().tolist()):
            ce, ae, si, sj = [int(x) for x in row]
            if ce != c or ae != a or si not in xy or sj not in xy:
                continue
            x0, y0 = xy[si]; x1, y1 = xy[sj]
            req = float(grammar.edge_required[e].detach().cpu().item())
            color = "tab:red" if req > 0 else "tab:blue"
            ax.plot([x0, x1], [y0, y1], "--", linewidth=1.5, color=color, alpha=0.8)
            ax.text((x0 + x1)/2, (y0 + y1)/2, f"e{e}", fontsize=7, alpha=0.8)
        for j, s in enumerate(slot_ids):
            k = int(grammar.slot_part[c, a, s].detach().cpu().item())
            x, y = xy[s]
            req = float(grammar.slot_required[c, a, s].detach().cpu().item())
            ax.scatter([x], [y], s=220, color=_color_for_index(k), edgecolor="black", linewidth=1.0, alpha=0.9)
            ax.text(x, y, f"s{s}\n{_part_name(grammar,k)}", ha="center", va="center", fontsize=8, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
            if req > 0:
                ax.add_patch(patches.Circle((x, y), radius=0.035, fill=False, edgecolor="red", linewidth=1.5))
        fig.tight_layout()
        path = out_dir / f"category_{_safe_name(cname)}_template_{a}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        template_paths[f"template_{a}"] = path
    return template_paths


# -----------------------------------------------------------------------------
# Image-sample and parse visualization
# -----------------------------------------------------------------------------


def _record_by_position_or_index(ds: AOGTerminalDataset, *, sample_pos: int | None = None, sample_index: int | None = None) -> tuple[int, TerminalRecord]:
    if sample_index is not None and int(sample_index) >= 0:
        target = int(sample_index)
        for pos, rec in enumerate(ds.records):
            if int(rec.index) == target:
                return pos, rec
        raise ValueError(f"sample_index={target} not found in cache. Use --sample-pos for positional indexing.")
    pos = int(sample_pos or 0)
    if pos < 0 or pos >= len(ds):
        raise ValueError(f"sample_pos={pos} out of range [0,{len(ds)})")
    return pos, ds[pos]


def _single_batch_from_record(rec: TerminalRecord, *, include_masks: bool = True, include_images: bool = True) -> dict[str, Any]:
    return collate_terminal_batch([rec], include_masks=include_masks, include_images=include_images)


def plot_terminal_proposals(grammar: SpatialAOGGrammar, batch: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = _to_image(batch.get("image", None)[0] if "image" in batch else None)
    geom = batch["terminal_geom"][0].detach().float().cpu()
    part = batch["terminal_part"][0].detach().long().cpu()
    score = batch["terminal_score"][0].detach().float().cpu()
    valid = batch["terminal_valid"][0].detach().bool().cpu()
    masks = batch.get("terminal_mask", None)
    masks0 = masks[0].detach().float().cpu() if torch.is_tensor(masks) else None
    if img is None:
        img = torch.ones(512, 512, 3)
    h, w = int(img.shape[0]), int(img.shape[1])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title("Stage-1 terminal proposals used as AOG terminals")
    for n in valid.nonzero(as_tuple=False).flatten().tolist():
        k = int(part[n].item())
        color = _color_for_index(k)
        if masks0 is not None:
            m = torch.nn.functional.interpolate(masks0[n].view(1, 1, *masks0[n].shape), size=(h, w), mode="nearest")[0, 0].numpy()
            ax.contour(m, levels=[0.5], colors=[color], linewidths=1.2)
        x, y = float(geom[n, 0].item()) * w, float(geom[n, 1].item()) * h
        ax.scatter([x], [y], s=35, color=color, edgecolor="white", linewidth=0.6)
        ax.text(x, y, f"t{n}:{_part_name(grammar,k)}\n{float(score[n]):.2f}", fontsize=7, color="black", bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_sample_parse_overlay(
    parser: SpatialAOGParser,
    batch: dict[str, Any],
    out_path: str | Path,
    *,
    use_true_class: bool = False,
) -> tuple[Path, dict[str, Any]]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = parser.parse_batch(batch, return_parse=True)
    y = int(batch["obj_label"][0].item())
    pred = int(out["logits"].argmax(-1)[0].detach().cpu().item())
    parse = out["parse_graph"][0]
    if use_true_class and pred != y:
        # The parser returns only predicted-class parse. For a simple true-class
        # overlay we change the title only. Use compare_sample_true_pred for both.
        pass
    img = _to_image(batch.get("image", None)[0] if "image" in batch else None)
    geom = batch["terminal_geom"][0].detach().float().cpu()
    masks = batch.get("terminal_mask", None)
    masks0 = masks[0].detach().float().cpu() if torch.is_tensor(masks) else None
    if img is None:
        img = torch.ones(512, 512, 3)
    h, w = int(img.shape[0]), int(img.shape[1])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.set_axis_off()
    correct = pred == y
    ax.set_title(
        f"AOG parse overlay | idx={int(batch['index'][0])} | true={_class_name(parser.grammar,y)} | pred={_class_name(parser.grammar,pred)} | correct={correct}\n"
        f"template={parse['template']} edge_cov={float(out['edge_cov'][0].detach().cpu()):.3f} inst_edges={float(out['inst_edges'][0].detach().cpu()):.2f}",
        fontsize=10,
    )
    assigned_terms: set[int] = set()
    for slot in parse["slots"]:
        n = int(slot["terminal"])
        if n < 0 or n >= geom.shape[0]:
            continue
        assigned_terms.add(n)
        part_name = slot["part"]
        color = _color_for_index(int(slot["slot"]))
        if masks0 is not None:
            m = torch.nn.functional.interpolate(masks0[n].view(1, 1, *masks0[n].shape), size=(h, w), mode="nearest")[0, 0].numpy()
            ax.contour(m, levels=[0.5], colors=[color], linewidths=2.0)
        x, yy = float(geom[n, 0].item()) * w, float(geom[n, 1].item()) * h
        ax.scatter([x], [yy], s=45, color=color, edgecolor="white", linewidth=0.8)
        ax.text(x, yy, f"s{slot['slot']}:{part_name}\nt{n}", fontsize=8, bbox=dict(facecolor="white", alpha=0.72, edgecolor="none"))
    for edge in parse["edges"]:
        if not edge.get("instantiated", False):
            continue
        ni, nj = int(edge["terminal_i"]), int(edge["terminal_j"])
        if ni < 0 or nj < 0 or ni >= geom.shape[0] or nj >= geom.shape[0]:
            continue
        x0, y0 = float(geom[ni, 0].item()) * w, float(geom[ni, 1].item()) * h
        x1, y1 = float(geom[nj, 0].item()) * w, float(geom[nj, 1].item()) * h
        ax.plot([x0, x1], [y0, y1], "--", linewidth=1.8, color="white", alpha=0.9)
        ax.plot([x0, x1], [y0, y1], "--", linewidth=1.0, color="black", alpha=0.9)
        ax.text((x0 + x1)/2, (y0 + y1)/2, str(edge.get("type", "rel"))[:18], fontsize=6, bbox=dict(facecolor="white", alpha=0.60, edgecolor="none"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path, parse


def write_sample_parse_dot(parser: SpatialAOGParser, parse: dict[str, Any], out_dir: str | Path, *, render: bool = True, prefix: str = "sample_parse") -> dict[str, Path | list[Path]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = parser.grammar
    c = int(parse["class_idx"])
    a = int(parse["template"])
    lines = [
        f"digraph Parse_{prefix} {{",
        "  rankdir=LR;",
        f"  graph [fontsize=10, labelloc=t, label=\"Instantiated parse graph: {_class_name(g,c)} template {a}\"];",
        "  node [fontsize=9];",
        f'  C [label="O: {_class_name(g,c)}", shape=oval, style=filled, fillcolor="#dbeafe"];',
        f'  T [label="A: template {a}", shape=box, style=filled, fillcolor="#dcfce7"];',
        "  C -> T [label=\"switch\"];",
    ]
    for slot in parse["slots"]:
        s = int(slot["slot"])
        n = int(slot["terminal"])
        pn = str(slot["part"])
        fill = "#fee2e2" if n < 0 else "#fef9c3"
        lines.append(f'  S{s} [label="slot s{s}: {pn}\\nsupport={float(slot.get("support",0)):.2f}", shape=box, style=filled, fillcolor="{fill}"];')
        lines.append(f"  T -> S{s} [style=dotted];")
        if n >= 0:
            lines.append(f'  N{n} [label="terminal t{n}\\n{pn}", shape=ellipse, style=filled, fillcolor="#f5f5f5"];')
            lines.append(f"  S{s} -> N{n} [label=\"address\"];")
    for e in parse["edges"]:
        if not e.get("instantiated", False):
            continue
        ni, nj = int(e["terminal_i"]), int(e["terminal_j"])
        etype = str(e.get("type", "rel")).replace('"', "'")
        lines.append(f'  N{ni} -> N{nj} [dir=none, color="#dc2626", penwidth=2.0, label="{etype}"];')
    lines.append("}")
    dot_path = _write_text(out_dir / f"{prefix}.dot", "\n".join(lines))
    rendered = _render_dot(dot_path) if render else []
    return {"dot": dot_path, "rendered": rendered}


def visualize_sample(
    grammar: SpatialAOGGrammar,
    cache_path: str | Path,
    out_dir: str | Path,
    *,
    sample_pos: int | None = None,
    sample_index: int | None = None,
    parser_cfg: ParserConfig | None = None,
    device: str | torch.device = "cpu",
    render_dot: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = AOGTerminalDataset(cache_path)
    pos, rec = _record_by_position_or_index(ds, sample_pos=sample_pos, sample_index=sample_index)
    batch = _single_batch_from_record(rec, include_masks=True, include_images=True)
    parser = SpatialAOGParser(grammar, parser_cfg, device=device)
    terminal_path = plot_terminal_proposals(grammar, batch, out_dir / f"sample_pos{pos}_terminals.png")
    overlay_path, parse = plot_sample_parse_overlay(parser, batch, out_dir / f"sample_pos{pos}_parse_overlay.png")
    dot_info = write_sample_parse_dot(parser, parse, out_dir, render=render_dot, prefix=f"sample_pos{pos}_parse_graph")
    summary = {
        "sample_pos": int(pos),
        "sample_index": int(rec.index),
        "true_class": _class_name(grammar, int(rec.label)),
        "pred_class": parse.get("pred_class"),
        "template": parse.get("template"),
        "num_slots": len(parse.get("slots", [])),
        "num_instantiated_edges": sum(1 for e in parse.get("edges", []) if e.get("instantiated", False)),
        "terminal_plot": str(terminal_path),
        "overlay_plot": str(overlay_path),
        "parse_dot": str(dot_info["dot"]),
        "parse_rendered": [str(p) for p in dot_info.get("rendered", [])],
        "parse": parse,
    }
    (out_dir / f"sample_pos{pos}_parse_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@torch.no_grad()
def visualize_wrong_samples(
    grammar: SpatialAOGGrammar,
    cache_path: str | Path,
    out_dir: str | Path,
    *,
    max_wrong: int = 16,
    batch_size: int = 16,
    parser_cfg: ParserConfig | None = None,
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parser = SpatialAOGParser(grammar, parser_cfg, device=device)
    ds = AOGTerminalDataset(cache_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: collate_terminal_batch(b, include_masks=True, include_images=True))
    rows: list[dict[str, Any]] = []
    count = 0
    for batch in loader:
        out = parser.parse_batch(batch, return_parse=True)
        pred = out["logits"].argmax(-1).detach().cpu()
        y = batch["obj_label"].detach().cpu()
        for i in range(int(y.numel())):
            if int(pred[i]) == int(y[i]):
                continue
            rec = TerminalRecord(
                label=int(y[i]),
                terminal_valid=batch["terminal_valid"][i],
                terminal_part=batch["terminal_part"][i],
                terminal_score=batch["terminal_score"][i],
                terminal_geom=batch["terminal_geom"][i],
                terminal_token=batch["terminal_token"][i],
                terminal_mask=batch.get("terminal_mask", None)[i] if "terminal_mask" in batch else None,
                image=batch.get("image", None)[i] if "image" in batch else None,
                index=int(batch["index"][i]),
            )
            single = _single_batch_from_record(rec, include_masks=True, include_images=True)
            overlay_path, parse = plot_sample_parse_overlay(parser, single, out_dir / f"wrong_{count:04d}_idx{int(rec.index)}_overlay.png")
            dot_info = write_sample_parse_dot(parser, parse, out_dir, prefix=f"wrong_{count:04d}_idx{int(rec.index)}_parse_graph")
            rows.append({
                "rank": count,
                "index": int(rec.index),
                "true": _class_name(grammar, int(y[i])),
                "pred": _class_name(grammar, int(pred[i])),
                "template": int(parse.get("template", -1)),
                "overlay": str(overlay_path),
                "dot": str(dot_info["dot"]),
            })
            count += 1
            if count >= int(max_wrong):
                _csv_write(out_dir / "wrong_samples.csv", rows)
                return rows
    _csv_write(out_dir / "wrong_samples.csv", rows)
    return rows


def visualize_all(
    grammar_path: str | Path,
    out_dir: str | Path,
    *,
    cache_path: str | Path | None = None,
    class_name: str | None = None,
    class_idx: int | None = None,
    sample_pos: int | None = None,
    sample_index: int | None = None,
    max_wrong: int = 0,
    device: str | torch.device = "cpu",
    render_dot: bool = True,
) -> dict[str, Any]:
    grammar = load_spatial_aog(grammar_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"out_dir": str(out_dir), "grammar": str(grammar_path)}
    global_dir = out_dir / "global"
    manifest["global_dot"] = write_global_aog_dot(grammar, global_dir, render=render_dot)
    manifest["global_plots"] = plot_global_summary(grammar, global_dir)
    c = _resolve_class(grammar, class_name=class_name, class_idx=class_idx) if (class_name or class_idx is not None) else 0
    cat_dir = out_dir / "category" / _safe_name(_class_name(grammar, c))
    manifest["category_class"] = _class_name(grammar, c)
    manifest["category_dot"] = write_category_aog_dot(grammar, c, cat_dir, render=render_dot)
    manifest["category_template_plots"] = plot_category_templates(grammar, c, cat_dir)
    if cache_path is not None:
        sample_dir = out_dir / "sample"
        manifest["sample"] = visualize_sample(grammar, cache_path, sample_dir, sample_pos=sample_pos, sample_index=sample_index, device=device, render_dot=render_dot)
        if max_wrong and int(max_wrong) > 0:
            wrong_dir = out_dir / "wrong_samples"
            manifest["wrong_samples"] = visualize_wrong_samples(grammar, cache_path, wrong_dir, max_wrong=int(max_wrong), device=device)
    # Convert Path objects to strings for JSON.
    def clean(x: Any) -> Any:
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x
    manifest = clean(manifest)
    (out_dir / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
