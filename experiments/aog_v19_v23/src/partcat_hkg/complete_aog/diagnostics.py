from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .parser import CompleteAOGParser
from .terminals import CompleteAOGTerminalDataset, collate_terminal_batch


def _denorm_img(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu().float()
    if x.ndim != 3:
        return torch.zeros(3, 224, 224)
    # The project transform may already be 0..1 or normalized.  Robustly map to display range.
    lo, hi = float(x.min()), float(x.max())
    if lo < -0.1 or hi > 1.1:
        x = (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)
    return x.clamp(0, 1)


def _centroid_from_geom(g: torch.Tensor) -> tuple[float, float]:
    return float(g[0].detach().cpu()), float(g[1].detach().cpu())


def save_wrong_overlays(
    model: CompleteAOGParser,
    cache_path: str | Path,
    out_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    batch_size: int = 16,
    max_wrong: int = 32,
    num_workers: int = 0,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "wrong_overlays"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ds = CompleteAOGTerminalDataset(cache_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_terminal_batch)
    dev = torch.device(device if str(device) != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(dev)
    rows: list[dict[str, Any]] = []
    saved = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch["obj_label"].to(dev)
            out = model(batch, labels=labels, return_parse=True)
            pred = out["logits"].argmax(-1).detach().cpu()
            wrong = (pred != batch["obj_label"]).nonzero(as_tuple=False).flatten().tolist()
            if not wrong:
                continue
            parses = out.get("parse_graph", [])
            for local in wrong:
                if saved >= max_wrong:
                    break
                parse = parses[local] if local < len(parses) else None
                label = int(batch["obj_label"][local].item())
                pcls = int(pred[local].item())
                fig, ax = plt.subplots(figsize=(7, 5))
                if "image" in batch:
                    img = _denorm_img(batch["image"][local])
                    ax.imshow(img.permute(1, 2, 0))
                    h, w = img.shape[-2:]
                else:
                    h = w = int(batch["terminal_mask"].shape[-1])
                    ax.imshow(torch.ones(h, w, 3))
                geom = batch["terminal_geom"][local]
                masks = batch["terminal_mask"][local].float()
                if parse:
                    # Draw assigned masks and slot labels.
                    for slot in parse["slots"]:
                        tid = slot.get("terminal")
                        if tid is None:
                            continue
                        mask = masks[int(tid)]
                        ax.contour(mask.numpy(), levels=[0.5], linewidths=1.2, alpha=0.8, extent=(0, w, h, 0))
                        cx, cy = _centroid_from_geom(geom[int(tid)])
                        ax.text(cx * w, cy * h, f"s{slot['slot']}:{slot['part']}", fontsize=7, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
                    for e in parse["edges"]:
                        if not e.get("instantiated"):
                            continue
                        ti, tj = int(e["terminal_i"]), int(e["terminal_j"])
                        x1, y1 = _centroid_from_geom(geom[ti])
                        x2, y2 = _centroid_from_geom(geom[tj])
                        ax.plot([x1*w, x2*w], [y1*h, y2*h], linestyle="--", linewidth=1.0)
                        ax.text((x1+x2)*0.5*w, (y1+y2)*0.5*h, e.get("type", "rel"), fontsize=6, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
                ax.set_title(f"wrong #{saved}: true={model.schema.obj_names[label]} pred={model.schema.obj_names[pcls]}")
                ax.axis("off")
                fpath = fig_dir / f"wrong_{saved:04d}_true_{label}_pred_{pcls}.png"
                fig.tight_layout()
                fig.savefig(fpath, dpi=160)
                plt.close(fig)
                rows.append({"file": str(fpath), "true": model.schema.obj_names[label], "pred": model.schema.obj_names[pcls]})
                saved += 1
            if saved >= max_wrong:
                break
    return {"num_wrong_saved": saved, "rows": rows}
