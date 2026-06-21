from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .parser import SpatialAOGParser
from .terminals import AOGTerminalDataset, collate_terminal_batch


def _to_img(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    if x.ndim == 3 and x.shape[0] == 3:
        if float(x.min()) < -0.05 or float(x.max()) > 1.05:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(3, 1, 1)
            x = x * std + mean
        x = x.permute(1, 2, 0)
    return x.clamp(0, 1)


@torch.no_grad()
def save_wrong_parse_overlays(
    parser: SpatialAOGParser,
    cache_path: str | Path,
    out_dir: str | Path,
    *,
    batch_size: int = 16,
    max_wrong: int = 32,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = AOGTerminalDataset(cache_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: collate_terminal_batch(b, include_masks=True, include_images=True))
    saved: list[Path] = []
    for batch in loader:
        out = parser.parse_batch(batch, return_parse=True)
        y = batch["obj_label"].to(out["logits"].device)
        pred = out["logits"].argmax(-1)
        for i in range(int(y.numel())):
            if int(pred[i].item()) == int(y[i].item()):
                continue
            if "image" not in batch or "terminal_mask" not in batch:
                continue
            parse = out["parse_graph"][i]
            img = _to_img(batch["image"][i])
            masks = batch["terminal_mask"][i].float()
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(f"idx={int(batch['index'][i])} true={parser.grammar.schema.obj_names[int(y[i])]} pred={parser.grammar.schema.obj_names[int(pred[i])]}\ntemplate={parse['template']}")
            # Draw assigned masks and slot labels.
            h, w = img.shape[:2]
            for slot in parse["slots"]:
                n = int(slot["terminal"])
                if n < 0 or n >= masks.shape[0]:
                    continue
                m = torch.nn.functional.interpolate(masks[n].view(1, 1, *masks.shape[-2:]), size=(h, w), mode="nearest")[0, 0].numpy()
                ax.contour(m, levels=[0.5], linewidths=1.5)
                geom = batch["terminal_geom"][i, n]
                ax.text(float(geom[0]) * w, float(geom[1]) * h, f"s{slot['slot']}:{slot['part']}", fontsize=8, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
            # Draw relation lines.
            for edge in parse["edges"]:
                if not edge["instantiated"]:
                    continue
                ni, nj = int(edge["terminal_i"]), int(edge["terminal_j"])
                gi = batch["terminal_geom"][i, ni]
                gj = batch["terminal_geom"][i, nj]
                x0, y0 = float(gi[0]) * w, float(gi[1]) * h
                x1, y1 = float(gj[0]) * w, float(gj[1]) * h
                ax.plot([x0, x1], [y0, y1], linestyle="--", linewidth=1.0)
            p = out_dir / f"wrong_{len(saved):04d}_idx{int(batch['index'][i])}.png"
            fig.tight_layout()
            fig.savefig(p, dpi=150)
            plt.close(fig)
            saved.append(p)
            if len(saved) >= int(max_wrong):
                return saved
    return saved
