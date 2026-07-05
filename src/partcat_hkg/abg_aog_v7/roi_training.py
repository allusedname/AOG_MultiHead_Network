from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .queryable_stage1 import crop_normalized
from .roi_requery_head import ROIRequeryHeadV7
from .terminal_adapter import terminal_packets_from_record


@dataclass
class ROITrainConfigV7:
    crop_size: int = 64
    score_tau: float = 0.20
    lr: float = 1e-3
    epochs: int = 5
    device: str = "cuda"


class ROICacheDatasetV7(Dataset):
    """Build ROI re-query supervision from cache records with image and terminal_mask."""

    def __init__(self, records: list[dict[str, Any]], *, cfg: ROITrainConfigV7 | None = None) -> None:
        self.records = records
        self.cfg = cfg or ROITrainConfigV7()
        self.index: list[tuple[int, int]] = []
        for ri, r in enumerate(records):
            if not torch.is_tensor(r.get("image")):
                continue
            terms = terminal_packets_from_record(r, sample_id=ri, score_tau=self.cfg.score_tau, include_masks=True)
            for ti, t in enumerate(terms):
                if t.visible_mask is not None:
                    self.index.append((ri, ti))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ri, ti = self.index[int(idx)]
        r = self.records[ri]
        terms = terminal_packets_from_record(r, sample_id=ri, score_tau=self.cfg.score_tau, include_masks=True)
        t = terms[ti]
        image = r["image"].float()
        crop = crop_normalized(image, t.visible_box_xyxy, size=int(self.cfg.crop_size))[0]
        mask = t.visible_mask.float().unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=(self.cfg.crop_size, self.cfg.crop_size), mode="nearest")[0]
        ports = torch.zeros(5, self.cfg.crop_size, self.cfg.crop_size)
        return {"crop": crop, "part_id": torch.tensor(t.functional_part_id, dtype=torch.long), "mask": mask, "ports": ports, "presence": torch.tensor(1.0)}


def roi_loss_v7(out, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = batch["mask"].float().to(out.visible_mask_logits.device)
    ports = batch["ports"].float().to(out.port_heatmaps.device)
    presence = batch["presence"].float().to(out.visible_score.device)
    bce = F.binary_cross_entropy_with_logits(out.visible_mask_logits, mask)
    amodal = F.binary_cross_entropy_with_logits(out.amodal_mask_logits, mask)
    p_loss = F.binary_cross_entropy(out.visible_score, presence)
    port_loss = F.mse_loss(torch.sigmoid(out.port_heatmaps[:, : ports.shape[1]]), ports)
    return bce + 0.5 * amodal + p_loss + 0.1 * port_loss


def train_roi_requery_head_v7(model: ROIRequeryHeadV7, loader, *, cfg: ROITrainConfigV7 | None = None) -> list[dict[str, float]]:
    cfg = cfg or ROITrainConfigV7()
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    logs: list[dict[str, float]] = []
    for ep in range(int(cfg.epochs)):
        total = 0.0
        n = 0
        for batch in loader:
            crop = batch["crop"].to(device)
            part_id = batch["part_id"].to(device)
            out = model(crop, part_id)
            loss = roi_loss_v7(out, {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            n += 1
        logs.append({"epoch": float(ep), "loss": total / max(1, n)})
    return logs
