from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from functools import partial
import csv
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from partcat_hkg.utils.io import save_checkpoint, save_json
from .parser import CompleteAOGParser
from .terminals import CompleteAOGTerminalDataset, ShardBatchSampler, collate_terminal_batch


def _save_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in history:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def _acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(-1) == labels).float().mean().detach().cpu().item())


def complete_aog_loss(
    out: dict[str, Any],
    labels: torch.Tensor,
    *,
    cls_weight: float = 1.0,
    relation_nll_weight: float = 1.0,
    relation_coverage_weight: float = 0.50,
    duplicate_weight: float = 0.50,
    edge_coverage_tau: float = 0.60,
    relation_nll_clip: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = labels.long().to(out["logits"].device)
    ce = F.cross_entropy(out["logits"], labels)
    loss = float(cls_weight) * ce
    logs = {"ce": float(ce.detach().cpu())}
    if relation_nll_weight > 0 and "gt_edge_score" in out:
        # Edge score is a log-likelihood-like quantity.  Maximizing it is direct
        # relation-template learning for the ground-truth parse branch.
        edge_score = out["gt_edge_score"]
        if relation_nll_clip and float(relation_nll_clip) > 0:
            edge_score = edge_score.clamp(min=-float(relation_nll_clip), max=float(relation_nll_clip))
        rel_nll = -edge_score.mean()
        loss = loss + float(relation_nll_weight) * rel_nll
        logs["rel_nll"] = float(rel_nll.detach().cpu())
    if relation_coverage_weight > 0 and "gt_edge_coverage" in out:
        cov_loss = torch.relu(torch.tensor(float(edge_coverage_tau), device=labels.device) - out["gt_edge_coverage"]).mean()
        loss = loss + float(relation_coverage_weight) * cov_loss
        logs["edge_cov_loss"] = float(cov_loss.detach().cpu())
    if duplicate_weight > 0 and "gt_duplicate_mass" in out:
        dup = out["gt_duplicate_mass"].mean()
        loss = loss + float(duplicate_weight) * dup
        logs["dup_loss"] = float(dup.detach().cpu())
    logs["loss"] = float(loss.detach().cpu())
    return loss, logs


@torch.no_grad()
def evaluate_complete_aog(
    model: CompleteAOGParser,
    loader: DataLoader,
    *,
    device: str | torch.device = "cuda",
    max_batches: int = 0,
    loss_kwargs: dict[str, Any] | None = None,
) -> dict[str, float]:
    model.eval()
    dev = torch.device(device)
    run = defaultdict(float)
    total = correct = 0
    n_batches = 0
    t0 = time.time()
    for bi, batch in enumerate(loader):
        if max_batches and bi >= int(max_batches):
            break
        labels = batch["obj_label"].to(dev, non_blocking=True)
        out = model(batch, labels=labels)
        loss, logs = complete_aog_loss(out, labels, **(loss_kwargs or {}))
        for k, v in logs.items():
            run[k] += float(v)
        correct += int((out["logits"].argmax(-1) == labels).sum().item())
        total += int(labels.numel())
        run["logit_std"] += float(out["logits"].std(dim=-1).mean().detach().cpu())
        run["edge_coverage"] += float(out["edge_coverage_pred"].mean().detach().cpu())
        run["duplicate_mass"] += float(out["duplicate_mass"].mean().detach().cpu())
        run["inst_edges"] += float(out["inst_edges"].mean().detach().cpu())
        run["edge_miss"] += float(out["edge_miss"].mean().detach().cpu())
        n_batches += 1
    row = {f"val_{k}": v / max(n_batches, 1) for k, v in run.items()}
    row["val_acc"] = float(correct / max(total, 1))
    row["val_wall_sec"] = time.time() - t0
    return row


def train_complete_aog(
    model: CompleteAOGParser,
    train_cache: str | Path,
    val_cache: str | Path,
    *,
    save_dir: str | Path,
    device: str | torch.device = "cuda",
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 2,
    pin_memory: bool = False,
    grad_clip: float = 5.0,
    progress_every: int = 20,
    max_train_batches: int = 0,
    max_val_batches: int = 0,
    loss_kwargs: dict[str, Any] | None = None,
    preload_cache: bool = False,
    shard_batches: bool = False,
    lru_shards: int = 4,
    include_train_masks: bool = False,
    include_val_masks: bool = False,
    profile_time: bool = False,
) -> list[dict[str, float]]:
    dev = torch.device(device if str(device) != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    train_ds = CompleteAOGTerminalDataset(
        train_cache,
        preload=bool(preload_cache),
        include_masks=bool(include_train_masks),
        include_images=False,
        lru_shards=int(lru_shards),
    )
    val_ds = CompleteAOGTerminalDataset(
        val_cache,
        preload=bool(preload_cache),
        include_masks=bool(include_val_masks),
        include_images=False,
        lru_shards=int(lru_shards),
    )
    train_collate = partial(collate_terminal_batch, include_masks=bool(include_train_masks), include_images=False)
    val_collate = partial(collate_terminal_batch, include_masks=bool(include_val_masks), include_images=False)
    common = dict(num_workers=int(num_workers), pin_memory=bool(pin_memory))
    if bool(shard_batches) and not bool(preload_cache):
        train_loader = DataLoader(train_ds, batch_sampler=ShardBatchSampler(train_ds, int(batch_size), shuffle=True), collate_fn=train_collate, **common)
    else:
        train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, collate_fn=train_collate, **common)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, collate_fn=val_collate, **common)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=float(weight_decay))
    save_dir = Path(save_dir)
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best = -1.0
    loss_kwargs = dict(loss_kwargs or {})
    for epoch in range(1, int(epochs) + 1):
        model.train()
        run = defaultdict(float)
        n_batches = total = correct = 0
        t0 = time.time()
        data_t0 = time.time()
        for bi, batch in enumerate(train_loader, start=1):
            data_time = time.time() - data_t0
            if max_train_batches and bi > int(max_train_batches):
                break
            labels = batch["obj_label"].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            fwd_t0 = time.time()
            out = model(batch, labels=labels)
            loss, logs = complete_aog_loss(out, labels, **loss_kwargs)
            if torch.cuda.is_available() and dev.type == "cuda":
                torch.cuda.synchronize()
            fwd_time = time.time() - fwd_t0
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Complete AOG non-finite loss at epoch={epoch} batch={bi}: {loss}")
            bwd_t0 = time.time()
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            if torch.cuda.is_available() and dev.type == "cuda":
                torch.cuda.synchronize()
            bwd_time = time.time() - bwd_t0
            for k, v in logs.items():
                run[k] += float(v)
            with torch.no_grad():
                correct += int((out["logits"].argmax(-1) == labels).sum().item())
                total += int(labels.numel())
                run["logit_std"] += float(out["logits"].std(dim=-1).mean().detach().cpu())
                run["edge_coverage"] += float(out["edge_coverage_pred"].mean().detach().cpu())
                run["duplicate_mass"] += float(out["duplicate_mass"].mean().detach().cpu())
                run["inst_edges"] += float(out["inst_edges"].mean().detach().cpu())
                run["edge_miss"] += float(out["edge_miss"].mean().detach().cpu())
                if profile_time:
                    run["data_time"] += float(data_time)
                    run["fwd_time"] += float(fwd_time)
                    run["bwd_time"] += float(bwd_time)
            n_batches += 1
            if progress_every and (bi % int(progress_every) == 0):
                print(
                    f"[complete-aog][train] epoch={epoch} batch={bi}/{len(train_loader)} "
                    f"loss={run['loss']/max(n_batches,1):.4f} ce={run.get('ce',0.0)/max(n_batches,1):.4f} rel_nll={run.get('rel_nll',0.0)/max(n_batches,1):.4f} acc={correct/max(total,1):.4f} "
                    f"edge_cov={run['edge_coverage']/max(n_batches,1):.3f} "
                    f"dup={run['duplicate_mass']/max(n_batches,1):.5f} "
                    f"inst_edges={run['inst_edges']/max(n_batches,1):.3f} "
                    f"edge_miss={run['edge_miss']/max(n_batches,1):.3f} "
                    f"elapsed={time.time()-t0:.1f}s" + (f" data={run.get('data_time',0.0)/max(n_batches,1):.3f}s fwd={run.get('fwd_time',0.0)/max(n_batches,1):.3f}s bwd={run.get('bwd_time',0.0)/max(n_batches,1):.3f}s" if profile_time else ""),
                    flush=True,
                )
            data_t0 = time.time()
        row = {f"train_{k}": v / max(n_batches, 1) for k, v in run.items()}
        row["epoch"] = float(epoch)
        row["train_acc"] = float(correct / max(total, 1))
        row["wall_sec"] = time.time() - t0
        if val_loader is not None:
            row.update(evaluate_complete_aog(model, val_loader, device=dev, max_batches=int(max_val_batches), loss_kwargs=loss_kwargs))
        history.append(row)
        save_json(save_dir / "complete_aog_history.json", history)
        _save_history_csv(save_dir / "complete_aog_history.csv", history)
        extra = {"epoch": epoch, "history": history, "score": row.get("val_acc", 0.0), "schema": model.schema.to_payload()}
        save_checkpoint(ckpt_dir / "complete_aog_last.pt", model, extra=extra)
        if row.get("val_acc", 0.0) >= best:
            best = float(row.get("val_acc", 0.0))
            save_checkpoint(ckpt_dir / "complete_aog_best.pt", model, extra=extra)
        print(
            f"[complete-aog] epoch={epoch} "
            f"train_loss={row.get('train_loss', float('nan')):.4f} "
            f"train_acc={row.get('train_acc', float('nan')):.4f} "
            f"val_acc={row.get('val_acc', float('nan')):.4f} "
            f"logit_std={row.get('val_logit_std', float('nan')):.4f} "
            f"edge_cov={row.get('val_edge_coverage', float('nan')):.3f} "
            f"dup={row.get('val_duplicate_mass', float('nan')):.5f} "
            f"inst_edges={row.get('val_inst_edges', float('nan')):.3f} "
            f"edge_miss={row.get('val_edge_miss', float('nan')):.3f}",
            flush=True,
        )
    return history
