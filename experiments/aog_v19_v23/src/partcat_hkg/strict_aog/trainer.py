from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import time

import torch

from .parser import StrictAOGParser, strict_aog_loss


def _accuracy(correct: int, total: int) -> float:
    return float(correct) / float(max(total, 1))


def _save_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _pred_template_values(out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    pred = out["logits"].argmax(-1)
    b = torch.arange(pred.shape[0], device=pred.device)
    t = out["best_template"][b, pred]
    node = out["node_scores"][b, pred, t]
    edge = out["edge_scores"][b, pred, t]
    return {"pred": pred, "template": t, "node": node, "edge": edge}


@torch.no_grad()
def evaluate_strict_aog(
    model: StrictAOGParser,
    loader,
    *,
    device: str | torch.device = "cuda",
    enable_edges: bool = True,
    label_smoothing: float = 0.0,
    edge_aux_weight: float = 0.0,
    node_aux_weight: float = 0.0,
    margin_weight: float = 0.0,
    margin: float = 0.50,
    max_batches: int = 0,
    progress_every: int = 0,
) -> dict[str, float]:
    model.eval()
    total = correct = 0
    run = defaultdict(float)
    n_batches = 0
    t_eval = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        if int(max_batches or 0) > 0 and batch_idx > int(max_batches):
            break
        labels = batch["obj_label"].to(device, non_blocking=True)
        out = model(batch, enable_edges=enable_edges)
        _loss, logs = strict_aog_loss(
            out,
            labels,
            label_smoothing=float(label_smoothing),
            edge_aux_weight=float(edge_aux_weight),
            node_aux_weight=float(node_aux_weight),
            margin_weight=float(margin_weight),
            margin=float(margin),
        )
        pred = out["logits"].argmax(-1)
        total += int(labels.numel())
        correct += int((pred == labels).sum().item())
        vals = _pred_template_values(out)
        edge_fraction = (vals["edge"].abs() / (vals["node"].abs() + vals["edge"].abs() + 1e-6)).mean()
        run["logit_std"] += float(out["logits"].std(dim=-1).mean().detach().cpu())
        run["edge_fraction_raw"] += float(edge_fraction.detach().cpu())
        run["assignment_reuse"] += float(out.get("assignment_reuse_mean", torch.tensor(0.0, device=labels.device)).detach().cpu())
        run["edge_missing"] += float(out.get("edge_missing_mean", torch.tensor(0.0, device=labels.device)).detach().cpu())
        run["instantiated_edges"] += float(out.get("instantiated_edge_mean", torch.tensor(0.0, device=labels.device)).detach().cpu())
        for k, v in logs.items():
            if k in {"loss", "ce_final", "ce_edge", "ce_node", "count_fraction_raw", "count_score", "role_overlap", "edge_coverage", "parse_validity_penalty"}:
                run[k] += float(v)
        n_batches += 1
        if progress_every and batch_idx % int(progress_every) == 0:
            print(
                f"[strict-aog][val] batch={batch_idx}/{len(loader) if hasattr(loader, '__len__') else '?'} "
                f"acc={_accuracy(correct, total):.4f} elapsed={time.time() - t_eval:.1f}s",
                flush=True,
            )
    row = {f"val_{k}": v / max(n_batches, 1) for k, v in run.items()}
    row["val_acc"] = _accuracy(correct, total)
    return row


def train_strict_aog(
    model: StrictAOGParser,
    train_loader,
    val_loader,
    *,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str | torch.device = "cuda",
    save_dir: str | Path = "runs/strict_aog",
    enable_edges: bool = True,
    edge_start_epoch: int = 1,
    label_smoothing: float = 0.0,
    edge_aux_weight: float = 0.0,
    node_aux_weight: float = 0.0,
    margin_weight: float = 0.0,
    margin: float = 0.50,
    fail_on_uniform: bool = True,
    fail_on_reuse: bool = True,
    progress_every: int = 10,
    max_train_batches: int = 0,
    max_val_batches: int = 0,
) -> list[dict[str, float]]:
    """Train grammar calibration/projection parameters.

    Edges are enabled from epoch 1 by default.  The default parser is the
    edge-aware beam parser, so horizontal relations influence the selected parse,
    not just the final score.
    """
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=float(weight_decay))
    save_dir = Path(save_dir)
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best = -1.0
    uniform_floor = 1e-5
    for epoch in range(1, int(epochs) + 1):
        t0 = time.time()
        model.train()
        run = defaultdict(float)
        nb = 0
        total = correct = 0
        enable_edges_epoch = bool(enable_edges and epoch >= int(edge_start_epoch))
        for batch_idx, batch in enumerate(train_loader, start=1):
            if int(max_train_batches or 0) > 0 and batch_idx > int(max_train_batches):
                break
            labels = batch["obj_label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(batch, enable_edges=enable_edges_epoch)
            loss, logs = strict_aog_loss(
                out,
                labels,
                label_smoothing=float(label_smoothing),
                edge_aux_weight=float(edge_aux_weight),
                node_aux_weight=float(node_aux_weight),
                margin_weight=float(margin_weight),
                margin=float(margin),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Strict AOG non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            pred = out["logits"].argmax(-1)
            total += int(labels.numel())
            correct += int((pred == labels).sum().item())
            for k, v in logs.items():
                run[k] += float(v)
            nb += 1
            if progress_every and batch_idx % int(progress_every) == 0:
                denom = max(nb, 1)
                print(
                    f"[strict-aog][train] epoch={epoch} batch={batch_idx}/{len(train_loader) if hasattr(train_loader, '__len__') else '?'} "
                    f"loss={run.get('loss', 0.0)/denom:.4f} acc={_accuracy(correct, total):.4f} "
                    f"edge_frac={run.get('edge_fraction_raw', 0.0)/denom:.3f} "
                    f"count_frac={run.get('count_fraction_raw', 0.0)/denom:.3f} "
                    f"role={run.get('role_overlap', 0.0)/denom:.3f} "
                    f"edge_cov={run.get('edge_coverage', 0.0)/denom:.3f} "
                    f"valid_pen={run.get('parse_validity_penalty', 0.0)/denom:.3f} "
                    f"reuse={run.get('assignment_reuse', 0.0)/denom:.4f} "
                    f"edge_miss={run.get('edge_missing', 0.0)/denom:.3f} "
                    f"inst_edges={run.get('instantiated_edges', 0.0)/denom:.3f} "
                    f"elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )
        row = {f"train_{k}": v / max(nb, 1) for k, v in run.items()}
        row["train_acc"] = _accuracy(correct, total)
        row["epoch"] = float(epoch)
        row["wall_sec"] = time.time() - t0
        row["edges_enabled_train"] = float(enable_edges_epoch)
        if val_loader is not None:
            row.update(evaluate_strict_aog(
                model,
                val_loader,
                device=device,
                enable_edges=bool(enable_edges),
                label_smoothing=float(label_smoothing),
                edge_aux_weight=float(edge_aux_weight),
                node_aux_weight=float(node_aux_weight),
                margin_weight=float(margin_weight),
                margin=float(margin),
                max_batches=int(max_val_batches or 0),
                progress_every=int(progress_every) if int(progress_every or 0) > 0 else 0,
            ))
        history.append(row)
        _save_csv(save_dir / "strict_aog_history.csv", history)
        score = float(row.get("val_acc", row.get("train_acc", 0.0)))
        ckpt = {"model": model.state_dict(), "epoch": epoch, "history": history, "grammar": model.grammar.to_payload()}
        torch.save(ckpt, save_dir / "checkpoints" / "strict_aog_last.pt")
        if score >= best:
            best = score
            torch.save(ckpt, save_dir / "checkpoints" / "strict_aog_best.pt")
        msg = (
            f"[strict-aog] epoch={epoch} train_loss={row.get('train_loss', float('nan')):.4f} "
            f"train_acc={row.get('train_acc', float('nan')):.4f} "
            f"val_acc={row.get('val_acc', float('nan')):.4f} "
            f"logit_std={row.get('val_logit_std', row.get('train_logit_std', float('nan'))):.6f} "
            f"edge_frac={row.get('val_edge_fraction_raw', row.get('train_edge_fraction_raw', float('nan'))):.3f} "
            f"count_frac={row.get('val_count_fraction_raw', row.get('train_count_fraction_raw', float('nan'))):.3f} "
            f"role={row.get('val_role_overlap', row.get('train_role_overlap', float('nan'))):.3f} "
            f"edge_cov={row.get('val_edge_coverage', row.get('train_edge_coverage', float('nan'))):.3f} "
            f"valid_pen={row.get('val_parse_validity_penalty', row.get('train_parse_validity_penalty', float('nan'))):.3f} "
            f"reuse={row.get('train_assignment_reuse', float('nan')):.4f} "
            f"edge_miss={row.get('train_edge_missing', float('nan')):.3f} "
            f"inst_edges={row.get('train_instantiated_edges', float('nan')):.3f} "
            f"edges={int(enable_edges_epoch)}"
        )
        print(msg)
        if fail_on_uniform and epoch >= 2:
            std = float(row.get("val_logit_std", row.get("train_logit_std", 0.0)))
            if std < uniform_floor:
                raise RuntimeError(
                    "Strict AOG logits are effectively uniform. Check terminal cache coverage, "
                    "grammar slot_valid/template_valid, and label/schema alignment."
                )
        if fail_on_reuse:
            reuse = float(row.get("train_assignment_reuse", 0.0))
            assignment = str(getattr(getattr(model, "cfg", None), "assignment", "")).lower()
            soft_mode = assignment in {"gpu_mf", "mf", "mean_field", "meanfield", "gpu-meanfield"}
            # gpu_mf uses a soft posterior over terminals.  A tiny positive
            # duplicate mass is a relaxation artifact, not necessarily an
            # invalid decoded parse.  Hard parsers must keep reuse exactly zero;
            # soft parsers only fail if the selected parse posterior is clearly
            # violating uniqueness.
            threshold = 0.25 if soft_mode else 1e-5
            if reuse > threshold:
                raise RuntimeError(
                    f"Invalid parse: selected terminal reuse is {reuse} with assignment={assignment}. "
                    f"Threshold={threshold}. Increase --mf-column-iters, lower --mf-tau, or inspect terminal cache / slot construction."
                )
    return history
