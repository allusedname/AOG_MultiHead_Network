#!/usr/bin/env python
from __future__ import annotations

"""Stronger ResNet baseline training for PartImageNet object labels.

This replaces the minimal baseline loop I gave earlier.  The old loop was valid
as a smoke-test classifier, but it was not a strong baseline: it used full-model
AdamW with no LR schedule, no class balancing, and only the repository's light
resize/flip/color-jitter transform.  This script adds standard ImageNet-style
fine-tuning controls and saves preprocessing metadata for cue-conflict eval.
"""

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path and SRC.exists():
    sys.path.insert(0, str(SRC))

from partcat_hkg.config import load_config
from partcat_hkg.data.loaders import make_datasets
from partcat_hkg.data.partimagenet import Stage2ImageOnlyDataset
from partcat_hkg.data.collate import collate_stage2_image_only
from partcat_hkg.data.transforms import ImageOnlyTransform
from partcat_hkg.utils.seed import set_seed


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


class ImagePathLabelDataset(Dataset):
    """Image-only dataset over the already-resolved PartImageNet samples."""

    def __init__(self, full_dataset: Any, transform: Any):
        self.full = full_dataset
        self.samples = full_dataset.samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.samples[int(idx)]
        img = Image.open(rec["img_path"]).convert("RGB")
        x = self.transform(img)
        if isinstance(x, tuple):
            x = x[0]
        return {
            "image": x,
            "obj_label": torch.tensor(int(rec["obj_label"]), dtype=torch.long),
        }


def collate_image_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "obj_label": torch.stack([b["obj_label"] for b in batch], dim=0),
    }


def make_resnet_transforms(preprocess: str, img_size: int, val_resize: int, *, randaugment: bool, erase: float):
    if preprocess == "repo":
        return ImageOnlyTransform(img_size, train=True), ImageOnlyTransform(img_size, train=False)

    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_ops: list[Any] = [
        T.RandomResizedCrop(img_size, scale=(0.50, 1.00), ratio=(0.75, 1.3333333333), interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
    ]
    if randaugment and hasattr(T, "RandAugment"):
        train_ops.append(T.RandAugment(num_ops=2, magnitude=9))
    else:
        train_ops.append(T.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15))
    train_ops.extend([T.ToTensor(), normalize])
    if float(erase) > 0:
        train_ops.append(T.RandomErasing(p=float(erase), scale=(0.02, 0.18), ratio=(0.3, 3.3), value="random"))
    train_tf = T.Compose(train_ops)
    val_tf = T.Compose([
        T.Resize(int(val_resize), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        normalize,
    ])
    return train_tf, val_tf


def make_resnet(model_name: str, num_classes: int, *, imagenet_pretrained: bool) -> nn.Module:
    from torchvision import models
    name = str(model_name).lower()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if imagenet_pretrained else None)
    elif name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if imagenet_pretrained else None)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if imagenet_pretrained else None)
    elif name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if imagenet_pretrained else None)
    else:
        raise ValueError(f"unsupported model: {model_name}")
    model.fc = nn.Linear(model.fc.in_features, int(num_classes))
    return model


def load_model_checkpoint(path: str | Path, model: nn.Module, *, strict: bool = True) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=strict)
    return payload if isinstance(payload, dict) else {}


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad_(bool(trainable))
        else:
            p.requires_grad_(True)


def set_bn_eval(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.startswith("fc.") else backbone).append(p)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": float(args.backbone_lr)})
    if head:
        groups.append({"params": head, "lr": float(args.head_lr)})
    if str(args.optimizer).lower() == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True, weight_decay=float(args.weight_decay))
    return torch.optim.AdamW(groups, weight_decay=float(args.weight_decay))


def make_balanced_sampler(full_dataset: Any) -> WeightedRandomSampler:
    labels = [int(r["obj_label"]) for r in full_dataset.samples]
    counts = Counter(labels)
    weights = torch.tensor([1.0 / float(max(counts[y], 1)) for y in labels], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, *, device: torch.device, amp: bool, num_classes: int) -> dict[str, Any]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    n_batches = 0
    per_total = torch.zeros(num_classes, dtype=torch.long)
    per_correct = torch.zeros(num_classes, dtype=torch.long)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["obj_label"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=bool(amp and device.type == "cuda")):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
        loss_sum += float(loss.detach().cpu())
        n_batches += 1
        for c in range(num_classes):
            mask = labels.detach().cpu() == c
            per_total[c] += int(mask.sum())
            per_correct[c] += int((pred.detach().cpu()[mask] == c).sum())
    per_acc = torch.where(per_total > 0, per_correct.float() / per_total.clamp_min(1).float(), torch.zeros_like(per_total, dtype=torch.float))
    return {
        "acc": correct / float(max(total, 1)),
        "macro_acc": float(per_acc[per_total > 0].mean().item()) if bool((per_total > 0).any()) else 0.0,
        "loss": loss_sum / float(max(n_batches, 1)),
        "n": int(total),
        "per_class_acc": [float(x) for x in per_acc.tolist()],
        "per_class_n": [int(x) for x in per_total.tolist()],
    }


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.partimagenet_root:
        cfg.paths.partimagenet_root = args.partimagenet_root
    if args.num_workers is not None:
        cfg.data.num_workers = int(args.num_workers)
        cfg.data.persistent_workers = cfg.data.num_workers > 0 and cfg.data.persistent_workers
    set_seed(int(args.seed if args.seed is not None else cfg.seed))
    device = _device(args.device)

    train_full, val_full = make_datasets(cfg)
    num_classes = int(train_full.schema.num_classes)
    class_names = list(train_full.schema.obj_names)
    train_tf, val_tf = make_resnet_transforms(
        args.preprocess,
        int(args.img_size),
        int(args.val_resize),
        randaugment=bool(args.randaugment),
        erase=float(args.random_erasing),
    )
    if args.preprocess == "repo":
        train_ds = Stage2ImageOnlyDataset(train_full, train=True)
        val_ds = Stage2ImageOnlyDataset(val_full, train=False)
        collate_fn = collate_stage2_image_only
    else:
        train_ds = ImagePathLabelDataset(train_full, train_tf)
        val_ds = ImagePathLabelDataset(val_full, val_tf)
        collate_fn = collate_image_batch

    common = dict(num_workers=int(cfg.data.num_workers), pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)
    if int(cfg.data.num_workers) > 0:
        common.update(persistent_workers=bool(cfg.data.persistent_workers), prefetch_factor=int(cfg.data.prefetch_factor))
    sampler = make_balanced_sampler(train_full) if bool(args.balanced_sampler) else None
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=(sampler is None), sampler=sampler, **common)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, **common)

    model = make_resnet(args.model, num_classes, imagenet_pretrained=bool(args.imagenet_pretrained)).to(device)
    if args.resume:
        load_model_checkpoint(args.resume, model, strict=True)

    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval-only requires --resume")
        val = evaluate(model, val_loader, device=device, amp=bool(args.amp), num_classes=num_classes)
        print(json.dumps({"val_acc": val["acc"], "val_macro_acc": val["macro_acc"], "val_loss": val["loss"]}, indent=2))
        return

    history: list[dict[str, Any]] = []
    best = -math.inf
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    opt: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None

    for epoch in range(1, int(args.epochs) + 1):
        if epoch == 1:
            set_backbone_trainable(model, trainable=(int(args.freeze_epochs) <= 0))
            opt = make_optimizer(model, args)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(args.epochs)))
        if epoch == int(args.freeze_epochs) + 1 and int(args.freeze_epochs) > 0:
            set_backbone_trainable(model, trainable=True)
            opt = make_optimizer(model, args)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(args.epochs) - int(args.freeze_epochs)))

        assert opt is not None
        t0 = time.time()
        model.train()
        if args.freeze_bn:
            set_bn_eval(model)
        correct = total = 0
        loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["obj_label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=bool(args.amp and device.type == "cuda")):
                logits = model(images)
                loss = F.cross_entropy(logits, labels, label_smoothing=float(args.label_smoothing))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.grad_clip))
            scaler.step(opt)
            scaler.update()
            pred = logits.argmax(-1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
            loss_sum += float(loss.detach().cpu())
            n_batches += 1
        if scheduler is not None:
            scheduler.step()

        val = evaluate(model, val_loader, device=device, amp=bool(args.amp), num_classes=num_classes)
        row = {
            "epoch": int(epoch),
            "train_loss": loss_sum / float(max(n_batches, 1)),
            "train_acc": correct / float(max(total, 1)),
            "val_loss": float(val["loss"]),
            "val_acc": float(val["acc"]),
            "val_macro_acc": float(val["macro_acc"]),
            "lr_backbone": float(opt.param_groups[0]["lr"]) if opt.param_groups else 0.0,
            "lr_head": float(opt.param_groups[-1]["lr"]) if opt.param_groups else 0.0,
            "wall_sec": time.time() - t0,
        }
        history.append(row)
        save_json(save_dir / "resnet_history.json", history)
        save_csv(save_dir / "resnet_history.csv", history)
        save_json(save_dir / "resnet_per_class_val.json", {"class_names": class_names, "last": val})

        payload = {
            "state_dict": model.state_dict(),
            "epoch": int(epoch),
            "history": history,
            "model_name": args.model,
            "num_classes": num_classes,
            "class_names": class_names,
            "preprocess": {
                "kind": args.preprocess,
                "img_size": int(args.img_size),
                "val_resize": int(args.val_resize),
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "config": cfg.to_dict(),
        }
        torch.save(payload, ckpt_dir / "resnet_last.pt")
        if row["val_acc"] >= best:
            best = float(row["val_acc"])
            torch.save(payload, ckpt_dir / "resnet_best.pt")
        print(
            f"[resnet-v2] epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.4f} val_acc={row['val_acc']:.4f} "
            f"macro={row['val_macro_acc']:.4f} lr={row['lr_backbone']:.2e}/{row['lr_head']:.2e}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strong ResNet baseline for PartImageNet object-label classification.")
    p.add_argument("--config", default="configs/stage1_quality_upgrade.yaml")
    p.add_argument("--partimagenet-root", default="")
    p.add_argument("--save-dir", default="runs/resnet50_partimagenet_v2")
    p.add_argument("--model", default="resnet50", choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    p.add_argument("--imagenet-pretrained", action="store_true")
    p.add_argument("--resume", default="")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--backbone-lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--freeze-epochs", type=int, default=1)
    p.add_argument("--freeze-bn", action="store_true")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--balanced-sampler", action="store_true")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--preprocess", choices=["imagenet", "repo"], default="imagenet")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--val-resize", type=int, default=256)
    p.add_argument("--randaugment", action="store_true")
    p.add_argument("--random-erasing", type=float, default=0.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
