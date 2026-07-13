from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


OPEN_VOCAB_FEATURES_V7 = [
    "slot_presence",
    "slot_absence",
    "slot_token",
    "slot_geom",
    "part_token",
    "shared_part_token",
    "part_coverage",
    "missing",
    "weak_missing",
    "extra_unassigned",
    "relation",
    "motif_coverage",
    "motif_violation",
    "matched_slots",
    "missing_slots",
    "active_terms",
    "coverage",
]

# +1: non-negative, -1: non-positive, 0: unconstrained.
FEATURE_SIGNS_V7 = {
    "slot_presence": 1,
    "slot_absence": 0,
    "slot_token": 1,
    "slot_geom": 1,
    "part_token": 1,
    "shared_part_token": 1,
    "part_coverage": 1,
    "missing": -1,
    "weak_missing": -1,
    "extra_unassigned": -1,
    "relation": 1,
    "motif_coverage": 1,
    "motif_violation": -1,
    "matched_slots": 1,
    "missing_slots": -1,
    "active_terms": 0,
    "coverage": 1,
}


@dataclass
class CalibratorTrainConfigV7:
    epochs: int = 400
    lr: float = 0.03
    weight_decay: float = 1e-3
    query_rank: int = 8
    max_query_delta: float = 0.75
    device: str = "cuda"


class OpenVocabCalibratorV7(nn.Module):
    """Class-agnostic monotonic scorer with a low-rank object-query adapter.

    There is no learned per-class bias. The same physical score semantics apply
    to seen and unseen object text queries. The low-rank adapter may modulate
    feature magnitudes from the object embedding while preserving sign constraints.
    """

    def __init__(
        self,
        text_dim: int,
        *,
        feature_names: list[str] | None = None,
        query_rank: int = 8,
        max_query_delta: float = 0.75,
    ) -> None:
        super().__init__()
        self.text_dim = int(text_dim)
        self.feature_names = list(feature_names or OPEN_VOCAB_FEATURES_V7)
        self.query_rank = int(query_rank)
        self.max_query_delta = float(max_query_delta)
        self.raw_weight = nn.Parameter(torch.zeros(len(self.feature_names)))
        self.global_bias = nn.Parameter(torch.zeros(()))
        self.query_projection = nn.Linear(self.text_dim, self.query_rank, bias=False)
        self.query_basis = nn.Parameter(torch.zeros(self.query_rank, len(self.feature_names)))
        nn.init.normal_(self.query_basis, std=0.01)
        signs = torch.tensor([int(FEATURE_SIGNS_V7.get(name, 0)) for name in self.feature_names], dtype=torch.float32)
        self.register_buffer("feature_signs", signs)
        self.register_buffer("feature_mean", torch.zeros(len(self.feature_names)))
        self.register_buffer("feature_std", torch.ones(len(self.feature_names)))

    def fit_standardization(self, features: torch.Tensor) -> None:
        if features.shape[-1] != len(self.feature_names):
            raise ValueError("feature dimension does not match calibrator")
        flat = features.detach().float().reshape(-1, features.shape[-1])
        self.feature_mean.copy_(flat.mean(0))
        self.feature_std.copy_(flat.std(0).clamp_min(1e-4))

    def effective_weights(self, object_embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(object_embeddings.float(), dim=-1)
        query_code = torch.tanh(self.query_projection(embeddings))
        delta = self.max_query_delta * torch.tanh(query_code @ self.query_basis)
        parameter = self.raw_weight + delta
        positive = F.softplus(parameter)
        negative = -F.softplus(parameter)
        unconstrained = parameter
        signs = self.feature_signs.to(parameter)
        return torch.where(signs > 0, positive, torch.where(signs < 0, negative, unconstrained))

    def forward(self, features: torch.Tensor, object_embeddings: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != len(self.feature_names):
            raise ValueError("feature dimension does not match calibrator")
        standardized = (features.float() - self.feature_mean) / self.feature_std.clamp_min(1e-4)
        weights = self.effective_weights(object_embeddings)
        return (standardized * weights).sum(-1) + self.global_bias

    def feature_tensor(self, feature_dict: dict[str, float], *, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.tensor([float(feature_dict.get(name, 0.0)) for name in self.feature_names], dtype=torch.float32, device=device)

    @torch.no_grad()
    def score_dict(self, feature_dict: dict[str, float], object_embedding: torch.Tensor) -> float:
        device = next(self.parameters()).device
        features = self.feature_tensor(feature_dict, device=device).unsqueeze(0)
        embedding = object_embedding.detach().float().flatten().to(device).unsqueeze(0)
        return float(self.forward(features, embedding)[0].cpu().item())

    def checkpoint(self) -> dict:
        return {
            "kind": "open_vocab_calibrator_v7",
            "text_dim": self.text_dim,
            "feature_names": list(self.feature_names),
            "query_rank": self.query_rank,
            "max_query_delta": self.max_query_delta,
            "state_dict": self.state_dict(),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "OpenVocabCalibratorV7":
        payload = torch.load(Path(path), map_location=map_location)
        model = cls(
            int(payload["text_dim"]),
            feature_names=list(payload["feature_names"]),
            query_rank=int(payload.get("query_rank", 8)),
            max_query_delta=float(payload.get("max_query_delta", 0.75)),
        )
        model.load_state_dict(payload["state_dict"])
        return model


def train_open_vocab_calibrator_v7(
    model: OpenVocabCalibratorV7,
    features: torch.Tensor,
    object_embeddings: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    cfg: CalibratorTrainConfigV7 | None = None,
) -> list[dict[str, float]]:
    """Train on candidate sets shaped [N,C,F], [N,C,D], and targets [N]."""
    cfg = cfg or CalibratorTrainConfigV7(query_rank=model.query_rank, max_query_delta=model.max_query_delta)
    device = torch.device(cfg.device if torch.cuda.is_available() and str(cfg.device).startswith("cuda") else "cpu")
    model.to(device)
    features = features.to(device).float()
    object_embeddings = object_embeddings.to(device).float()
    target_indices = target_indices.to(device).long()
    model.fit_standardization(features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    logs: list[dict[str, float]] = []
    for epoch in range(int(cfg.epochs)):
        logits = model(features, object_embeddings)
        loss = F.cross_entropy(logits, target_indices)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == int(cfg.epochs) - 1:
            accuracy = float((logits.argmax(-1) == target_indices).float().mean().detach().cpu().item())
            logs.append({"epoch": float(epoch), "loss": float(loss.detach().cpu().item()), "accuracy": accuracy})
    return logs
