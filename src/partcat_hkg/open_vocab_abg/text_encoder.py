from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import OpenVocabQueryBatchV7, OpenVocabQueryKindV7, OpenVocabTextQueryV7


@dataclass
class TextEncoderStatusV7:
    backend: str
    semantic: bool
    model_name: str
    pretrained: str
    dim: int


def _stable_fallback_embedding(text: str, dim: int) -> torch.Tensor:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31 - 1)
    gen = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(dim, generator=gen), dim=0)


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = " ".join(str(item).split())
        if cleaned and cleaned not in seen:
            out.append(cleaned)
            seen.add(cleaned)
    return out


def prompts_for_query(query: OpenVocabTextQueryV7) -> list[str]:
    text = str(query.text).strip()
    if query.kind is OpenVocabQueryKindV7.OBJECT:
        return _dedupe([
            f"a photo of a {text}",
            f"the {text} in the image",
            f"a close-up photo of a {text}",
            f"an object of type {text}",
        ])
    if query.kind is OpenVocabQueryKindV7.PART:
        obj = query.object_text
        base = [
            f"a photo of a {text}",
            f"a photo of the {text}",
            f"the {text} of an object",
            f"a close-up of the {text}",
        ]
        if obj:
            base.extend([
                f"the {text} of a {obj}",
                f"a close-up of the {text} of a {obj}",
                f"{obj} {text}",
            ])
        return _dedupe(base)
    if query.kind is OpenVocabQueryKindV7.ROLE:
        obj = query.object_text or "object"
        part = query.part_text or text
        role = query.role_text or text
        return _dedupe([
            f"the {role} of a {obj}",
            f"the {part} serving as {role} on a {obj}",
            f"a close-up of the {role}",
            f"{obj} {role}",
        ])
    return _dedupe(["an unknown object", "an unfamiliar object", text or "unknown object"])


class DynamicTextQueryEncoderV7(nn.Module):
    """Runtime CLIP text encoder with prompt ensembling and deterministic fallback.

    The query count is dynamic. Encoded queries are cached on CPU by prompt tuple.
    When ``require_semantic=True`` the deterministic fallback is rejected, which
    prevents a performance run from silently using random pseudo-semantics.
    """

    def __init__(
        self,
        *,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        enabled: bool = True,
        require_semantic: bool = False,
        fallback_dim: int = 512,
        cache_size: int = 4096,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)
        self.enabled = bool(enabled)
        self.require_semantic = bool(require_semantic)
        self.fallback_dim = int(fallback_dim)
        self.cache_size = int(cache_size)
        self.model: nn.Module | None = None
        self.tokenizer = None
        self._cache: OrderedDict[tuple[str, ...], torch.Tensor] = OrderedDict()
        self.status = TextEncoderStatusV7("fallback", False, self.model_name, self.pretrained, self.fallback_dim)
        if self.enabled:
            try:
                import open_clip

                model, _, _ = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
                tokenizer = open_clip.get_tokenizer(self.model_name)
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                with torch.no_grad():
                    probe = tokenizer(["a photo of an object"])
                    embedding = model.encode_text(probe)
                self.model = model
                self.tokenizer = tokenizer
                self.status = TextEncoderStatusV7(
                    backend="open_clip",
                    semantic=True,
                    model_name=self.model_name,
                    pretrained=self.pretrained,
                    dim=int(embedding.shape[-1]),
                )
            except Exception as exc:  # pragma: no cover - optional package/cache
                self.status = TextEncoderStatusV7(
                    backend=f"fallback:{type(exc).__name__}",
                    semantic=False,
                    model_name=self.model_name,
                    pretrained=self.pretrained,
                    dim=self.fallback_dim,
                )
        if self.require_semantic and not self.status.semantic:
            raise RuntimeError(
                "DynamicTextQueryEncoderV7 requires a semantic text backend, but "
                f"OpenCLIP was unavailable ({self.status.backend})."
            )

    @property
    def dim(self) -> int:
        return int(self.status.dim)

    def _cache_put(self, key: tuple[str, ...], value: torch.Tensor) -> None:
        self._cache[key] = value.detach().cpu()
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def encode_prompts(self, prompts: list[str], *, device: torch.device | str | None = None) -> torch.Tensor:
        key = tuple(_dedupe(prompts))
        if not key:
            key = ("an unknown object",)
        if key in self._cache:
            value = self._cache[key]
            self._cache.move_to_end(key)
            return value.to(device=device) if device is not None else value.clone()
        if self.model is not None and self.tokenizer is not None:
            model_device = next(self.model.parameters()).device
            with torch.no_grad():
                tokens = self.tokenizer(list(key)).to(model_device)
                embedding = self.model.encode_text(tokens).float()
                embedding = F.normalize(embedding, dim=-1)
                value = F.normalize(embedding.mean(0), dim=0).cpu()
        else:
            value = F.normalize(torch.stack([_stable_fallback_embedding(p, self.dim) for p in key]).mean(0), dim=0)
        self._cache_put(key, value)
        return value.to(device=device) if device is not None else value.clone()

    def encode_query(self, query: OpenVocabTextQueryV7, *, device: torch.device | str | None = None) -> torch.Tensor:
        if query.embedding is not None:
            return F.normalize(query.embedding.float().flatten(), dim=0).to(device=device)
        embedding = self.encode_prompts(prompts_for_query(query), device=device)
        query.embedding = embedding.detach().cpu()
        return embedding

    def encode_queries(self, queries: list[OpenVocabTextQueryV7], *, device: torch.device | str | None = None) -> torch.Tensor:
        if not queries:
            return torch.empty(0, self.dim, device=device)
        return torch.stack([self.encode_query(q, device=device) for q in queries], dim=0)

    def encode_batch(self, batch: OpenVocabQueryBatchV7, *, device: torch.device | str | None = None) -> torch.Tensor:
        return self.encode_queries(batch.all_queries(), device=device)
