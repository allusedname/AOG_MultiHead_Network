from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GraphFeedbackConfig:
    temperature: float = 1.0
    min_prior: float = 0.02
    sharpen: float = 1.0


class PRAAOGGraphPrior:
    """Top-down part prior extracted from a PRA-AOG bundle.

    The prior is intentionally coarse: it converts class posterior mass into an
    expected part-presence vector using the learned object grammar.  Stage 1 then
    uses this prior as a refinement bias, not as visible-mask ground truth.
    """

    def __init__(self, bundle, cfg: GraphFeedbackConfig | None = None) -> None:
        self.bundle = bundle
        self.grammar = bundle.grammar
        self.cfg = cfg or GraphFeedbackConfig()
        self.class_part_presence = self._class_part_presence()

    def _class_part_presence(self) -> torch.Tensor:
        grammar = self.grammar
        out = torch.zeros(grammar.num_classes, len(grammar.part_names), dtype=torch.float32)
        template_prior = grammar.template_prior.float() * grammar.template_valid.float()
        template_prior = template_prior / template_prior.sum(-1, keepdim=True).clamp_min(1e-8)
        for class_id in range(grammar.num_classes):
            for template_id in range(grammar.num_templates):
                weight = float(template_prior[class_id, template_id].item())
                if weight <= 0:
                    continue
                for slot in range(grammar.max_slots):
                    if float(grammar.slot_valid[class_id, template_id, slot].item()) <= 0.5:
                        continue
                    part_id = int(grammar.slot_part[class_id, template_id, slot].item())
                    if 0 <= part_id < out.shape[1]:
                        out[class_id, part_id] += weight * float(grammar.slot_presence[class_id, template_id, slot].item())
        return out.clamp(0, 1)

    def from_logits(
        self,
        obj_logits: torch.Tensor,
        *,
        image_hw: tuple[int, int] | None = None,
        subpart_to_part: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        class_prob = F.softmax(obj_logits.float() / max(float(self.cfg.temperature), 1e-6), dim=1)
        part_prior = class_prob @ self.class_part_presence.to(obj_logits.device)
        part_prior = part_prior.clamp_min(float(self.cfg.min_prior)).clamp_max(1.0)
        if float(self.cfg.sharpen) != 1.0:
            part_prior = part_prior.pow(float(self.cfg.sharpen)).clamp(0, 1)
        out: dict[str, torch.Tensor] = {"part_prior": part_prior}
        if subpart_to_part is not None:
            out["subpart_prior"] = part_prior[:, subpart_to_part.to(obj_logits.device).long()]
        if image_hw is not None:
            h, w = int(image_hw[0]), int(image_hw[1])
            out = {k: v[:, :, None, None].expand(-1, -1, h, w) for k, v in out.items()}
        return out


def two_pass_hierarchical_stage1(model, image: torch.Tensor, prior_builder: PRAAOGGraphPrior):
    """Run bottom-up Stage 1, build an AOG prior, then refine Stage 1."""

    first = model(image, graph_prior=None)
    prior = prior_builder.from_logits(
        first["obj_logits"].mean(dim=(-2, -1)),
        subpart_to_part=getattr(model, "subpart_to_part", None),
    )
    second = model(image, graph_prior=prior)
    second["bottomup_first_pass_part_logits"] = first["part_logits"]
    second["bottomup_first_pass_subpart_logits"] = first.get("subpart_logits")
    return second
