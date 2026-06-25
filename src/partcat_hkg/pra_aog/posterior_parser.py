from __future__ import annotations

from typing import Any

import torch

from .parser import PRAAOGParser as _StrictBackedPRAAOGParser
from .runtime import (
    canonical_box_to_image,
    prepare_batch_for_parser,
    preprocess_config_from_metadata,
)


class PRAAOGParser(_StrictBackedPRAAOGParser):
    """Posterior parser with runtime/build coordinate consistency."""

    def __init__(self, grammar_or_bundle, strict_cfg=None, cfg=None) -> None:
        super().__init__(grammar_or_bundle, strict_cfg, cfg)
        self.runtime_preprocess_cfg = preprocess_config_from_metadata(
            getattr(self.bundle, "metadata", {})
        )

    def forward(
        self,
        batch: dict[str, Any],
        *,
        enable_edges: bool = True,
        return_forest: bool = False,
        return_readouts: bool = False,
    ) -> dict[str, Any]:
        prepared = prepare_batch_for_parser(
            batch,
            part_names=list(self.grammar.part_names),
            cfg=self.runtime_preprocess_cfg,
        )
        parser_batch = self._class_agnostic_batch(prepared)
        out = super().forward(
            parser_batch,
            enable_edges=enable_edges,
            return_forest=False,
            return_readouts=False,
        )
        joint = out["parse_joint_scores"]
        batch_size, classes, templates = joint.shape
        temperature = max(float(self.pra_cfg.posterior_tau), 1e-6)

        class_log_posterior = torch.log_softmax(out["logits"], dim=-1)
        template_log_posterior = torch.log_softmax(joint / temperature, dim=-1)
        full_log_posterior = (
            class_log_posterior[:, :, None] + template_log_posterior
        ).reshape(batch_size, classes * templates)
        full_posterior = full_log_posterior.exp()
        entropy = -torch.where(
            full_posterior > 0,
            full_posterior * full_log_posterior,
            torch.zeros_like(full_posterior),
        ).sum(-1)

        top_k = max(1, min(int(self.pra_cfg.top_k), classes * templates))
        top_probability, top_flat = torch.topk(full_posterior, k=top_k, dim=-1)
        retained_mass = top_probability.sum(-1)
        top_weight = top_probability / retained_mass[:, None].clamp_min(1e-12)
        top_class = torch.div(top_flat, templates, rounding_mode="floor")
        top_template = top_flat.remainder(templates)

        out.update(
            {
                "structural_class_posterior": out.get(
                    "structural_class_posterior", out["class_posterior"]
                ),
                "class_posterior": class_log_posterior.exp(),
                "calibrated_class_posterior": class_log_posterior.exp(),
                "parse_class": top_class,
                "parse_template": top_template,
                "parse_log_score": joint.reshape(batch_size, -1).gather(1, top_flat),
                "parse_posterior": top_weight,
                "parse_unconditional_posterior": top_probability,
                "parse_retained_mass": retained_mass,
                "parse_entropy": entropy,
            }
        )

        if return_forest or return_readouts:
            forests = self.decode_parse_forest(
                parser_batch,
                out,
                enable_edges=enable_edges,
            )
            out["parse_forest"] = forests
            query_rows = []
            frames = prepared.get("pra_object_frame")
            flags = prepared.get("pra_reflected")
            for batch_index, forest in enumerate(forests):
                frame = (
                    frames[batch_index]
                    if torch.is_tensor(frames)
                    else (0.0, 0.0, 1.0, 1.0)
                )
                reflected = bool(flags[batch_index].item()) if torch.is_tensor(flags) else False
                row = []
                for query in self.propose_topdown_queries(forest):
                    item = query.to_dict()
                    item["box_xyxy"] = canonical_box_to_image(
                        query.box_xyxy,
                        frame,
                        reflected=reflected,
                    )
                    item["coordinate_frame"] = "image_normalized"
                    row.append(item)
                query_rows.append(row)
            out["topdown_queries"] = query_rows
            if return_readouts:
                from .readouts import posterior_readouts

                out["readouts"] = posterior_readouts(
                    forests,
                    batch=prepared,
                    num_parts=len(self.grammar.part_names),
                    num_classes=self.grammar.num_classes,
                )
        return out
