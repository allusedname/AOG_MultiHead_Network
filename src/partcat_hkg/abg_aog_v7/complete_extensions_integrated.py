from __future__ import annotations

from typing import Any

import torch

from .complete_extensions import *  # noqa: F401,F403
from .complete_extensions import PoseAwareNativeMultiSlotParserV7 as _BasePoseAwareNativeMultiSlotParserV7
from .multislot_native import MultiSlotBankV7, NativeMultiSlotParserV7, geom4
from .types import ParseForestV7, ParseHypothesisV7, TerminalPacketV7


class PoseAwareNativeMultiSlotParserV7(_BasePoseAwareNativeMultiSlotParserV7):
    """Pose-aware parser with actual hypothesis geometry features.

    The first implementation stored pose clusters but could not score a hypothesis
    by pose because SlotAssignmentV7 does not itself store boxes.  This override
    keeps the terminal list during parse and reconstructs the pose feature from
    matched terminal boxes in the canonical slot order.
    """

    def parse(self, terminals: list[TerminalPacketV7], *, sample_id: int = 0) -> ParseForestV7:
        self._terminal_by_id = {int(t.terminal_id): t for t in terminals}
        base = NativeMultiSlotParserV7.parse(self, terminals, sample_id=sample_id)
        out: list[ParseHypothesisV7] = []
        for h in base.hypotheses:
            if h.class_id is None:
                continue
            poses = self.pose_bank.poses.get(int(h.class_id), [])
            if not poses:
                out.append(h)
                continue
            feat = self._hyp_pose_feature(h)
            for p in poses:
                bonus = float(torch.log(torch.tensor(max(float(p.prior), 1e-6))).item())
                if feat is not None and p.geom_mean:
                    mu = torch.tensor(p.geom_mean, dtype=torch.float32)
                    var = torch.tensor(p.geom_var, dtype=torch.float32).clamp_min(1e-3)
                    if mu.numel() == feat.numel():
                        bonus += float(torch.exp(-0.5 * torch.clamp((((feat - mu) ** 2) / var).mean(), max=4.0)).item())
                out.append(ParseHypothesisV7(hypothesis_id=len(out), root_node_id=h.root_node_id, score=float(h.score + bonus), class_id=h.class_id, pose_template_id=int(p.pose_id), slots=h.slots, terminal_ids=h.terminal_ids, relation_scores=h.relation_scores + [{"pose_bonus": bonus, "pose_id": int(p.pose_id)}], scene_objects=h.scene_objects))
        out.sort(key=lambda x: x.score, reverse=True)
        return ParseForestV7(hypotheses=out[: self.top_k]).normalize_posteriors()

    def _hyp_pose_feature(self, h: ParseHypothesisV7) -> torch.Tensor | None:
        if h.class_id is None:
            return None
        slots = sorted(self.bank.by_class.get(int(h.class_id), []), key=lambda s: int(s.slot_uid))
        assign_by_uid = {int(s.slot_id): s for s in h.slots}
        vals: list[float] = []
        for slot in slots:
            a = assign_by_uid.get(int(slot.slot_uid))
            if a is None or a.terminal_id is None or int(a.terminal_id) not in self._terminal_by_id:
                vals.extend([0.0, 0.0, 0.0, 0.0])
            else:
                vals.extend([float(x) for x in geom4(self._terminal_by_id[int(a.terminal_id)].visible_box_xyxy).tolist()])
        return torch.tensor(vals, dtype=torch.float32) if vals else None


__all__ = [name for name in globals() if not name.startswith('_')]
