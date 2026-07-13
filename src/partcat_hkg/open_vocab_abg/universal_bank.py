from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from partcat_hkg.abg_aog_v7.multislot_native import MultiSlotBankV7, semantic_subparts_for_part

from .text_encoder import DynamicTextQueryEncoderV7
from .types import OpenVocabQueryKindV7, OpenVocabTextQueryV7


@dataclass
class UniversalPartPrototypeV7:
    part_id: int
    name: str
    text_embedding: list[float]
    visual_prototype: list[float]
    token_support: int
    occurrence_rate: float
    multiplicity_prob: list[float]
    geometry_mean: list[float]
    geometry_var: list[float]
    semantic_subparts: list[str]
    compatible_roles: list[str] = field(default_factory=list)


@dataclass
class UniversalRelationPrimitiveV7:
    relation_type: str
    description: str
    embedding: list[float]
    default_mean: list[float]
    default_var: list[float]


@dataclass
class UniversalMotifV7:
    motif_id: str
    name: str
    part_names: list[str]
    role_names: list[str]
    prior: float
    complexity_cost: float
    source_classes: list[int] = field(default_factory=list)


@dataclass
class KnownGrammarDescriptorV7:
    class_id: int
    class_name: str
    object_embedding: list[float]
    slots: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    poses: list[dict[str, Any]]
    motifs: list[dict[str, Any]]
    part_histogram: dict[str, float]
    support: int = 0


@dataclass
class UniversalStructuralBankV7:
    text_dim: int
    universal_parts: list[UniversalPartPrototypeV7]
    relation_primitives: list[UniversalRelationPrimitiveV7]
    universal_motifs: list[UniversalMotifV7]
    known_grammars: list[KnownGrammarDescriptorV7]
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.part_by_name = {p.name.lower(): p for p in self.universal_parts}
        self.grammar_by_class = {int(g.class_id): g for g in self.known_grammars}
        self.grammar_by_name = {g.class_name.lower(): g for g in self.known_grammars}
        self.motif_by_id = {m.motif_id: m for m in self.universal_motifs}
        self.relation_by_type = {r.relation_type: r for r in self.relation_primitives}

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "open_vocab_universal_structural_bank_v7",
            "text_dim": int(self.text_dim),
            "universal_parts": [asdict(x) for x in self.universal_parts],
            "relation_primitives": [asdict(x) for x in self.relation_primitives],
            "universal_motifs": [asdict(x) for x in self.universal_motifs],
            "known_grammars": [asdict(x) for x in self.known_grammars],
            "config": dict(self.config),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "UniversalStructuralBankV7":
        return cls(
            text_dim=int(payload["text_dim"]),
            universal_parts=[UniversalPartPrototypeV7(**x) for x in payload.get("universal_parts", [])],
            relation_primitives=[UniversalRelationPrimitiveV7(**x) for x in payload.get("relation_primitives", [])],
            universal_motifs=[UniversalMotifV7(**x) for x in payload.get("universal_motifs", [])],
            known_grammars=[KnownGrammarDescriptorV7(**x) for x in payload.get("known_grammars", [])],
            config=dict(payload.get("config", {})),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "UniversalStructuralBankV7":
        return cls.from_payload(torch.load(Path(path), map_location=map_location))


def _mean_var(rows: list[list[float]], dim: int = 4) -> tuple[list[float], list[float]]:
    if not rows:
        return [0.5, 0.5, 0.25, 0.25][:dim], [0.10] * dim
    tensor = torch.tensor(rows, dtype=torch.float32)
    return [float(x) for x in tensor.mean(0).tolist()], [float(x) for x in tensor.var(0, unbiased=False).clamp_min(0.01).tolist()]


def _role_from_geometry(mean: list[float], slot_index: int, count: int) -> str:
    cx, cy = float(mean[0]), float(mean[1])
    horizontal = "left" if cx < 0.40 else ("right" if cx > 0.60 else "center")
    vertical = "upper" if cy < 0.40 else ("lower" if cy > 0.60 else "middle")
    if count > 1:
        return f"{horizontal}-{vertical}-{slot_index}"
    return f"{horizontal}-{vertical}"


def _default_relation_primitives(text_encoder: DynamicTextQueryEncoderV7) -> list[UniversalRelationPrimitiveV7]:
    specs = [
        ("above", "one part lies above another", [1, 0, 0, 0, 0, 0, 0, 0]),
        ("below", "one part lies below another", [0, 1, 0, 0, 0, 0, 0, 0]),
        ("lateral", "two parts lie side by side", [0, 0, 1, 0, 0, 0, 0, 0]),
        ("near", "two parts are spatially near", [0, 0, 0, 1, 0, 0, 0, 0]),
        ("touching", "two parts touch or attach", [0, 0, 0, 0, 1, 0, 0, 0]),
        ("overlap", "two parts overlap", [0, 0, 0, 0, 0, 1, 0, 0]),
        ("contains", "one part contains another", [0, 0, 0, 0, 0, 0, 1, 0]),
        ("inside", "one part is inside another", [0, 0, 0, 0, 0, 0, 0, 1]),
        ("symmetric_pair", "a bilateral symmetric pair of parts", [0, 0, 1, 1, 0, 0, 0, 0]),
        ("collinear_axis", "parts lie along a common axis", [1, 1, 0, 1, 0, 0, 0, 0]),
        ("radial_around", "parts are arranged radially around a center", [0, 0, 1, 1, 0, 0, 0, 1]),
    ]
    out: list[UniversalRelationPrimitiveV7] = []
    for i, (name, description, mean) in enumerate(specs):
        query = OpenVocabTextQueryV7(9_000_000 + i, description, OpenVocabQueryKindV7.ROLE, role_text=name)
        embedding = text_encoder.encode_query(query).cpu().tolist()
        out.append(UniversalRelationPrimitiveV7(name, description, [float(x) for x in embedding], [float(x) for x in mean], [0.20] * len(mean)))
    return out


def build_universal_structural_bank_v7(
    bank: MultiSlotBankV7,
    text_encoder: DynamicTextQueryEncoderV7,
    *,
    pose_bank: Any | None = None,
    max_multiplicity: int = 6,
) -> UniversalStructuralBankV7:
    part_slots: dict[int, list[Any]] = defaultdict(list)
    class_part_count: dict[tuple[int, int], int] = Counter()
    for slot in bank.slots:
        part_slots[int(slot.part_id)].append(slot)
        class_part_count[(int(slot.class_id), int(slot.part_id))] += 1
    universal_parts: list[UniversalPartPrototypeV7] = []
    for part_id, part_name in enumerate(bank.part_names):
        rows = part_slots.get(part_id, [])
        query = OpenVocabTextQueryV7(part_id, part_name, OpenVocabQueryKindV7.PART)
        text_embedding = text_encoder.encode_query(query).cpu()
        shared = bank.shared_part_by_id.get(part_id)
        visual = list(shared.token_mean) if shared is not None else []
        token_support = int(shared.token_support) if shared is not None else 0
        occurrence = float(shared.rate) if shared is not None else float(bank.global_part_rate.get(part_id, 0.0))
        multiplicity = torch.ones(max_multiplicity + 1, dtype=torch.float32) * 1e-3
        for class_id in range(len(bank.class_names)):
            count = min(max_multiplicity, int(class_part_count.get((class_id, part_id), 0)))
            multiplicity[count] += 1.0
        multiplicity = multiplicity / multiplicity.sum()
        geom_mean, geom_var = _mean_var([list(s.geom_mean) for s in rows])
        universal_parts.append(UniversalPartPrototypeV7(
            part_id=part_id,
            name=part_name,
            text_embedding=[float(x) for x in text_embedding.tolist()],
            visual_prototype=[float(x) for x in visual],
            token_support=token_support,
            occurrence_rate=occurrence,
            multiplicity_prob=[float(x) for x in multiplicity.tolist()],
            geometry_mean=geom_mean,
            geometry_var=geom_var,
            semantic_subparts=semantic_subparts_for_part(part_name),
            compatible_roles=["front", "rear", "left", "right", "upper", "lower", "center", "attachment", "contact"],
        ))
    poses_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if pose_bank is not None:
        raw = getattr(pose_bank, "poses", {})
        for class_id, poses in raw.items():
            poses_by_class[int(class_id)] = [asdict(p) if hasattr(p, "__dataclass_fields__") else dict(p) for p in poses]
    motifs_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in bank.cfg.get("pursued_blocks", []):
        item = dict(raw)
        motifs_by_class[int(item.get("class_id", -1))].append(item)
    known_grammars: list[KnownGrammarDescriptorV7] = []
    for class_id, class_name in enumerate(bank.class_names):
        object_query = OpenVocabTextQueryV7(class_id, class_name, OpenVocabQueryKindV7.OBJECT)
        object_embedding = text_encoder.encode_query(object_query).cpu().tolist()
        slots = []
        class_slots = list(bank.by_class.get(class_id, []))
        counts = Counter(int(s.part_id) for s in class_slots)
        for slot in class_slots:
            payload = asdict(slot)
            payload["role_text"] = _role_from_geometry(list(slot.geom_mean), int(slot.slot_id), int(counts[int(slot.part_id)]))
            slots.append(payload)
        relations = [asdict(r) for r in bank.relations_by_class.get(class_id, [])]
        histogram = Counter(s.part_name for s in class_slots)
        total = max(1, sum(histogram.values()))
        known_grammars.append(KnownGrammarDescriptorV7(
            class_id=class_id,
            class_name=class_name,
            object_embedding=[float(x) for x in object_embedding],
            slots=slots,
            relations=relations,
            poses=poses_by_class.get(class_id, []),
            motifs=motifs_by_class.get(class_id, []),
            part_histogram={str(k): float(v / total) for k, v in histogram.items()},
            support=max((int(s.support_images) for s in class_slots), default=0),
        ))
    motif_groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for grammar in known_grammars:
        for motif in grammar.motifs:
            slot_uids = set(int(x) for x in motif.get("slot_uids", []))
            names = tuple(sorted(str(s.get("part_name", "part")) for s in grammar.slots if int(s.get("slot_uid", -1)) in slot_uids))
            if names:
                motif_groups[names].append(int(grammar.class_id))
    universal_motifs = [
        UniversalMotifV7(
            motif_id=f"motif_{i}",
            name="-".join(names),
            part_names=list(names),
            role_names=[],
            prior=float(len(classes) / max(1, len(known_grammars))),
            complexity_cost=0.03 * len(names),
            source_classes=sorted(set(classes)),
        )
        for i, (names, classes) in enumerate(sorted(motif_groups.items()))
    ]
    return UniversalStructuralBankV7(
        text_dim=text_encoder.dim,
        universal_parts=universal_parts,
        relation_primitives=_default_relation_primitives(text_encoder),
        universal_motifs=universal_motifs,
        known_grammars=known_grammars,
        config={"max_multiplicity": int(max_multiplicity), "text_backend": text_encoder.status.backend},
    )
