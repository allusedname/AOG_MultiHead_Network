from __future__ import annotations

from typing import Any

from partcat_hkg.abg_aog_v7.grammar import NativeGrammarV7
from partcat_hkg.abg_aog_v7.types import NodeKindV7, RuleKindV7

from .types import DynamicGrammarSpecV7


def materialize_dynamic_grammars_v7(
    grammars: list[DynamicGrammarSpecV7],
    *,
    allow_absent: bool = True,
) -> tuple[NativeGrammarV7, dict[int, int]]:
    """Convert provisional runtime grammar specs into an auditable native AOG.

    The returned mapping gives object-query id -> object-query node id. Runtime
    query names remain attributes; no fixed training class index is required.
    """
    grammar = NativeGrammarV7(root_id=0, class_names=[g.object_query.text for g in grammars], part_names=[])
    root = grammar.add_node(NodeKindV7.OR, "open_vocab_root", "open_vocab_root")
    grammar.root_id = root
    query_node_map: dict[int, int] = {}
    for local_class_id, spec in enumerate(grammars):
        query = spec.object_query
        object_node = grammar.add_node(
            NodeKindV7.AND,
            "open_vocab_object_query",
            query.text,
            attributes={
                "runtime_query_id": int(query.query_id),
                "runtime_text": query.text,
                "stage1_object_score": float(query.stage1_object_score),
                "is_unknown": bool(spec.is_unknown),
                "retrieval_scores": dict(spec.retrieval_scores),
                "provenance": dict(spec.provenance),
                "class_id": int(local_class_id),
            },
            complexity_cost=float(spec.complexity_cost),
        )
        query_node_map[int(query.query_id)] = object_node
        grammar.add_rule(root, [object_node], kind=RuleKindV7.OR_SELECT, branch_prior=max(1e-4, float(query.stage1_object_score) if query.stage1_object_score > 0 else 1.0 / max(1, len(grammars))))

        slot_nodes: dict[str, int] = {}
        for slot in spec.slots:
            slot_or = grammar.add_node(
                NodeKindV7.OR,
                "dynamic_functional_slot",
                f"{query.text}:{slot.role_text}:{slot.part_text}",
                attributes={
                    "runtime_query_id": int(query.query_id),
                    "slot_uid": slot.slot_uid,
                    "part_text": slot.part_text,
                    "part_query_id": int(slot.part_query_id),
                    "role_text": slot.role_text,
                    "multiplicity_index": int(slot.multiplicity_index),
                    "occurrence_prior": float(slot.occurrence_prior),
                    "requiredness": float(slot.requiredness_prior),
                    "geometry_mean": list(slot.geometry_mean),
                    "geometry_var": list(slot.geometry_var),
                    "source": slot.source,
                },
                complexity_cost=0.01,
            )
            template = grammar.add_node(
                NodeKindV7.AND,
                "dynamic_slot_template",
                f"template:{slot.slot_uid}",
                attributes={
                    "slot_uid": slot.slot_uid,
                    "part_text": slot.part_text,
                    "role_text": slot.role_text,
                    "geometry_mean": list(slot.geometry_mean),
                    "geometry_var": list(slot.geometry_var),
                },
                complexity_cost=0.01,
            )
            terminal = grammar.add_node(
                NodeKindV7.TERMINAL,
                "open_vocab_terminal",
                f"terminal:{slot.slot_uid}",
                attributes={
                    "slot_uid": slot.slot_uid,
                    "part_text": slot.part_text,
                    "functional_part_id": int(slot.part_query_id),
                    "role_text": slot.role_text,
                    "requiredness": float(slot.requiredness_prior),
                    "template_geom_mean": list(slot.geometry_mean),
                    "template_geom_var": list(slot.geometry_var),
                    "allow_absent": False,
                },
            )
            grammar.add_rule(slot_or, [template], kind=RuleKindV7.OR_SELECT, branch_prior=max(1e-4, float(slot.occurrence_prior)))
            grammar.add_rule(template, [terminal], kind=RuleKindV7.AND_COMPOSE)
            if allow_absent:
                absent = grammar.add_node(
                    NodeKindV7.TERMINAL,
                    "open_vocab_absent_terminal",
                    f"absent:{slot.slot_uid}",
                    attributes={
                        "slot_uid": slot.slot_uid,
                        "part_text": slot.part_text,
                        "functional_part_id": int(slot.part_query_id),
                        "requiredness": float(slot.requiredness_prior),
                        "allow_absent": True,
                    },
                )
                grammar.add_rule(slot_or, [absent], kind=RuleKindV7.OR_SELECT, branch_prior=max(1e-4, 1.0 - float(slot.occurrence_prior)), complexity_cost=0.005)
            slot_nodes[slot.slot_uid] = slot_or

        relation_ids: list[int] = []
        for relation in spec.relations:
            if relation.source_slot_uid not in slot_nodes or relation.target_slot_uid not in slot_nodes:
                continue
            relation_ids.append(grammar.add_relation(
                slot_nodes[relation.source_slot_uid],
                slot_nodes[relation.target_slot_uid],
                relation.relation_type,
                mean=tuple(float(x) for x in relation.mean),
                var=tuple(float(x) for x in relation.var),
                weight=float(relation.reliability),
                support=int(relation.support),
                reliability=float(relation.reliability),
            ))

        motif_nodes: dict[str, int] = {}
        covered_slots: set[str] = set()
        for motif in spec.motifs:
            children = [slot_nodes[uid] for uid in motif.slot_uids if uid in slot_nodes]
            if len(children) < 2 or any(uid in covered_slots for uid in motif.slot_uids):
                continue
            motif_node = grammar.add_node(
                NodeKindV7.AND,
                "dynamic_motif",
                f"{query.text}:{motif.name}",
                attributes={
                    "motif_id": motif.motif_id,
                    "runtime_query_id": int(query.query_id),
                    "slot_uids": list(motif.slot_uids),
                    "source": motif.source,
                },
                complexity_cost=float(motif.complexity_cost),
            )
            grammar.add_rule(motif_node, children, kind=RuleKindV7.AND_COMPOSE, branch_prior=max(1e-4, float(motif.prior)))
            motif_nodes[motif.motif_id] = motif_node
            covered_slots.update(motif.slot_uids)

        pose_choice = grammar.add_node(NodeKindV7.OR, "dynamic_pose_choice", f"{query.text}:pose_choice", attributes={"runtime_query_id": int(query.query_id)})
        grammar.add_rule(object_node, [pose_choice], kind=RuleKindV7.AND_COMPOSE)
        poses = spec.poses or []
        if not poses:
            from .types import DynamicPoseSpecV7

            poses = [DynamicPoseSpecV7(f"q{query.query_id}:free", "free_pose", 1.0, list(slot_nodes), [], [], "fallback")]
        for pose in poses:
            pose_node = grammar.add_node(
                NodeKindV7.AND,
                "dynamic_pose_template",
                f"{query.text}:{pose.name}",
                attributes={
                    "runtime_query_id": int(query.query_id),
                    "pose_id": pose.pose_id,
                    "pose_template_id": pose.pose_id,
                    "geometry_mean": list(pose.geometry_mean),
                    "geometry_var": list(pose.geometry_var),
                    "source": pose.source,
                },
                complexity_cost=0.02,
            )
            grammar.add_rule(pose_choice, [pose_node], kind=RuleKindV7.OR_SELECT, branch_prior=max(1e-4, float(pose.prior)))
            children = list(motif_nodes.values()) + [node for uid, node in slot_nodes.items() if uid not in covered_slots]
            grammar.add_rule(pose_node, children, kind=RuleKindV7.AND_COMPOSE, relation_factors=relation_ids, complexity_cost=0.02)
    grammar.validate()
    return grammar, query_node_map
