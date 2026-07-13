from __future__ import annotations

from typing import Any

import torch

from partcat_hkg.abg_aog_v7.terminal_adapter import terminal_packets_from_record

from .types import OpenVocabObjectQueryV7, OpenVocabTerminalV7
from .universal_bank import UniversalStructuralBankV7


def open_vocab_terminals_from_record_v7(
    record: dict[str, Any],
    bank: UniversalStructuralBankV7,
    *,
    sample_id: int | None = None,
    score_tau: float = 0.05,
    include_masks: bool = True,
    include_tokens: bool = True,
) -> list[OpenVocabTerminalV7]:
    packets = terminal_packets_from_record(
        record,
        sample_id=sample_id,
        score_tau=score_tau,
        include_masks=include_masks,
        include_tokens=include_tokens,
    )
    part_by_id = {int(part.part_id): part for part in bank.universal_parts}
    out: list[OpenVocabTerminalV7] = []
    for packet in packets:
        part = part_by_id.get(int(packet.functional_part_id))
        if part is None:
            name = f"part_{int(packet.functional_part_id)}"
            embedding = None
        else:
            name = part.name
            embedding = torch.tensor(part.text_embedding, dtype=torch.float32)
        out.append(OpenVocabTerminalV7(
            packet=packet,
            part_query_id=int(packet.functional_part_id),
            part_text=name,
            part_embedding=embedding,
            generic_score=float(packet.visible_score),
            contextual_score=0.0,
            provenance={"source": "legacy_terminal_cache_adapter"},
        ))
    return out


def known_object_queries_v7(bank: UniversalStructuralBankV7) -> list[OpenVocabObjectQueryV7]:
    return [
        OpenVocabObjectQueryV7(
            query_id=int(grammar.class_id),
            text=grammar.class_name,
            embedding=torch.tensor(grammar.object_embedding, dtype=torch.float32),
            provenance="known_grammar_teacher",
        )
        for grammar in bank.known_grammars
    ]
