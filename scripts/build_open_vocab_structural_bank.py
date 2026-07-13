#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.abg_aog_v7.multislot_native import MultiSlotBankV7
from partcat_hkg.open_vocab_abg.text_encoder import DynamicTextQueryEncoderV7
from partcat_hkg.open_vocab_abg.universal_bank import build_universal_structural_bank_v7


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the universal HKG/AOG structural bank from a known-class MultiSlotBankV7")
    parser.add_argument("--multislot-bank", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pose-bank", default="")
    parser.add_argument("--text-model", default="ViT-B-16")
    parser.add_argument("--text-pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--allow-fallback-text", action="store_true")
    parser.add_argument("--max-multiplicity", type=int, default=6)
    args = parser.parse_args()

    text_encoder = DynamicTextQueryEncoderV7(
        model_name=args.text_model,
        pretrained=args.text_pretrained,
        require_semantic=not bool(args.allow_fallback_text),
    )
    multislot = MultiSlotBankV7.load(args.multislot_bank, map_location="cpu")
    pose_bank = None
    if args.pose_bank:
        try:
            from partcat_hkg.abg_aog_v7.complete_extensions import PoseBankV7

            pose_bank = PoseBankV7.load(args.pose_bank, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(f"Failed to load pose bank: {exc}") from exc
    bank = build_universal_structural_bank_v7(multislot, text_encoder, pose_bank=pose_bank, max_multiplicity=int(args.max_multiplicity))
    bank.save(args.out)
    print(f"saved: {args.out}")
    print(f"parts={len(bank.universal_parts)} grammars={len(bank.known_grammars)} motifs={len(bank.universal_motifs)} relations={len(bank.relation_primitives)}")
    print(f"text_backend={text_encoder.status.backend} semantic={text_encoder.status.semantic}")


if __name__ == "__main__":
    main()
