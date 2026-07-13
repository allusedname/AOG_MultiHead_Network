#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.data.partimagenet import RoleAwarePartImageNetDataset, resolve_partimagenet_annotation, resolve_partimagenet_image_root
from partcat_hkg.open_vocab_abg.data import OpenVocabPartImageNetDatasetV7, OpenVocabPseudoLabelDatasetV7, serialize_open_vocab_records_v7


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dynamic-query Stage-1 training records")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--partimagenet-root", default="")
    source.add_argument("--pseudo-manifest", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--negative-objects", type=int, default=2)
    parser.add_argument("--negative-parts", type=int, default=4)
    parser.add_argument("--negative-roles", type=int, default=2)
    parser.add_argument("--disable-roles", action="store_true")
    parser.add_argument("--disable-port-pseudo-labels", action="store_true")
    parser.add_argument("--pseudo-min-confidence", type=float, default=0.70)
    parser.add_argument("--pseudo-part-vocabulary", default="", help="Comma-separated vocabulary used to sample negative pseudo queries")
    parser.add_argument("--eval-transform", action="store_true")
    args = parser.parse_args()

    if args.partimagenet_root:
        root = Path(args.partimagenet_root)
        annotation = resolve_partimagenet_annotation(root, args.split)
        image_root = resolve_partimagenet_image_root(root)
        # Build the complete schema before applying max-sample limits.
        schema = RoleAwarePartImageNetDataset(annotation, image_root, img_size=args.img_size, train=False).schema
        dataset = OpenVocabPartImageNetDatasetV7(
            annotation,
            image_root,
            img_size=int(args.img_size),
            train=not bool(args.eval_transform),
            schema=schema,
            max_samples=int(args.max_samples) if args.max_samples > 0 else None,
            negative_objects=int(args.negative_objects),
            negative_parts=int(args.negative_parts),
            negative_roles=int(args.negative_roles),
            include_roles=not bool(args.disable_roles),
            include_port_pseudo_labels=not bool(args.disable_port_pseudo_labels),
        )
        source_summary = {"source": "PartImageNet", "annotation": str(annotation), "image_root": str(image_root), "split": args.split}
    else:
        vocabulary = [item.strip() for item in args.pseudo_part_vocabulary.split(",") if item.strip()]
        dataset = OpenVocabPseudoLabelDatasetV7(
            args.pseudo_manifest,
            img_size=int(args.img_size),
            train=not bool(args.eval_transform),
            min_confidence=float(args.pseudo_min_confidence),
            negative_part_vocabulary=vocabulary,
            negative_parts=int(args.negative_parts),
        )
        source_summary = {"source": "pseudo_manifest", "manifest": args.pseudo_manifest}

    summary = serialize_open_vocab_records_v7(dataset, args.out, max_samples=int(args.max_samples))
    summary.update(source_summary)
    Path(str(args.out) + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
