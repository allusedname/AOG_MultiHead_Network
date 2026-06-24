#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.pra_aog import PRAAOGConfig, PRAAOGParser, load_pra_aog
from partcat_hkg.strict_aog.data import StrictAOGTerminalDataset, collate_strict_aog
from partcat_hkg.strict_aog.parser import ParserConfig


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return name


def _load_checkpoint(model: torch.nn.Module, path: str) -> None:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path} does not contain a model state dictionary")
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(
        f"loaded checkpoint={path} missing_keys={len(missing)} "
        f"unexpected_keys={len(unexpected)}"
    )


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode one cached validation example with PRA-AOG."
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="runs/pra_aog/inference")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--assignment",
        choices=["gpu_mf", "edge_greedy", "beam", "greedy", "sinkhorn"],
        default="gpu_mf",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--posterior-tau", type=float, default=0.75)
    parser.add_argument("--relation-weight", type=float, default=1.25)
    parser.add_argument("--count-weight", type=float, default=0.15)
    parser.add_argument("--missing-weight", type=float, default=0.35)
    parser.add_argument("--posterior-logits", action="store_true")
    parser.add_argument("--use-class-role-evidence", action="store_true")
    parser.add_argument("--disable-edges", action="store_true")
    args = parser.parse_args()

    device = torch.device(_device(args.device))
    dataset = StrictAOGTerminalDataset(
        args.cache,
        preload=True,
        include_visual=True,
    )
    index = int(args.sample_index)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"sample-index {index} is outside [0, {len(dataset) - 1}]")
    batch = collate_strict_aog([dataset[index]])

    bundle = load_pra_aog(args.bundle)
    parser_cfg = ParserConfig(
        assignment=str(args.assignment),
        relation_weight=float(args.relation_weight),
        count_weight=float(args.count_weight),
        missing_weight=float(args.missing_weight),
        role_overlap_weight=(0.40 if args.use_class_role_evidence else 0.0),
        template_tau=float(args.posterior_tau),
    )
    pra_cfg = PRAAOGConfig(
        top_k=int(args.top_k),
        posterior_tau=float(args.posterior_tau),
        use_class_role_evidence=bool(args.use_class_role_evidence),
        replace_logits_with_posterior=bool(args.posterior_logits),
    )
    model = PRAAOGParser(bundle, parser_cfg, pra_cfg).to(device)
    _load_checkpoint(model, args.checkpoint)
    model.eval()

    with torch.no_grad():
        output = model(
            batch,
            enable_edges=not args.disable_edges,
            return_forest=True,
            return_readouts=True,
        )

    class_probability = output["class_posterior"][0].detach().cpu()
    predicted_class = int(class_probability.argmax().item())
    class_names = list(bundle.grammar.class_names)
    forest = output["parse_forest"][0]
    map_parse = forest.map_parse
    readouts = dict(output.get("readouts", {}))
    semantic_masks = readouts.pop("semantic_mask_posterior", None)

    summary = {
        "sample_index": index,
        "ground_truth_class_id": (
            int(batch["obj_label"][0].item()) if "obj_label" in batch else None
        ),
        "predicted_class_id": predicted_class,
        "predicted_class_name": class_names[predicted_class],
        "class_names": class_names,
        "class_posterior": class_probability.tolist(),
        "parse_retained_mass": float(output["parse_retained_mass"][0].item()),
        "parse_entropy": float(output["parse_entropy"][0].item()),
        "map_parse": None if map_parse is None else map_parse.to_dict(),
        "parse_forest": forest.to_dict(),
        "topdown_queries": output.get("topdown_queries", [[]])[0],
        "readouts": _jsonable(readouts),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"sample_{index:05d}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if torch.is_tensor(semantic_masks):
        torch.save(
            semantic_masks.detach().cpu(),
            out_dir / f"sample_{index:05d}_semantic_masks.pt",
        )

    print(f"saved summary: {summary_path}")
    print(
        f"sample={index} predicted={class_names[predicted_class]} "
        f"probability={float(class_probability[predicted_class]):.4f} "
        f"retained_mass={summary['parse_retained_mass']:.4f} "
        f"entropy={summary['parse_entropy']:.4f}"
    )
    if map_parse is not None:
        print(
            f"map_template={map_parse.template_id} "
            f"integrality_gap={map_parse.integrality_gap:.4f}"
        )
        for slot in map_parse.slots:
            print(
                f"slot={slot.slot} part={slot.part} "
                f"visibility={slot.visibility.value} terminal={slot.terminal}"
            )


if __name__ == "__main__":
    main()
