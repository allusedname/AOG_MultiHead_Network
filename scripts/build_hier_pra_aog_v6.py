#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from partcat_hkg.data.schema import RoleSchema
from partcat_hkg.pra_aog import PRAAOGBuildConfig, build_pra_aog_from_records, save_pra_aog
from partcat_hkg.pra_aog_v6 import PartTemplateBank, PartTemplateDiscoveryConfig
from partcat_hkg.strict_aog.terminals import load_terminal_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a v6 PRA-AOG bundle and part-template bank.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--part-template-out", default="")
    parser.add_argument("--part-template-grid-size", type=int, default=3)
    parser.add_argument("--part-template-min-cell-coverage", type=float, default=0.06)
    parser.add_argument("--part-template-min-support", type=int, default=6)
    parser.add_argument("--part-template-min-cells", type=int, default=2)
    parser.add_argument("--part-template-max-per-part", type=int, default=6)
    parser.add_argument("--part-template-max-subparts", type=int, default=6)
    parser.add_argument("--part-template-mdl-cell-penalty", type=float, default=0.03)
    parser.add_argument("--part-template-score-boost", type=float, default=0.30)
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring strict/PRA build options handled by defaults:", " ".join(unknown))

    payload = load_terminal_cache(args.cache, map_location="cpu", materialize=True)
    records = payload.get("records", [])
    if not records:
        raise ValueError("Terminal cache contains no records")
    if payload.get("schema") is None:
        raise ValueError("Terminal cache has no RoleSchema payload")
    schema = RoleSchema.from_payload(payload["schema"])
    token_dim = int(records[0]["terminal_token"].shape[-1])

    bundle = build_pra_aog_from_records(
        records,
        schema=schema,
        token_dim=token_dim,
        num_parts=schema.num_parts,
        cfg=PRAAOGBuildConfig(),
    )
    bundle.metadata.update({"architecture": "pra-aog-v6", "part_templates_saved_separately": True})
    save_pra_aog(bundle, args.out)

    part_cfg = PartTemplateDiscoveryConfig(
        grid_size=int(args.part_template_grid_size),
        min_cell_coverage=float(args.part_template_min_cell_coverage),
        min_template_support=int(args.part_template_min_support),
        min_cells_per_template=int(args.part_template_min_cells),
        max_templates_per_part=int(args.part_template_max_per_part),
        max_subparts_per_template=int(args.part_template_max_subparts),
        mdl_cell_penalty=float(args.part_template_mdl_cell_penalty),
        terminal_score_boost=float(args.part_template_score_boost),
    )
    part_names = list(getattr(schema, "part_names", getattr(bundle.grammar, "part_names", [])))
    bank = PartTemplateBank.from_records(records, part_names=part_names, cfg=part_cfg)
    bank_path = Path(args.part_template_out) if args.part_template_out else Path(args.out).with_suffix(Path(args.out).suffix + ".part_templates.pt")
    bank.save(bank_path)
    print(f"saved bundle: {args.out}")
    print(f"saved part-template bank: {bank_path}")
    print(bank.summary(limit=20))


if __name__ == "__main__":
    main()
