#!/usr/bin/env python
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts" / "build_hier_pra_aog_v6.py"), run_name="__main__")
