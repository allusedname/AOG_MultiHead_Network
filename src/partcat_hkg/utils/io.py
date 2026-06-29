from pathlib import Path
from typing import Any
import json
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_checkpoint(path: str | Path, model, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_checkpoint(
    path: str | Path,
    model=None,
    strict: bool = True,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint, optionally into a model.

    Backward-compatible behavior: ``load_checkpoint(path, model, strict=True)``
    loads ``state_dict`` into ``model`` and returns the ``extra`` metadata.

    Utility behavior for newer scripts: ``load_checkpoint(path, map_location=...)``
    returns the raw payload when ``model`` is omitted.
    """

    payload = torch.load(path, map_location=map_location)
    if model is None:
        return payload
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=strict)
    return payload.get("extra", {}) if isinstance(payload, dict) else {}
