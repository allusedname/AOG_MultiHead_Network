from __future__ import annotations

from typing import Any

import torch.nn as nn


class ABGHKGAOGParser(nn.Module):
    def __init__(self, base_parser: nn.Module) -> None:
        super().__init__()
        self.base_parser = base_parser

    @property
    def grammar(self):
        return self.base_parser.grammar

    @property
    def cfg(self):
        return self.base_parser.cfg

    @property
    def num_classes(self) -> int:
        return int(self.base_parser.num_classes)

    def forward(self, batch: dict[str, Any], *, enable_edges: bool = True, return_forest: bool = False, return_readouts: bool = False) -> dict[str, Any]:
        return self.base_parser(batch, enable_edges=enable_edges, return_forest=return_forest, return_readouts=return_readouts)
