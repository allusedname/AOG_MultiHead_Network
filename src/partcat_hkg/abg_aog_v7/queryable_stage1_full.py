from __future__ import annotations

from .queryable_stage1 import NeuralQueryableStage1V7


class FullNeuralQueryableStage1V7(NeuralQueryableStage1V7):
    """Compatibility alias for the contract-gated neural re-query wrapper.

    The old override consumed untrained amodal/port heads and returned ROI-local
    masks as if they were image-global evidence.  The parent implementation now
    handles projection and enables amodal/port outputs only when the checkpoint
    contract explicitly marks those heads as supervised.
    """
