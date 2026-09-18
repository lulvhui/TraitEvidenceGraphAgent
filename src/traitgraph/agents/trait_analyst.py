from __future__ import annotations

from ..backbones.base import MultimodalBackbone
from ..constants import DIRECTIONS
from ..schemas import EvidenceItem, TurnRecord
from ..utils import extract_json_object
from .prompts import trait_prompt


class TraitEvidenceAnalyst:
    def __init__(self, backbone: MultimodalBackbone):
        self.backbone = backbone

    def analyze(self, turn: TurnRecord, trait: str) -> EvidenceItem:
        raw = self.backbone.generate(f"trait_analyst:{trait}", trait_prompt(trait), [turn])
        data = extract_json_object(raw)
        direction = str(data.get("direction", "no_clear")).strip()

        if direction not in DIRECTIONS:
            direction = "no_clear"

        semantic_context = " | ".join(
            part for part in (turn.question, turn.answer_text)
            if str(part).strip()
        )

        return EvidenceItem(
            person_id=turn.person_id,
            turn_id=turn.turn_id,
            trait=trait,
            direction=direction,
            dialogue_context=semantic_context or "unknown",
            raw_multimodal_reference=turn.turn_id,
        )
