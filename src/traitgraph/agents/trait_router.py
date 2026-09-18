from __future__ import annotations

from ..backbones.base import MultimodalBackbone
from ..constants import ROUTE_CODE_TO_RELEVANCE, ROUTE_RELEVANCE, TRAITS
from ..schemas import TraitRoute, TraitRoutingResult, TurnRecord
from ..utils import extract_json_object
from .prompts import trait_router_prompt


class TraitEvidenceRouter:
    """Stage-2A high-recall router.

    One model call evaluates all five Big Five dimensions.  The router decides
    only whether a specialist should inspect the turn.  It never decides
    support_high/support_low or person-level personality.
    """

    def __init__(self, backbone: MultimodalBackbone):
        self.backbone = backbone

    def route(self, turn: TurnRecord) -> TraitRoutingResult:
        raw = self.backbone.generate("trait_router", trait_router_prompt(), [turn])
        data = extract_json_object(raw)

        routes: dict[str, TraitRoute] = {}
        for trait in TRAITS:
            item = data.get(trait)
            if isinstance(item, dict):
                raw_relevance = str(item.get("relevance", "")).strip()
            elif isinstance(item, str):
                # Some base-model generations omit the redundant relevance
                # wrapper while preserving the requested Y/U/N code.
                raw_relevance = item.strip()
            else:
                raise ValueError(
                    f"Router output is missing field {trait}: {raw!r}"
                )

            relevance = ROUTE_CODE_TO_RELEVANCE.get(
                raw_relevance.upper(), raw_relevance.lower()
            )
            if relevance not in ROUTE_RELEVANCE:
                raise ValueError(
                    f"Router output has invalid {trait}.relevance={relevance!r}: {raw!r}"
                )

            routes[trait] = TraitRoute(
                trait=trait,
                relevance=relevance,
            )

        return TraitRoutingResult(
            person_id=turn.person_id,
            turn_id=turn.turn_id,
            routes=routes,
        )
