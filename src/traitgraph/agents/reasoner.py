from __future__ import annotations

from typing import Dict, Sequence

from ..backbones.base import MultimodalBackbone
from ..schemas import EvidenceItem, TraitEvidenceGraph, TraitInference, TurnRecord
from ..utils import extract_json_object
from .prompts import reasoner_prompt, revision_prompt


class CrossTurnPersonalityReasoner:
    def __init__(self, backbone: MultimodalBackbone):
        self.backbone = backbone

    @staticmethod
    def _turn_sets(graph: TraitEvidenceGraph) -> tuple[list[str], list[str], list[str]]:
        high = [node.turn_id for node in graph.nodes if node.direction == "support_high"]
        low = [node.turn_id for node in graph.nodes if node.direction == "support_low"]
        mixed = [node.turn_id for node in graph.nodes if node.direction == "mixed"]
        low_is_support = graph.dominant_direction == "support_low"
        return (low, high, mixed) if low_is_support else (high, low, mixed)

    def infer(
        self,
        graph: TraitEvidenceGraph,
        turn_map: Dict[str, TurnRecord],
        dataset: str,
    ) -> TraitInference:
        evidence = list(graph.nodes)
        raw_turns = [turn_map[e.turn_id] for e in evidence if e.turn_id in turn_map]
        raw = self.backbone.generate(
            f"reasoner:{graph.trait}",
            reasoner_prompt(graph, evidence, dataset),
            raw_turns,
            {"max_new_tokens": 768},
        )
        data = extract_json_object(raw)

        support_turns, counter_turns, mixed_turns = self._turn_sets(graph)

        return TraitInference(
            trait=graph.trait,
            reasoning=str(data.get("reasoning", "")),
            support_turns=support_turns,
            counter_turns=counter_turns,
            mixed_turns=mixed_turns,
        )

    def revise(
        self,
        graph: TraitEvidenceGraph,
        inference: TraitInference,
        issues: list[dict],
        raw_turns: Sequence[TurnRecord],
        dataset: str,
    ) -> TraitInference:
        raw = self.backbone.generate(
            f"revision:{graph.trait}",
            revision_prompt(graph, inference.to_dict(), issues, dataset),
            raw_turns,
            {"max_new_tokens": 768},
        )
        data = extract_json_object(raw)

        if data.get("reasoning"):
            inference.reasoning = str(data["reasoning"])

        (
            inference.support_turns,
            inference.counter_turns,
            inference.mixed_turns,
        ) = self._turn_sets(graph)
        return inference
