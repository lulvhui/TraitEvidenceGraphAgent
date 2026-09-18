from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Tuple

from ..schemas import EvidenceEdge, EvidenceItem, TraitEvidenceGraph
from ..utils import clamp01


ContextPair = Tuple[str, str]


def context_pair(a: str, b: str) -> ContextPair:
    return tuple(sorted((str(a or "unknown"), str(b or "unknown"))))


class TraitEvidenceGraphBuilder:
    """Build a small evidence graph with only corroboration and conflict edges."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.cross_context_multiplier = float(cfg.get("cross_context_multiplier", 1.25))
        self.conflict_multiplier = float(cfg.get("conflict_multiplier", 1.10))
        if self.conflict_multiplier < 0:
            raise ValueError("conflict_multiplier must be non-negative")
        self.min_edge_weight = float(cfg.get("min_edge_weight", 0.08))
        self.min_context_distance = float(cfg.get("min_context_distance", 0.20))

    @staticmethod
    def _similarity(
        a: EvidenceItem,
        b: EvidenceItem,
        similarities: Mapping[ContextPair, float],
    ) -> float:
        ca = str(a.dialogue_context or "unknown")
        cb = str(b.dialogue_context or "unknown")
        if ca == cb:
            return 1.0
        pair = context_pair(ca, cb)
        if pair not in similarities:
            raise ValueError(
                f"Missing BGE-M3 semantic context similarity for {pair}"
            )
        return clamp01(similarities[pair])

    def _edge(
        self,
        a: EvidenceItem,
        b: EvidenceItem,
        similarities: Mapping[ContextPair, float],
    ) -> EvidenceEdge | None:
        opposite = {a.direction, b.direction} == {"support_high", "support_low"}
        same_direction = a.direction == b.direction and a.direction in {
            "support_high", "support_low"
        }
        if opposite:
            relation = "conflict"
            # Opposite poles form a maximal normalized edge. The configurable
            # multiplier is applied to aggregate conflict below, where values
            # above 1 can affect graph statistics without invalid edge weights.
            weight = 1.0
            cross_context = False
            rationale = "The turns support opposite trait poles."
        elif same_direction:
            similarity = self._similarity(a, b, similarities)
            context_distance = 1.0 - similarity
            if context_distance < self.min_context_distance:
                return None
            relation = "cross_context_support"
            weight = context_distance * self.cross_context_multiplier
            cross_context = True
            rationale = (
                f"Same-direction evidence recurs across semantically different contexts "
                f"(similarity={similarity:.3f})."
            )
        else:
            return None

        weight = clamp01(weight)
        if weight < self.min_edge_weight:
            return None
        return EvidenceEdge(
            source_turn=a.turn_id,
            target_turn=b.turn_id,
            relation=relation,
            weight=weight,
            same_direction=same_direction,
            cross_context=cross_context,
            rationale=rationale,
        )

    def build(
        self,
        person_id: str,
        trait: str,
        evidence: Iterable[EvidenceItem],
        context_similarities: Mapping[ContextPair, float] | None = None,
    ) -> TraitEvidenceGraph:
        nodes = [
            item for item in evidence
            if item.trait == trait
            and item.direction != "no_clear"
        ]
        similarities = context_similarities or {}

        edges: List[EvidenceEdge] = []
        for i, left in enumerate(nodes):
            for right in nodes[i + 1:]:
                edge = self._edge(left, right, similarities)
                if edge is not None:
                    edges.append(edge)

        direction_counts = Counter(node.direction for node in nodes)
        high_count = direction_counts.get("support_high", 0)
        low_count = direction_counts.get("support_low", 0)
        mixed_count = direction_counts.get("mixed", 0)
        if high_count == 0 and low_count == 0:
            dominant = "mixed" if mixed_count else "no_clear"
        elif high_count == low_count:
            dominant = "mixed"
        else:
            dominant = "support_high" if high_count > low_count else "support_low"

        pole_count = high_count + low_count
        base_conflict = (
            clamp01((2.0 * min(high_count, low_count) + mixed_count) / (pole_count + mixed_count))
            if pole_count + mixed_count > 0 else 0.0
        )
        conflict = clamp01(base_conflict * self.conflict_multiplier)
        dominant_nodes = (
            [node for node in nodes if node.direction == dominant]
            if dominant in {"support_high", "support_low"} else []
        )
        dominant_count = high_count if dominant == "support_high" else low_count if dominant == "support_low" else 0
        total_count = pole_count + mixed_count
        directional_consistency = dominant_count / total_count if total_count else 0.0

        dominant_ids = {node.turn_id for node in dominant_nodes}
        corroboration_edges = [
            edge for edge in edges
            if edge.relation == "cross_context_support"
            and edge.source_turn in dominant_ids
            and edge.target_turn in dominant_ids
        ]
        cross_context_support = clamp01(
            sum(edge.weight for edge in corroboration_edges)
            / max(1, len(dominant_nodes) - 1)
        )

        # Evidence mass grows smoothly with repeated nodes; semantic cross-context
        # recurrence adds corroboration, while opposing/mixed evidence reduces it.
        evidence_amount = 1.0 - math.exp(-dominant_count)
        stability = clamp01(
            directional_consistency
            * (0.60 * evidence_amount + 0.40 * cross_context_support)
            * (1.0 - conflict)
        )

        conflict_edges = [edge for edge in edges if edge.relation == "conflict"]
        summary = {
            "num_nodes": len(nodes),
            "direction_counts": dict(direction_counts),
            "high_turns": [n.turn_id for n in nodes if n.direction == "support_high"],
            "low_turns": [n.turn_id for n in nodes if n.direction == "support_low"],
            "mixed_turns": [n.turn_id for n in nodes if n.direction == "mixed"],
            "base_conflict": base_conflict,
            "conflict_multiplier": self.conflict_multiplier,
            "cross_context_support_edges": [
                [e.source_turn, e.target_turn, round(e.weight, 4)]
                for e in corroboration_edges
            ],
            "conflict_edges": [
                [e.source_turn, e.target_turn, round(e.weight, 4)]
                for e in conflict_edges
            ],
        }

        return TraitEvidenceGraph(
            person_id=person_id,
            trait=trait,
            nodes=nodes,
            edges=edges,
            dominant_direction=dominant,
            stability=stability,
            conflict=conflict,
            cross_context_support=cross_context_support,
            graph_summary=summary,
        )
