from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .agents import (
    ConsistencyVerifier,
    CrossTurnPersonalityReasoner,
    TraitEvidenceAnalyst,
    TraitEvidenceRouter,
)
from .backbones.base import MultimodalBackbone
from .constants import TRAITS, TRAIT_NAMES
from .context_similarity import BGEContextSimilarity, required_context_pairs
from .pool import TraitEvidenceGraphBuilder
from .pool.graph_builder import ContextPair
from .schemas import EvidenceItem, PersonInference, TraitInference, TurnRecord


class TraitEvidenceGraphPipeline:
    def __init__(self, backbone: MultimodalBackbone, config: dict | None = None):
        self.backbone = backbone
        self.config = config or {}
        self.router = TraitEvidenceRouter(backbone)
        self.analyst = TraitEvidenceAnalyst(backbone)
        self.context_similarity = BGEContextSimilarity(
            self.config.get("context_similarity", {})
        )
        self.graph_builder = TraitEvidenceGraphBuilder(self.config.get("graph", {}))
        self.reasoner = CrossTurnPersonalityReasoner(backbone)
        self.verifier = ConsistencyVerifier(backbone, self.config.get("verification", {}))

    def _semantic_context_similarities(
        self,
        evidence_by_trait: Dict[str, List[EvidenceItem]],
    ) -> Dict[ContextPair, float]:
        requested_pairs = required_context_pairs(
            item for items in evidence_by_trait.values() for item in items
        )
        if not requested_pairs:
            return {}

        return self.context_similarity.similarities(requested_pairs)

    def infer_person(self, turns: List[TurnRecord]) -> PersonInference:
        if not turns:
            raise ValueError("turns must not be empty")

        person_ids = {t.person_id for t in turns}
        if len(person_ids) != 1:
            raise ValueError("All turns must belong to one person")
        datasets = {t.source_dataset for t in turns}
        if len(datasets) != 1:
            raise ValueError("All turns must belong to one source_dataset")

        diagnostic_cursor = (
            self.backbone.diagnostic_cursor()
            if hasattr(self.backbone, "diagnostic_cursor")
            else 0
        )
        person_id = turns[0].person_id
        source_dataset = turns[0].source_dataset
        turn_map = {t.turn_id: t for t in turns}

        trait_routes = {}
        trait_analyses: Dict[str, Dict[str, EvidenceItem]] = defaultdict(dict)
        evidence_by_trait: Dict[str, List[EvidenceItem]] = defaultdict(list)

        # Every raw turn receives one router call. A specialist analyst inspects
        # both positive and uncertain routes; only an explicit "no" is skipped.
        for turn in turns:
            routing = self.router.route(turn)
            trait_routes[turn.turn_id] = routing
            trait_analyses.setdefault(turn.turn_id, {})

            for trait in routing.selected_traits:
                item = self.analyst.analyze(turn, trait)
                trait_analyses[turn.turn_id][trait] = item
                if item.direction != "no_clear":
                    evidence_by_trait[trait].append(item)

        context_similarities = self._semantic_context_similarities(evidence_by_trait)
        graphs = {
            trait: self.graph_builder.build(
                person_id,
                trait,
                evidence_by_trait.get(trait, []),
                context_similarities,
            )
            for trait in TRAITS
        }

        trait_outputs: Dict[str, TraitInference] = {}
        verification_enabled = bool(
            self.config.get("verification", {}).get("enabled", True)
        )
        max_revision_rounds = max(
            0,
            int(self.config.get("verification", {}).get("max_revision_rounds", 2)),
        )

        for trait, graph in graphs.items():
            inference = self.reasoner.infer(graph, turn_map, source_dataset)

            if not verification_enabled:
                inference.verification_status = "disabled"
                inference.verification_issues = []
            else:
                revision_round = 0
                while True:
                    deterministic_issues = self.verifier.deterministic_check(graph, inference)
                    result = self.verifier.verify(
                        graph,
                        inference,
                        deterministic_issues=deterministic_issues,
                    )
                    if result["decision"] == "accept":
                        inference.verification_status = (
                            "verified_accept"
                            if revision_round == 0
                            else "revised_verified_accept"
                        )
                        inference.verification_issues = []
                        break

                    if revision_round >= max_revision_rounds:
                        inference.verification_status = "revision_exhausted"
                        inference.verification_issues = result["issues"]
                        break

                    inference = self.reasoner.revise(
                        graph,
                        inference,
                        result["issues"],
                        [turn_map[node.turn_id] for node in graph.nodes if node.turn_id in turn_map],
                        source_dataset,
                    )
                    revision_round += 1

            trait_outputs[trait] = inference

        backbone_calls = (
            self.backbone.diagnostics_since(diagnostic_cursor)
            if hasattr(self.backbone, "diagnostics_since")
            else []
        )
        return PersonInference(
            person_id=person_id,
            source_dataset=source_dataset,
            traits=trait_outputs,
            trait_routes=trait_routes,
            trait_analyses=dict(trait_analyses),
            graphs=graphs,
            backbone_calls=backbone_calls,
        )
