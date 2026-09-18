from __future__ import annotations

import re
from typing import List

from ..backbones.base import MultimodalBackbone
from ..schemas import TraitEvidenceGraph, TraitInference
from ..utils import extract_json_object
from .prompts import verifier_prompt


class ConsistencyVerifier:
    def __init__(self, backbone: MultimodalBackbone, config: dict | None = None):
        self.backbone = backbone
        self.config = config or {}

    def deterministic_check(self, graph: TraitEvidenceGraph, inference: TraitInference) -> List[dict]:
        if not self.config.get("enabled", True):
            return []

        issues: List[dict] = []
        valid_turns = {n.turn_id for n in graph.nodes}
        referenced = set(
            inference.support_turns
            + inference.counter_turns
            + inference.mixed_turns
        )
        invalid = sorted(t for t in referenced if t not in valid_turns)
        if invalid:
            issues.append({
                "type": "invalid_turn_reference",
                "turn_ids": invalid,
                "message": "Inference references turns that are not nodes in the trait evidence graph.",
            })

        missing_discussion = [
            node.turn_id
            for node in graph.nodes
            if not re.search(
                rf"\bTurn\s+{re.escape(str(node.turn_id))}\b",
                inference.reasoning,
                re.IGNORECASE,
            )
        ]
        if missing_discussion:
            issues.append({
                "type": "missing_node_explanation",
                "turn_ids": missing_discussion,
                "message": "Reasoning must explain every material graph node using its Turn ID.",
            })

        return issues


    def verify(
        self,
        graph: TraitEvidenceGraph,
        inference: TraitInference,
        deterministic_issues: List[dict] | None = None,
    ) -> dict:
        deterministic_issues = list(deterministic_issues or [])

        raw = self.backbone.generate(
            f"verifier:{graph.trait}",
            verifier_prompt(graph, inference.to_dict()),
            [],
            {"max_new_tokens": 512},
        )
        data = extract_json_object(raw)
        raw_issues = data.get("issues", [])
        llm_issues = (
            [issue for issue in raw_issues if isinstance(issue, dict)]
            if isinstance(raw_issues, list)
            else []
        )

        # A revision without a concrete issue gives the reasoner nothing to fix.
        issues = deterministic_issues + llm_issues
        decision = "revise" if issues else "accept"
        return {"decision": decision, "issues": issues}
