from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class TurnRecord:
    """One unified IELTS/RecruitView turn."""

    person_id: str
    turn_id: str
    question: str = ""
    answer_text: str = ""
    time_q: List[float] = field(default_factory=list)
    time_a: List[float] = field(default_factory=list)
    video_path: str = ""
    source_dataset: str = ""

    def __post_init__(self) -> None:
        self.person_id = str(self.person_id)
        self.turn_id = str(self.turn_id)
        self.question = str(self.question or "")
        self.answer_text = str(self.answer_text or "")
        self.video_path = str(self.video_path or "")
        self.source_dataset = str(self.source_dataset or "")
        self.time_q = list(self.time_q or [])
        self.time_a = list(self.time_a or [])

    @property
    def text(self) -> str:
        parts: List[str] = []
        if self.question.strip():
            parts.append(f"Question: {self.question.strip()}")
        if self.answer_text.strip():
            parts.append(f"Answer: {self.answer_text.strip()}")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraitRoute:
    trait: str
    relevance: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraitRoutingResult:
    person_id: str
    turn_id: str
    routes: Dict[str, TraitRoute] = field(default_factory=dict)

    @property
    def selected_traits(self) -> List[str]:
        return [
            trait
            for trait, route in self.routes.items()
            if route.relevance in {"yes", "no_clear"}
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "turn_id": self.turn_id,
            "routes": {k: v.to_dict() for k, v in self.routes.items()},
            "selected_traits": self.selected_traits,
        }


@dataclass
class EvidenceItem:
    person_id: str
    turn_id: str
    trait: str
    direction: str
    dialogue_context: str = "unknown"
    raw_multimodal_reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceEdge:
    source_turn: str
    target_turn: str
    relation: str
    weight: float
    same_direction: bool
    cross_context: bool
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraitEvidenceGraph:
    person_id: str
    trait: str
    nodes: List[EvidenceItem] = field(default_factory=list)
    edges: List[EvidenceEdge] = field(default_factory=list)
    dominant_direction: str = "no_clear"
    stability: float = 0.0
    conflict: float = 0.0
    cross_context_support: float = 0.0
    graph_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "trait": self.trait,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "dominant_direction": self.dominant_direction,
            "stability": self.stability,
            "conflict": self.conflict,
            "cross_context_support": self.cross_context_support,
            "graph_summary": self.graph_summary,
        }


@dataclass
class TraitInference:
    trait: str
    reasoning: str
    support_turns: List[str] = field(default_factory=list)
    counter_turns: List[str] = field(default_factory=list)
    mixed_turns: List[str] = field(default_factory=list)
    verification_status: str = "not_run"
    verification_issues: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"reasoning": self.reasoning}

    def to_debug_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PersonInference:
    person_id: str
    source_dataset: str
    traits: Dict[str, TraitInference]
    trait_routes: Dict[str, TraitRoutingResult] = field(default_factory=dict)
    trait_analyses: Dict[str, Dict[str, EvidenceItem]] = field(default_factory=dict)
    graphs: Dict[str, TraitEvidenceGraph] = field(default_factory=dict)
    backbone_calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "source_dataset": self.source_dataset,
            "traits": {k: v.to_dict() for k, v in self.traits.items()},
        }

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id,
            "source_dataset": self.source_dataset,
            "traits": {k: v.to_debug_dict() for k, v in self.traits.items()},
            "trait_routes": {k: v.to_dict() for k, v in self.trait_routes.items()},
            "trait_analyses": {
                turn_id: {trait: item.to_dict() for trait, item in analyses.items()}
                for turn_id, analyses in self.trait_analyses.items()
            },
            "graphs": {k: v.to_dict() for k, v in self.graphs.items()},
            "backbone_calls": list(self.backbone_calls),
        }
