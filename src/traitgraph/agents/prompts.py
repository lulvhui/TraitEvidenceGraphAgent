from __future__ import annotations

from ..constants import TRAIT_NAMES
from ..schemas import EvidenceItem, TraitEvidenceGraph
from ..utils import compact_json


def _without_confidence(value):
    """Remove confidence fields from person-level LLM inputs."""
    if isinstance(value, dict):
        return {
            key: _without_confidence(item)
            for key, item in value.items()
            if key != "confidence"
        }
    if isinstance(value, list):
        return [_without_confidence(item) for item in value]
    return value


ROUTER_GUIDANCE = {
    "O": "Route curiosity, novelty/variety, imagination, aesthetics, intellectual/cultural interests, new activities, or notably concrete, routine, conventional, pragmatic, or non-exploratory framing that may contextually support the low pole.",
    "C": "Route schedules, routines, specific plans or goals, responsibility, persistence, preparation, rule adherence, concentration, precision/detail, structured reasoning, self-regulation, inexperience, poor planning, inconsistency, or weak follow-through.",
    "E": "Route sociability, social participation, assertiveness, talkativeness, positive expressive energy, enthusiastic elaboration, comfortable self-disclosure, solitude preference, reserve, low animation, or social withdrawal.",
    "A": "Route empathy, helping, warmth, accommodation, respect, relational commitment, cooperation/social harmony, concern for others, antagonism, distrust, dismissiveness, intolerance, or low interpersonal priority.",
    "N": "Route worry, self-doubt, uncertainty, stress sensitivity, discomfort, irritation, pessimism, emotional vulnerability, or calm/confident/positive composure that may contextually counter situational anxiety.",
}


def trait_router_prompt() -> str:
    guidance = "\n".join(f"- {k} ({TRAIT_NAMES[k]}): {v}" for k, v in ROUTER_GUIDANCE.items())
    return f"""You are Stage-2A: a high-recall Big Five TRAIT EVIDENCE ROUTER.
The ORIGINAL modalities supported by the current backbone are provided directly to you.

Your job is ROUTING ONLY. For each Big Five dimension, decide whether this turn deserves inspection by that trait's specialist analyst.
You are NOT deciding the trait direction or final personality.

Relevance labels:
- yes: there is a plausible trait-specific cue. The specialist analyst will inspect this trait.
- no_clear: relevance cannot be reliably confirmed or ruled out. Use this as a safe fallback; the specialist analyst will inspect this trait.
- no: there is no plausible cue for this trait. The specialist analyst will not inspect this trait.

Important calibration:
- Judge each trait independently for every turn.
- Use no_clear when the available signal is ambiguous or too uncertain for the router to make a reliable yes/no decision.
- Use no only when this trait can be confidently excluded. Pure identity, procedural ID handover,
  one-word readiness, or generic politeness will normally be no for all five traits.
- Use yes whenever a plausible trait cue might exist, even if it is weak, topic-prompted, contextual, or open to interpretation.
- When uncertain between no_clear and no, choose no_clear so that the analyst can safely resolve the ambiguity.
- A specific future plan can be weak/contextual C or O evidence and should be routed when a defensible mapping is plausible.
- A negative opinion by itself is not automatically relevant to N, E, or A.
- Social activity is relevant to E, but does not automatically make A relevant.
- Concrete factual or routine framing may be contextual low-O evidence under the reviewed GT policy; do not require explicit resistance to novelty before routing it.
- Calm, confident, or upbeat behavior may be contextual low-N evidence under the reviewed GT policy; do not require an explicit stressor before routing it.
- Technical interests/job choice are not automatically O or C, but specific curiosity, planning, commitment, or structured explanation may be relevant.
- Do NOT predict direction, confidence, or explanations.

Trait routing criteria:
{guidance}

Use the compact single-token routing codes Y=yes, U=no_clear, and N=no.
Return exactly one JSON object with all five keys:
{{
  "O": {{"relevance": "Y|U|N"}},
  "C": {{"relevance": "Y|U|N"}},
  "E": {{"relevance": "Y|U|N"}},
  "A": {{"relevance": "Y|U|N"}},
  "N": {{"relevance": "Y|U|N"}}
}}"""


TRAIT_GUIDANCE = {
    "O": """Openness concerns curiosity, preference for novelty/variety, imagination, intellectual exploration,
creativity, aesthetics, nature, languages/culture, and interest in varied or new experiences.
High O includes novelty-seeking, aesthetic description, varied interests, reflective meaning-making,
or willingness to try an activity. Low O may be indicated when the answer is notably literal, concrete, functional, routine-bound,
conventional, or pragmatically non-exploratory; explicit resistance to novelty is not required by the reviewed GT policy.""",
    "C": """Conscientiousness concerns organization, persistence, self-discipline, reliability, planning process,
detail orientation, sustained goal-directed effort, careful execution, and follow-through.
High C includes behavioral process evidence, schedules, routines, specific future plans, goal commitment, rule adherence,
precise details, and clearly structured reasoning even without sustained follow-through.
Low C includes disorganization, unreliability, procrastination, impulsive neglect, inconsistency, weak follow-through, limited preparation,
or admitted inexperience/lack of initiative when the mapping is appropriately bounded.""",
    "E": """Extraversion concerns sociability, assertiveness, interpersonal engagement, talkativeness, activity/energy,
positive affect, expressive enthusiasm, comfortable self-disclosure, and preference for social stimulation.
High E may be supported by enthusiastic wording, elaboration, upbeat visible affect, personal anecdotes, or conversational comfort,
even when the content is not explicitly social. Low E may be supported by explicit solitude/avoidance and patterns of reserve,
minimal animation, background-role preference, or predominantly solitary routines.""",
    "A": """Agreeableness concerns empathy, compassion, cooperation, trust, consideration, interpersonal warmth,
and willingness to accommodate others. Low A concerns antagonism, hostility, distrust, callousness, selfish disregard,
low interpersonal priority, dismissiveness, intolerance, or uncooperative behavior.
High A includes concrete helping/empathy, accommodation, respect, relational warmth or loyalty,
social harmony, cooperative norms, and concern for collective or interpersonal welfare. Merely naming family/friends or smiling remains no_clear.""",
    "N": """Neuroticism concerns recurring/disproportionate anxiety, worry, emotional volatility, stress sensitivity,
insecurity, irritability, or difficulty regulating negative affect.
High N may be indicated when the turn shows self-doubt, hesitation, uncertainty, perceived inadequacy/difficulty,
threat focus, discomfort, irritation, or situational stress sensitivity, even without severe emotional dysregulation.
Low N may be contextual when the speaker is visibly calm, confident, fluent, positive, or composed without worry/self-doubt in an ordinary
interaction. Such evidence describes low situational anxiety only and must not be upgraded into a stable low-trait conclusion.""",
}


def direct_personality_prompt(trait: str, dataset: str) -> str:
    """Build a graph-free person-level prompt for a direct baseline."""
    name = TRAIT_NAMES[trait]
    guidance = TRAIT_GUIDANCE[trait]
    return f"""You are a person-level multimodal personality reasoner.
Judge ONLY {name}; do not judge the other Big Five traits.
All available turns for one person are supplied directly with transcript, video,
and any audio supported by the current backbone.

Trait-specific guidance:
{guidance}

Requirements:
1. Inspect all supplied turns and modalities directly. Do not assume that every
   turn contains useful evidence.
2. Ground each material claim with its exact form \"Turn <id>\" and distinguish
   high-pole, low-pole, mixed, and uninformative observations.
3. Weigh corroboration, contradictions, context diversity, and limitations
   without inventing evidence or inferring an opposite pole from silence.
4. End with an integrated, relative conclusion bounded to dataset {dataset} and
   this observed interaction. Do not assign a numeric score or claim stable,
   cross-situational personality.

Return only JSON:
{{"reasoning": "complete evidence-grounded reasoning and final synthesis"}}"""


def trait_prompt(trait: str) -> str:
    name = TRAIT_NAMES[trait]
    guidance = TRAIT_GUIDANCE[trait]
    return f"""You are Stage-2B: a trait-specific multimodal evidence analyst for {name}.
This turn has been sent to you for trait analysis through normal routing or a recall fallback.
Do not validate, repeat, or challenge the router's relevance judgment.
You receive the ORIGINAL modalities supported by the current backbone for one turn.

Judge ONLY {name}. Do not judge the other Big Five traits.
Trait-specific guidance:
{guidance}

EVIDENCE RULES:
- First decide whether the turn contains a reliable, trait-specific directional cue. Router selection only means the turn was worth inspecting.
- Preserve defensible cues even when they are prompted, ordinary, indirect, or limited to the current interaction.
- Return no_clear only when no defensible mapping to this trait remains after applying the GT-aligned guidance.
- Otherwise use support_high, support_low, or mixed. mixed requires genuinely opposing cues within this same turn.
- Do not convert evidence for one Big Five trait into another trait.
- Procedural compliance, ID handover, generic politeness, and a one-word factual answer are no_clear unless another specific cue is present.
- For Openness and Neuroticism, follow the explicit contextual low-pole rules above rather than the general prohibition on absence-based inference.

Return only JSON:
{{
  "direction": "support_high|support_low|mixed|no_clear"
}}"""


def reasoner_prompt(
    graph: TraitEvidenceGraph,
    evidence: list[EvidenceItem],
    dataset: str,
) -> str:
    graph_data = {
        "person_id": graph.person_id,
        "trait": graph.trait,
        "nodes": [
            {
                "turn_id": node.turn_id,
                "direction": node.direction,
                "dialogue_context": node.dialogue_context,
            }
            for node in graph.nodes
        ],
        "edges": [
            [edge.source_turn, edge.target_turn, edge.relation, round(edge.weight, 4)]
            for edge in graph.edges
        ],
        "metrics": {
            "dominant_direction": graph.dominant_direction,
            "stability": round(graph.stability, 4),
            "conflict": round(graph.conflict, 4),
            "cross_context_support": round(graph.cross_context_support, 4),
        },
    }
    raw_turn_ids = [item.turn_id for item in evidence]
    return f"""You are a person-level personality reasoner.
Infer {TRAIT_NAMES[graph.trait]} from a CROSS-TURN TRAIT EVIDENCE GRAPH.
Every graph node's raw multimodal Turn is supplied directly; textual evidence summaries are indexes,
not replacements for the original modalities.

Graph:
{compact_json(graph_data)}

Raw Turn IDs supplied for all graph nodes:
{compact_json(raw_turn_ids)}

Requirements:
1. Treat graph diagnostics as authoritative.
2. Use the whole graph. Judge direction counts, cross-context corroboration, and conflict together.
3. Do not infer the opposite pole from absence of evidence or invent turns and observations.
4. Explain every material node separately as "Turn <id>". A mixed Turn must explain both poles once,
   without treating it as two independent observations.
5. End with a relative, integrated conclusion about the observed pattern for {TRAIT_NAMES[graph.trait]}.
   Combine direction, corroboration, conflict, context diversity, and limitations. Do not convert
   direction counts mechanically into a categorical score or invent a numeric/ordinal label.
6. Keep the conclusion bounded to dataset {dataset} and the observed interaction. Do not claim stable,
   cross-situational personality without corresponding evidence, and do not assign final confidence.

Return only JSON:
{{
  "reasoning": "complete evidence-grounded reasoning covering all material evidence and the final synthesis"
}}"""


def verifier_prompt(graph: TraitEvidenceGraph, inference: dict) -> str:
    graph_data = _without_confidence(graph.to_dict())
    inference_data = _without_confidence(inference)
    return f"""You are a consistency verifier, not a new Stage-2 trait analyst.
Check Evidence Graph -> Reasoning consistency for {TRAIT_NAMES[graph.trait]}.
Treat graph nodes and directions as given. Do not add, remove, relabel, or reassess Stage-2 evidence.
Dialogue context is an index to the supplied Turn, not an additional trait label.
Use the whole graph together with every supplied raw Turn.
Check only whether cited turns exist, material evidence is explained faithfully, graph corroboration/conflict is represented consistently,
and the integrated synthesis follows from the graph without overstating its scope.
Check each reasoning claim about a node or graph relation against the supplied graph.
Do not introduce a numeric or ordinal personality label. Do not report an issue merely because you
would phrase the relative synthesis differently from the reasoner.
Treat negated claims literally: saying corroboration is absent does not claim that it exists.
Do not use keyword matching for this rule. Only unsupported discussion of final inference confidence is out of scope.
Graph: {compact_json(graph_data)}
Inference: {compact_json(inference_data)}
Return only JSON: {{
  "issues": [{{"type": str, "turn_ids": [str], "message": str}}]
}}.
Return an empty issues list when no concrete correction is needed."""


def revision_prompt(
    graph: TraitEvidenceGraph,
    inference: dict,
    issues: list[dict],
    dataset: str,
) -> str:
    graph_data = _without_confidence(graph.to_dict())
    inference_data = _without_confidence(inference)
    return f"""Revise ONLY the faulty parts of the previous person-level reasoning for {TRAIT_NAMES[graph.trait]}.
Keep correct claims unchanged. Do not change Stage-2 evidence or the graph.
Graph: {compact_json(graph_data)}
Previous inference: {compact_json(inference_data)}
Verifier issues: {compact_json(issues)}

Rules:
- Use the whole graph and acknowledge material counter-evidence.
- Weigh direction counts, cross-context corroboration, and conflict together.
- Do not turn absence of evidence into opposite-pole evidence.
- End with a relative, integrated conclusion bounded to dataset {dataset} and the observed interaction.
- Do not add a numeric, categorical, or ordinal personality label.
- Do not describe or assign confidence for the final personality inference.
- Explain the evidence and final synthesis in plain language instead of copying raw graph metrics.
Return only JSON: {{"reasoning": str}}."""
