TRAITS = ["O", "C", "E", "A", "N"]
TRAIT_NAMES = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}
TRAIT_FOCI = {
    "O": "curiosity, creativity, novelty, and openness to experience",
    "C": "organization, planning, responsibility, and self-discipline",
    "E": "sociability, energy, assertiveness, and engagement",
    "A": "cooperation, empathy, consideration, and interpersonal warmth",
    "N": "anxiety, stress reactivity, emotional volatility, and emotional stability",
}
DIRECTIONS = ("support_high", "support_low", "mixed", "no_clear")
ROUTE_RELEVANCE = ("yes", "no_clear", "no")
ROUTE_CODE_TO_RELEVANCE = {"Y": "yes", "U": "no_clear", "N": "no"}
ROUTE_RELEVANCE_TO_CODE = {
    relevance: code for code, relevance in ROUTE_CODE_TO_RELEVANCE.items()
}
