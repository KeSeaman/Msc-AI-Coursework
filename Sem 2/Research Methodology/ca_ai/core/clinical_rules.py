from .protocols import SafetyAction, SafetyReport

def evaluate_distress_level(toxicity_score: float, self_harm_score: float) -> SafetyReport:
    """
    Pure function to evaluate safety scores.
    Returns ESCALATE if any score exceeds 0.8.
    """
    if toxicity_score > 0.8 or self_harm_score > 0.8:
        return SafetyReport(
            action=SafetyAction.ESCALATE,
            score=max(toxicity_score, self_harm_score),
            details="Critical distress or toxicity detected. Escalate immediately."
        )
    return SafetyReport(
        action=SafetyAction.PROCEED,
        score=max(toxicity_score, self_harm_score),
        details="Scores within acceptable limits."
    )
