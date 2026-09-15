import pytest
from core.protocols import SafetyAction
from core.clinical_rules import evaluate_distress_level
from core.prompt_builder import build_scaffolding_prompt

def test_evaluate_distress_level_safe():
    report = evaluate_distress_level(toxicity_score=0.1, self_harm_score=0.2)
    assert report.action == SafetyAction.PROCEED
    assert report.score == 0.2

def test_evaluate_distress_level_escalate_toxicity():
    report = evaluate_distress_level(toxicity_score=0.9, self_harm_score=0.2)
    assert report.action == SafetyAction.ESCALATE
    assert report.score == 0.9

def test_evaluate_distress_level_escalate_self_harm():
    report = evaluate_distress_level(toxicity_score=0.1, self_harm_score=0.85)
    assert report.action == SafetyAction.ESCALATE
    assert report.score == 0.85

def test_build_scaffolding_prompt():
    clinical = ["Use cognitive restructuring."]
    cultural = ["Validate 'kufikiria sana'."]
    
    prompt = build_scaffolding_prompt(clinical, cultural)
    
    assert "Use cognitive restructuring." in prompt
    assert "Validate 'kufikiria sana'." in prompt
    assert "CRITICAL CONSTRAINTS:" in prompt
    assert "community" in prompt.lower()
