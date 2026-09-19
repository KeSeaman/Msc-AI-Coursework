from typing import List, Dict, Any
from core.protocols import SafetyGateway, SafetyReport, SafetyAction, Retriever, LLMProvider, TelemetryLogger
import json

class MockSafetyGateway(SafetyGateway):
    def check_safety(self, text: str) -> SafetyReport:
        # Dummy logic: if "suicide" is in text, escalate.
        if "suicide" in text.lower():
            return SafetyReport(SafetyAction.ESCALATE, 0.9, "High self harm detected")
        return SafetyReport(SafetyAction.PROCEED, 0.1, "Safe")

class MockTrackARetriever(Retriever):
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        return ["CBT Protocol: Behavioural Activation - schedule positive activities."]

class MockTrackBRetriever(Retriever):
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        return ["Cultural context: 'kufikiria sana' implies deep communal disruption, not just anxiety."]

class DualTrackRetriever(Retriever):
    """Composition of multiple retrievers"""
    def __init__(self, track_a: Retriever, track_b: Retriever):
        self.track_a = track_a
        self.track_b = track_b
        
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        # In a real app we might return structured data, here we just merge strings for the mock
        a_results = self.track_a.retrieve(query, top_k)
        b_results = self.track_b.retrieve(query, top_k)
        return a_results + b_results

class MockLLMProvider(LLMProvider):
    def generate(self, prompt: str, system_prompt: str) -> str:
        return "Niaje! I hear that you are 'kufikiria sana'. Remember to talk to your elders or a close friend today. How about visiting your neighbour?"

class FileTelemetryLogger(TelemetryLogger):
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def log_interaction(self, user_id: str, query: str, response: str, metadata: Dict[str, Any]) -> None:
        log_entry = {
            "user_id": user_id,
            "query": query,
            "response": response,
            "metadata": metadata
        }
        with open(self.file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
