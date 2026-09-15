from typing import Protocol, List, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass

class SafetyAction(Enum):
    PROCEED = "PROCEED"
    ESCALATE = "ESCALATE"

@dataclass
class SafetyReport:
    action: SafetyAction
    score: float
    details: str

class SafetyGateway(Protocol):
    def check_safety(self, text: str) -> SafetyReport:
        """Evaluates the text and returns a safety report."""
        ...

class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieves relevant context for the query."""
        ...

class LLMProvider(Protocol):
    def generate(self, prompt: str, system_prompt: str) -> str:
        """Generates a response using the LLM."""
        ...

class TelemetryLogger(Protocol):
    def log_interaction(self, user_id: str, query: str, response: str, metadata: Dict[str, Any]) -> None:
        """Logs an interaction for causal analysis."""
        ...
