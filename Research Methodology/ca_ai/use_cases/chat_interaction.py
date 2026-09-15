from core.protocols import SafetyGateway, SafetyAction, Retriever, LLMProvider, TelemetryLogger
from core.prompt_builder import build_scaffolding_prompt

def handle_chat(
    user_id: str,
    message: str,
    safety_gateway: SafetyGateway,
    dual_retriever: Retriever,
    llm_provider: LLMProvider,
    telemetry: TelemetryLogger
) -> str:
    """
    Orchestrates the chat interaction keeping functions small and pure-ish.
    Relies purely on injected dependencies (Dependency Inversion).
    """
    # 1. Safety Check
    safety_report = safety_gateway.check_safety(message)
    if safety_report.action == SafetyAction.ESCALATE:
        response = "We care about your safety. Please reach out to the toll-free helpline at 1199 immediately."
        telemetry.log_interaction(user_id, message, response, {"escalated": True, "safety_score": safety_report.score})
        return response

    # 2. RAG Retrieval
    contexts = dual_retriever.retrieve(message)
    
    # Normally we'd split contexts explicitly, but here we just pass the mock array
    # In reality, DualTrackRetriever might return a Tuple[List, List]
    clinical_context = [c for c in contexts if "CBT" in c]
    cultural_context = [c for c in contexts if "Cultural" in c]

    # 3. Prompt Building
    system_prompt = build_scaffolding_prompt(clinical_context, cultural_context)

    # 4. LLM Generation
    response = llm_provider.generate(message, system_prompt)

    # 5. Telemetry
    telemetry.log_interaction(user_id, message, response, {"escalated": False, "safety_score": safety_report.score})
    
    return response
