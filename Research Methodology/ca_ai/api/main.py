from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from use_cases.chat_interaction import handle_chat
from adapters.mock_adapters import (
    MockSafetyGateway, 
    MockTrackARetriever, 
    MockTrackBRetriever, 
    DualTrackRetriever,
    MockLLMProvider,
    FileTelemetryLogger
)

app = FastAPI(title="Culturally Adaptive AI API")

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

# Dependency Injection setup
safety_gateway = MockSafetyGateway()
track_a = MockTrackARetriever()
track_b = MockTrackBRetriever()
dual_retriever = DualTrackRetriever(track_a, track_b)
llm_provider = MockLLMProvider()
telemetry = FileTelemetryLogger("telemetry.log")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response_text = handle_chat(
            user_id=request.user_id,
            message=request.message,
            safety_gateway=safety_gateway,
            dual_retriever=dual_retriever,
            llm_provider=llm_provider,
            telemetry=telemetry
        )
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
