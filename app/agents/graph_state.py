from typing import TypedDict, Optional

class GraphState(TypedDict):
    query: str
    route: str
    query_embedding: list[float]
    retrieved_chunks: list[dict]
    agent_outputs: dict
    final_answer: str
    source_chunks: list[dict]
    conversation_history: list[dict]
    error: Optional[str]
    environment: str
    reflection_note: str
    retry_count: int
