from typing import Literal

from pydantic import BaseModel, Field

# "tool" comes in with the booking tools (results of check_availability etc.)
Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    """One message in the conversation history sent to the LLM.

    Same shape for Ollama and vLLM/OpenAI, so history can be stored
    without caring which backend answers it.
    """

    role: Role
    content: str


class ChatResult(BaseModel):
    """Result of one LLM call."""

    text: str
    model: str
    duration_ms: int = Field(ge=0)
