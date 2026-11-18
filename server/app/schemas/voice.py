"""API contracts: what the React app receives from the voice routes."""

from typing import Literal

from pydantic import BaseModel, Field


class TurnTimings(BaseModel):
    stt_ms: int = 0
    llm_ms: int = 0
    tts_ms: int = 0
    total_ms: int = 0


class VoiceTurnResponse(BaseModel):
    """Response of POST /api/voice: one full conversational turn."""

    session_id: str
    skipped: bool = Field(
        default=False,
        description="True when the clip was noise: nothing else is filled in, keep listening",
    )
    transcript: str = ""
    reply: str = ""
    audio_base64: str | None = Field(
        default=None, description="WAV file, base64-encoded; play it with an <audio> element"
    )
    sample_rate: int | None = None
    timings: TurnTimings = Field(default_factory=TurnTimings)


class HealthResponse(BaseModel):
    """Response of GET /health."""

    status: Literal["ok", "degraded"]
    stt: bool
    tts: bool
    llm: bool
