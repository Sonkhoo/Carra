from pydantic import BaseModel, Field


class Transcript(BaseModel):
    """Result of one speech-to-text call (WhisperService.transcribe)."""

    text: str
    no_speech_prob: float = Field(ge=0, le=1, description="High = probably noise, not speech")
    is_speech: bool = Field(description="Has text and no_speech_prob is under the threshold")
    audio_s: float = Field(ge=0, description="Length of the clip in seconds")
    duration_ms: int = Field(ge=0, description="How long transcription took")


EMPTY_TRANSCRIPT = Transcript(
    text="", no_speech_prob=1.0, is_speech=False, audio_s=0.0, duration_ms=0
)
