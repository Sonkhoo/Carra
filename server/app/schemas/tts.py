from pydantic import BaseModel, Field


class Speech(BaseModel):
    """Result of one text-to-speech call (PiperService.synthesize)."""

    wav: bytes = Field(repr=False, description="Complete .wav file, header included")
    sample_rate: int = Field(gt=0, description="22050 Hz for Piper 'medium' voices")
    audio_s: float = Field(ge=0, description="Length of the audio in seconds")
    duration_ms: int = Field(ge=0, description="How long synthesis took")
