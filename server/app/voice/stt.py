import asyncio
import time
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from app.config.config import get_settings
from app.config.logging_setup import get_logger

settings = get_settings()
log = get_logger(__name__)

SAMPLE_RATE = 16_000  
MIN_SAMPLES = SAMPLE_RATE // 4  # 0.25 s
MAX_SAMPLES = SAMPLE_RATE * 30  # 30 s


@dataclass
class Transcript:
    text: str
    no_speech_prob: float  # 0..1, high = probably noise, not speech
    audio_s: float  # length of the clip
    duration_ms: int  # how long transcription took

    @property
    def is_speech(self) -> bool:
        return bool(self.text) and self.no_speech_prob < settings.stt_no_speech_threshold


EMPTY = Transcript(text="", no_speech_prob=1.0, audio_s=0.0, duration_ms=0)


class WhisperService:

    def __init__(self):
        start = time.perf_counter()
        self.model = WhisperModel(
            settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            download_root=str(settings.stt_download_dir),
        )

        self._lock = asyncio.Lock()
        log.info(
            "whisper loaded: model=%s device=%s compute=%s in %.1f s",
            settings.stt_model, settings.stt_device, settings.stt_compute_type,
            time.perf_counter() - start,
        )

    def _transcribe_sync(self, audio: np.ndarray) -> Transcript:
        """Blocking. Runs in a worker thread, never on the event loop."""
        start = time.perf_counter()

        segments, _info = self.model.transcribe(audio, language="en", beam_size=1)
        segments = list(segments)

        text = "".join(s.text for s in segments).strip()

        no_speech = min((s.no_speech_prob for s in segments), default=1.0)

        return Transcript(
            text=text,
            no_speech_prob=no_speech,
            audio_s=len(audio) / SAMPLE_RATE,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        """`audio` is mono float32 at 16 kHz, exactly what vad-web sends."""
        audio = sanitize_audio(audio)
        if audio.size < MIN_SAMPLES:
            log.info("stt skipped: clip too short (%.2f s)", audio.size / SAMPLE_RATE)
            return EMPTY

        async with self._lock:
            try:
                result = await asyncio.to_thread(self._transcribe_sync, audio)
            except Exception:
                log.exception("stt failed on %.2f s of audio", audio.size / SAMPLE_RATE)
                raise

        # Text length only at INFO: transcripts contain names and phone numbers.
        log.info(
            "stt %d ms | audio %.2f s | no_speech %.2f | speech=%s | %d chars",
            result.duration_ms, result.audio_s, result.no_speech_prob,
            result.is_speech, len(result.text),
        )
        log.debug("stt text: %r", result.text)
        return result


def sanitize_audio(audio: np.ndarray) -> np.ndarray:
    """Coerce whatever arrived into what Whisper expects: 1-D float32 in [-1, 1]."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)  # any shape -> mono 1-D
    if audio.size > MAX_SAMPLES:
        log.warning("stt clip truncated from %.1f s to 30 s", audio.size / SAMPLE_RATE)
        audio = audio[:MAX_SAMPLES]
    # Garbage bytes can decode to NaN/inf, which crash or confuse the model.
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(audio, -1.0, 1.0)
