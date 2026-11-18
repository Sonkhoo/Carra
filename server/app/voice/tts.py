import asyncio
import io
import re
import time
import unicodedata
import wave
from dataclasses import dataclass

from piper import PiperVoice

from app.config.config import get_settings
from app.config.logging_setup import get_logger

settings = get_settings()
log = get_logger(__name__)

MAX_CHARS = 500  # ~30 s of speech; a runaway LLM reply shouldn't block TTS for a minute
FALLBACK_TEXT = "Sorry, could you say that again?"


@dataclass
class Speech:
    wav: bytes  # a complete .wav file, header included
    sample_rate: int
    audio_s: float  # length of the audio
    duration_ms: int  # how long synthesis took


class PiperService:

    def __init__(self):
        path = settings.tts_voice_path
        if not path.exists():
            raise FileNotFoundError(
                f"Piper voice not found at {path}. Download it with:\n"
                f"  cd server && python -m piper.download_voices "
                f"--data-dir voices {path.stem}"
            )
        start = time.perf_counter()
        # Loads the .onnx model + its .onnx.json config (must sit side by side).
        self.voice = PiperVoice.load(str(path))
        self.sample_rate = self.voice.config.sample_rate  # 22050 for "medium"
        self._lock = asyncio.Lock()
        log.info(
            "piper loaded: voice=%s rate=%d Hz in %.1f s",
            path.stem, self.sample_rate, time.perf_counter() - start,
        )

    def _synthesize_sync(self, text: str) -> Speech:
        """Blocking. Runs in a worker thread, never on the event loop."""
        start = time.perf_counter()

        # Write the WAV into memory instead of a file on disk.
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            # Piper sets channels / sample width / rate on the header itself.
            self.voice.synthesize_wav(text, wav_file)
            frames = wav_file.getnframes()

        return Speech(
            wav=buffer.getvalue(),
            sample_rate=self.sample_rate,
            audio_s=frames / self.sample_rate,
            duration_ms=int((time.perf_counter() - start) * 1000),
        )

    async def synthesize(self, text: str) -> Speech:
        text = sanitize_text(text)

        async with self._lock:
            try:
                result = await asyncio.to_thread(self._synthesize_sync, text)
            except Exception:
                log.exception("tts failed on %d chars", len(text))
                raise

        log.info(
            "tts %d ms | audio %.2f s | %d chars",
            result.duration_ms, result.audio_s, len(text),
        )
        log.debug("tts text: %r", text)
        return result


def sanitize_text(text: str) -> str:
    """Turn an LLM reply into plain sentences a voice can read aloud."""
    text = unicodedata.normalize("NFKC", text or "")  # fancy quotes/widths -> plain
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # stray reasoning
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"https?://\S+", "", text)  # bare URLs are unreadable aloud
    text = re.sub(r"^\s*([-*•]|\d+\.)\s+", "", text, flags=re.MULTILINE)  # list markers
    text = re.sub(r"[*_`#>|~]+", "", text)  # bold, italics, code, headings, tables
    # Drop emoji and control characters; keep letters, digits, punctuation, spaces.
    text = "".join(
        ch for ch in text
        if ch.isspace() or unicodedata.category(ch)[0] in "LNPZ" or ch in "$%&+=@"
    )
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > MAX_CHARS:
        # Cut at the last sentence end inside the limit, so it doesn't stop mid-word.
        cut = text[:MAX_CHARS]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        text = cut[: end + 1] if end > 0 else cut
        log.warning("tts text truncated to %d chars", len(text))

    return text or FALLBACK_TEXT
