"""App settings, read once from the repo-root .env file.

pydantic-settings maps env vars to fields case-insensitively
(LLM_MODEL -> llm_model) and validates types at startup, so a typo
fails fast instead of halfway through a call.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# This file is server/app/config/config.py -> parents[0]=config, [1]=app, [2]=server
SERVER_DIR = Path(__file__).resolve().parents[2]  # .../Carra/server
REPO_ROOT = SERVER_DIR.parent  # .../Carra


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Clinic ---
    clinic_name: str = "Carra Clinic"
    clinic_timezone: str = "Asia/Kolkata"

    # --- LLM ---
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:4b-instruct"
    llm_api_key: str = "ollama"  # Ollama ignores it; the client requires one
    llm_max_tokens: int = 160  # spoken replies are short
    llm_temperature: float = 0.7
    llm_timeout_s: float = 60.0

    # --- Speech-to-text (faster-whisper) ---
    stt_model: str = "small"
    stt_device: str = "cpu"  # "cuda" needs cuBLAS + cuDNN 9 on Windows
    stt_compute_type: str = "int8"  # "float16" on cuda
    stt_no_speech_threshold: float = 0.6

    stt_download_dir: Path = SERVER_DIR / "models" / "whisper"

    # --- Text-to-speech (Piper) ---
    tts_voice_path: Path = SERVER_DIR / "voices" / "en_US-lessac-medium.onnx"

    # --- Frontend / HTTP ---
    # Origins allowed to call the API from a browser. Vite's dev server uses 5173.
    # In .env, set as JSON: CORS_ORIGINS=["http://localhost:5173","https://carra.example.com"]
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    # Only send HSTS when the API is actually served over HTTPS (prod behind a proxy).
    https_only: bool = False

    # --- Conversation ---
    history_max_messages: int = 12

    # --- Later (booking tools) ---
    database_url: str | None = None
    redis_url: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
