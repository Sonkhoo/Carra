
import time
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config.config import get_settings
from app.config.logging_setup import configure_logging, get_logger
from app.agents.llm import LLMService
from app.schemas.llm import ChatMessage
from app.voice.stt import WhisperService
from app.voice.tts import PiperService

configure_logging()  # must run before any logger is used
log = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm_service = None
    try:
        start = time.perf_counter()
        app.state.tts_service = PiperService()
        app.state.stt_service = WhisperService()
        app.state.llm_service = LLMService()

        # Warm up so the first caller doesn't pay one-time setup costs.
        await app.state.tts_service.synthesize("Hello.")
        await app.state.stt_service.transcribe(np.zeros(16000, dtype=np.float32))
        try:
            # First request makes Ollama load the model into VRAM (several seconds).
            await app.state.llm_service.chat([ChatMessage(role="user", content="Hi")])
        except Exception:
            # Don't block startup: STT/TTS still work, and Ollama may come up later.
            log.warning("llm warm-up failed; is Ollama running? continuing without it")
        log.info("startup done in %.1f s", time.perf_counter() - start)

        yield
    finally:
        log.info("shutting down")
        if app.state.llm_service:
            await app.state.llm_service.close()


app = FastAPI(
    title="Carra",
    description="A self-hosted voice assistant that answers clinic questions and books, reschedules and cancels appointments by voice.",
    version="0.1.0",
    lifespan=lifespan,
)



settings = get_settings()

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Cache-Control", "no-store")
    if not request.url.path.startswith(("/docs", "/redoc", "/openapi.json")):
        response.headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
    if settings.https_only:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response



app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # exact origins, never "*"
    allow_credentials=False,  # no cookies or auth headers yet; flip when you add login
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Session-Id"],
    max_age=600,  # browser caches the preflight answer for 10 min
)

if __name__ == "__main__":
    # 127.0.0.1, not 0.0.0.0: the browser mic only works on localhost or HTTPS anyway,
    # and this keeps the server off your Wi-Fi network.
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
