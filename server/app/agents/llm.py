"""
LLM client over the OpenAI-compatible API.
"""

import re
import time
from typing import cast
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from app.config.config import get_settings
from app.config.logging_setup import get_logger
from app.schemas.llm import ChatMessage, ChatResult

settings = get_settings()
log = get_logger(__name__)

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMService:

    def __init__(self):
        self.model = settings.llm_model
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=1,  
        )
        log.info("llm client: model=%s url=%s", self.model, settings.llm_base_url)

    async def chat(self, messages: list[ChatMessage]) -> ChatResult:
        start = time.perf_counter()
        try:
            payload = cast(list[ChatCompletionMessageParam], [m.model_dump() for m in messages])
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=payload,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
        except (APIConnectionError, APITimeoutError):
            log.exception("llm unreachable at %s (is Ollama running?)", settings.llm_base_url)
            raise
        except Exception:
            log.exception("llm request failed")
            raise

        choice = response.choices[0]
        # Safety net: thinking-mode models can leak <think>...</think> into the reply.
        text = THINK_BLOCK.sub("", choice.message.content or "").strip()
        duration_ms = int((time.perf_counter() - start) * 1000)

        usage = response.usage
        log.info(
            "llm %d ms | prompt %s tok | reply %s tok | finish=%s",
            duration_ms,
            usage.prompt_tokens if usage else "?",
            usage.completion_tokens if usage else "?",
            choice.finish_reason,  # "length" = cut off by max_tokens
        )
        log.debug("llm text: %r", text)
        return ChatResult(text=text, model=self.model, duration_ms=duration_ms)

    async def close(self) -> None:
        await self.client.close()
