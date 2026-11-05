"""Clients for the services the orchestrator depends on.

Both dependencies are optional at runtime: when they are missing or failing, the
graph falls back to deterministic templates instead of failing the request.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from app.schemas import Passage

LOGGER = logging.getLogger("agent_orchestrator.clients")


class LLMClient(Protocol):
    async def generate(self, prompt: str) -> str | None: ...


class Retriever(Protocol):
    async def search(self, query: str, k: int = 3) -> list[Passage]: ...


class OllamaClient:
    """Calls an Ollama server (llm-server) through POST /api/generate."""

    def __init__(self, base_url: str, model: str, http: httpx.AsyncClient, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.http = http
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    async def generate(self, prompt: str) -> str | None:
        if not self.configured:
            return None
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        try:
            response = await self.http.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            LOGGER.warning("llm-server call failed: %s", exc)
            return None
        text = body.get("response") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            return None
        return text.strip()


class NullRetriever:
    """Placeholder until rag_engine and vector-db exist. Always returns no passages."""

    async def search(self, query: str, k: int = 3) -> list[Passage]:
        return []
