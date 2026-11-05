"""Agent orchestrator: runs the Analyst -> Cardiologist -> Auditor graph."""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request

from app.clients import NullRetriever, OllamaClient
from app.graph import build_graph
from app.schemas import AnalysisReport, AnalyzeRequest

LOGGER = logging.getLogger("agent_orchestrator")

LLM_URL = os.getenv("LLM_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "cardio-qwen")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    http_client = httpx.AsyncClient(follow_redirects=False)
    llm = OllamaClient(LLM_URL, LLM_MODEL, http_client, timeout=LLM_TIMEOUT_SECONDS)
    app.state.http_client = http_client
    app.state.llm_configured = llm.configured
    app.state.graph = build_graph(llm, NullRetriever())
    yield
    await http_client.aclose()


app = FastAPI(
    title="CardioAgent Agent Orchestrator",
    description="Multi-agent clinical analysis (Analyst, Cardiologist, Auditor).",
    version=os.getenv("SERVICE_VERSION", "1.0.0"),
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["x-request-id"] = request.state.request_id
    return response


@app.get("/health", tags=["health"])
async def health(request: Request) -> dict[str, object]:
    return {
        "status": "ok",
        "service": "agent-orchestrator",
        "llm_configured": request.app.state.llm_configured,
        "rag_configured": False,
    }


@app.post("/analyze", response_model=AnalysisReport, tags=["analysis"])
async def analyze(body: AnalyzeRequest, request: Request) -> AnalysisReport:
    request_id = body.request_id or request.state.request_id
    request.state.request_id = request_id  # the middleware stamps the final id on the response
    try:
        result = await request.app.state.graph.ainvoke({"request": body, "request_id": request_id})
    except Exception as exc:
        LOGGER.exception("pipeline failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="The analysis pipeline failed") from exc
    return result["report"]
