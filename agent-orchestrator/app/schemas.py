"""Pydantic v2 schemas: the contract between gateway, orchestrator and clients."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RiskLevel = Literal["low", "moderate", "high", "indeterminate"]


class Probabilities(BaseModel):
    normal: float = Field(ge=0.0, le=1.0)
    pathological: float = Field(ge=0.0, le=1.0)


class AudioResult(BaseModel):
    """What the audio-engine returns. Extra fields are tolerated so the audio
    contract can grow without breaking the orchestrator."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    label: str = "unknown"
    probabilities: Probabilities
    features: dict[str, float] = Field(default_factory=dict)
    model_loaded: bool = False
    model_version: str | None = None


class AnalyzeRequest(BaseModel):
    """Body sent by the api-gateway to POST /analyze."""

    audio: AudioResult
    patient_id: str | None = Field(default=None, max_length=64)
    metadata: str | None = Field(default=None, max_length=2000)
    request_id: str | None = Field(default=None, max_length=128)


class AnalystOutput(BaseModel):
    risk_level: RiskLevel
    pathological_probability: float = Field(ge=0.0, le=1.0)
    low_confidence: bool
    summary: str


class Passage(BaseModel):
    source: str
    text: str
    score: float | None = None


class CardiologistOutput(BaseModel):
    assessment: str
    recommendations: list[str]
    passages: list[Passage] = Field(default_factory=list)
    llm_used: bool
    rag_used: bool


class AuditResult(BaseModel):
    status: Literal["approved", "corrected"]
    issues: list[str] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    """Final structured report returned to the client (validated by Pydantic)."""

    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    patient_id: str | None
    prediction: str
    probabilities: Probabilities
    risk_level: RiskLevel
    analyst_summary: str
    clinical_assessment: str
    recommendations: list[str]
    sources: list[Passage]
    audit: AuditResult
    llm_used: bool
    rag_used: bool
    model_loaded: bool
    disclaimer: str
