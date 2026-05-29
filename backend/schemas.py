"""Schémas Pydantic pour les requêtes et réponses de l'API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class StyleDefaults(BaseModel):
    num_frames: int
    fps: int
    steps: int
    guidance: float
    width: int
    height: int


class StyleInfo(BaseModel):
    """Description d'un style exposée par l'API."""

    name: str
    label: str
    backend: str
    defaults: StyleDefaults


class BackendInfo(BaseModel):
    name: str
    device: str
    supports_styles: list[str]


class GenerateRequest(BaseModel):
    """Requête de génération de vidéo."""

    prompt: str = Field(..., min_length=1, max_length=2000)
    style: str = Field(..., description="Nom d'un style défini dans styles.yaml")

    # Surcharges optionnelles des défauts du style.
    num_segments: int = Field(
        default=1,
        ge=1,
        description="Nombre de clips enchaînés (durée totale ≈ num_segments * num_frames / fps)",
    )
    num_frames: Optional[int] = Field(default=None, ge=1)
    fps: Optional[int] = Field(default=None, ge=1, le=60)
    steps: Optional[int] = Field(default=None, ge=1)
    guidance: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    width: Optional[int] = Field(default=None, ge=64)
    height: Optional[int] = Field(default=None, ge=64)
    seed: Optional[int] = Field(default=None, ge=0)
    negative_prompt: Optional[str] = Field(default=None, max_length=2000)


class GenerateResponse(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    """État d'un job de génération."""

    job_id: str
    status: str  # queued | running | done | error
    progress: float  # 0.0 -> 1.0
    message: Optional[str] = None
    has_video: bool = False
