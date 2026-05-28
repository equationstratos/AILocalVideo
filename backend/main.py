"""Application FastAPI exposant la génération vidéo locale.

Routes principales :
- ``GET  /api/health``           : vérification de vie
- ``GET  /api/styles``           : styles disponibles + défauts
- ``GET  /api/backends``         : moteurs disponibles + device
- ``POST /api/generate``         : lance un job, renvoie un job_id
- ``GET  /api/jobs/{id}``        : statut/progression d'un job
- ``GET  /api/jobs/{id}/video``  : récupère le .mp4 généré
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import get_settings
from .generation.base import GenerationParams
from .generation.registry import get_backend, list_backends
from .jobs import get_job_store
from .schemas import (
    BackendInfo,
    GenerateRequest,
    GenerateResponse,
    JobStatus,
    StyleDefaults,
    StyleInfo,
)
from .styles import get_styles

app = FastAPI(title="AILocalVideo", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # usage local ; restreindre si exposé
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "device": get_settings().device}


@app.get("/api/styles", response_model=list[StyleInfo])
def list_styles() -> list[StyleInfo]:
    out = []
    for name, style in get_styles().items():
        out.append(
            StyleInfo(
                name=name,
                label=style.label,
                backend=style.backend,
                defaults=StyleDefaults(**style.defaults),
            )
        )
    return out


@app.get("/api/backends", response_model=list[BackendInfo])
def list_backend_info() -> list[BackendInfo]:
    device = get_settings().device
    out = []
    for name in list_backends():
        backend = get_backend(name, device=device)
        out.append(
            BackendInfo(
                name=name,
                device=device,
                supports_styles=backend.supports_styles,
            )
        )
    return out


def _resolve_params(req: GenerateRequest) -> tuple[str, GenerationParams]:
    """Fusionne la requête avec les défauts du style et applique les limites."""
    styles = get_styles()
    if req.style not in styles:
        raise HTTPException(status_code=404, detail=f"Style inconnu: {req.style}")
    style = styles[req.style]
    d = style.defaults
    settings = get_settings()

    num_frames = min(req.num_frames or d["num_frames"], settings.max_num_frames)
    steps = min(req.steps or d["steps"], settings.max_steps)
    width = min(req.width or d["width"], settings.max_resolution)
    height = min(req.height or d["height"], settings.max_resolution)

    params = GenerationParams(
        prompt=style.build_prompt(req.prompt),
        negative_prompt=req.negative_prompt
        if req.negative_prompt is not None
        else style.negative_prompt,
        num_frames=num_frames,
        fps=req.fps or d["fps"],
        steps=steps,
        guidance=req.guidance if req.guidance is not None else d["guidance"],
        width=width,
        height=height,
        seed=req.seed,
    )
    return req.style, params


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    style, params = _resolve_params(req)
    job_id = get_job_store().submit(style, params)
    return GenerateResponse(job_id=job_id)


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    job = get_job_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job introuvable")
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        has_video=job.output_path is not None and job.status == "done",
    )


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    path = get_job_store().video_path(job_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Vidéo non disponible")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")


def run() -> None:
    """Point d'entrée du script ``ailocalvideo`` (lance uvicorn)."""
    import uvicorn

    settings = get_settings()
    settings.ensure_dirs()
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
