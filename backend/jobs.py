"""File de jobs de génération vidéo, exécutés en arrière-plan.

Sur CPU, une seule génération a du sens à la fois : un unique thread worker
consomme une file FIFO. Les jobs sont conservés en mémoire (pas de DB pour le
MVP) — ils sont donc perdus au redémarrage du serveur.
"""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import get_settings
from .generation.base import GenerationParams
from .generation.registry import get_backend
from .styles import get_style

# Statuts possibles d'un job.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"


@dataclass
class Job:
    job_id: str
    style: str
    params: GenerationParams
    status: str = QUEUED
    progress: float = 0.0
    message: Optional[str] = None
    output_path: Optional[Path] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class JobStore:
    """Gère la file et l'exécution des jobs via un thread worker unique."""

    def __init__(self, device: str | None = None) -> None:
        settings = get_settings()
        self.device = device or settings.device
        self.outputs_dir = settings.outputs_dir
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, Job] = {}
        self._jobs_lock = threading.Lock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._started = False
        self._start_lock = threading.Lock()

    # ----- cycle de vie du worker -------------------------------------------
    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            self._worker = threading.Thread(
                target=self._run_worker, name="ailv-worker", daemon=True
            )
            self._worker.start()
            self._started = True

    # ----- API publique -----------------------------------------------------
    def submit(self, style: str, params: GenerationParams) -> str:
        self.start()
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id, style=style, params=params)
        with self._jobs_lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        return job_id

    def get(self, job_id: str) -> Optional[Job]:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def video_path(self, job_id: str) -> Optional[Path]:
        job = self.get(job_id)
        if job and job.status == DONE and job.output_path:
            return job.output_path
        return None

    # ----- exécution --------------------------------------------------------
    def _run_worker(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                self._queue.task_done()
                continue
            self._process(job)
            self._queue.task_done()

    def _process(self, job: Job) -> None:
        self._update(job, status=RUNNING, progress=0.0, message=None)
        try:
            style = get_style(job.style)
            backend = get_backend(style.backend, device=self.device)
            output_path = self.outputs_dir / f"{job.job_id}.mp4"

            def _progress(frac: float) -> None:
                self._update(job, progress=float(frac))

            backend.generate(job.params, output_path, progress_cb=_progress)
            self._update(
                job, status=DONE, progress=1.0, output_path=output_path
            )
        except Exception as exc:  # noqa: BLE001 - on remonte l'erreur au job
            self._update(job, status=ERROR, message=str(exc))

    def _update(self, job: Job, **changes) -> None:
        with job._lock:
            for key, value in changes.items():
                setattr(job, key, value)


# Instance partagée par l'application.
_store: Optional[JobStore] = None


def get_job_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore()
    return _store
