"""Tests de la file de jobs avec un backend factice (sans modèle réel)."""

import time

from backend.generation.base import GenerationParams
from backend.jobs import DONE, ERROR, JobStore


def _params():
    return GenerationParams(
        prompt="p",
        negative_prompt="",
        num_frames=2,
        fps=8,
        steps=2,
        guidance=7.0,
        width=64,
        height=64,
    )


def _wait(store, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get(job_id)
        if job and job.status in (DONE, ERROR):
            return job
        time.sleep(0.02)
    raise AssertionError("Job non terminé dans le délai imparti")


def test_job_runs_with_fake_backend(monkeypatch, tmp_path):
    store = JobStore(device="cpu")
    store.outputs_dir = tmp_path

    class FakeBackend:
        def generate(self, params, output_path, progress_cb=None):
            output_path.write_bytes(b"fake-mp4")
            if progress_cb:
                progress_cb(1.0)
            return output_path

    # Style "animation" existe ; on remplace seulement le backend résolu.
    monkeypatch.setattr("backend.jobs.get_backend", lambda *a, **k: FakeBackend())

    job_id = store.submit("animation", _params())
    job = _wait(store, job_id)
    assert job.status == DONE
    assert job.progress == 1.0
    assert store.video_path(job_id) is not None
    assert store.video_path(job_id).read_bytes() == b"fake-mp4"


def test_job_records_error(monkeypatch, tmp_path):
    store = JobStore(device="cpu")
    store.outputs_dir = tmp_path

    class BoomBackend:
        def generate(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr("backend.jobs.get_backend", lambda *a, **k: BoomBackend())

    job_id = store.submit("animation", _params())
    job = _wait(store, job_id)
    assert job.status == ERROR
    assert "boom" in job.message
