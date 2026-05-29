"""Tests de l'orchestrateur de chaînage de segments (backend factice)."""

from pathlib import Path
from unittest import mock

from backend.generation.base import GenerationParams, VideoBackend


def _params(seed=None):
    return GenerationParams(
        prompt="p",
        negative_prompt="",
        num_frames=4,
        fps=8,
        steps=2,
        guidance=7.0,
        width=64,
        height=64,
        seed=seed,
    )


class FakeFrame:
    """Frame factice qui mémorise l'image de départ utilisée."""

    def __init__(self, tag):
        self.tag = tag

    def resize(self, size):
        return self


class FakeBackend(VideoBackend):
    name = "fake"
    supports_init_image = True

    def __init__(self):
        super().__init__(device="cpu")
        self._loaded = True
        self.calls = []

    def load(self):
        self._loaded = True

    def generate_frames(self, params, init_image=None, progress_cb=None):
        self.calls.append({"seed": params.seed, "init": init_image})
        if progress_cb:
            progress_cb(1.0)
        # 4 frames par segment.
        return [FakeFrame(f"{params.seed}-{i}") for i in range(params.num_frames)]


def test_chaining_concatenates_segments(tmp_path):
    from backend.generation import chaining

    backend = FakeBackend()
    out = tmp_path / "long.mp4"
    seen = {}

    def fake_export(frames, path, fps):
        seen["count"] = len(frames)
        Path(path).write_bytes(b"x")

    with mock.patch.object(chaining, "export_to_video", fake_export):
        chaining.generate_long_video(backend, _params(seed=10), 3, out)

    # 3 segments de 4 frames ; les 2 segments suivants perdent leur 1re frame
    # (continuité) => 4 + 3 + 3 = 10 frames.
    assert seen["count"] == 10
    # La continuité a propagé une image de départ aux segments 2 et 3.
    assert backend.calls[0]["init"] is None
    assert backend.calls[1]["init"] is not None
    # La seed varie par segment.
    assert [c["seed"] for c in backend.calls] == [10, 11, 12]
