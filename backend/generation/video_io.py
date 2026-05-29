"""Encodage des frames en fichier vidéo.

Isolé dans son propre module pour que l'import lourd de ``diffusers`` reste
paresseux (et facilement remplaçable dans les tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import List


def export_to_video(frames: List, path: Path | str, fps: int) -> None:
    """Encode une liste de frames (images PIL) en MP4 via diffusers."""
    from diffusers.utils import export_to_video as _export

    _export(frames, str(path), fps=fps)
