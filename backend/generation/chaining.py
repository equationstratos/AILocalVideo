"""Chaînage de segments pour produire des vidéos longues.

Les modèles de diffusion vidéo ne génèrent qu'un court clip à la fois (≈ 2-6 s).
Pour obtenir des vidéos de plusieurs secondes/minutes, on enchaîne N segments
et on concatène leurs frames en une seule vidéo.

Continuité : si le backend la supporte (``supports_init_image``), la dernière
frame d'un segment sert d'image de départ au segment suivant, ce qui assure une
transition fluide. Sinon, les segments sont simplement mis bout à bout (des
coupures franches peuvent apparaître).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from .base import GenerationParams, ProgressCallback, VideoBackend
from .video_io import export_to_video


def generate_long_video(
    backend: VideoBackend,
    params: GenerationParams,
    num_segments: int,
    output_path: Path,
    progress_cb: Optional[ProgressCallback] = None,
) -> Path:
    """Génère ``num_segments`` clips enchaînés et les écrit en une seule vidéo.

    Durée totale ≈ ``num_segments * num_frames / fps`` secondes.
    """
    if num_segments < 1:
        raise ValueError("num_segments doit être >= 1")
    if not backend.is_loaded:
        backend.load()

    all_frames: List = []
    init_image = None

    for i in range(num_segments):
        # Variation de seed par segment pour éviter des clips identiques quand
        # il n'y a pas de continuité par image.
        seg_seed = None if params.seed is None else params.seed + i
        seg_params = replace(params, seed=seg_seed)

        def _seg_progress(frac: float, _i=i) -> None:
            if progress_cb is not None:
                progress_cb((_i + max(0.0, min(frac, 1.0))) / num_segments)

        frames = backend.generate_frames(
            seg_params, init_image=init_image, progress_cb=_seg_progress
        )

        # En mode continuité, la 1re frame reprend l'image de départ : on
        # l'élimine pour ne pas créer un doublon figé dans la vidéo finale.
        if i > 0 and backend.supports_init_image and len(frames) > 1:
            frames = frames[1:]

        all_frames.extend(frames)

        if backend.supports_init_image and frames:
            init_image = frames[-1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(all_frames, str(output_path), fps=params.fps)
    if progress_cb is not None:
        progress_cb(1.0)
    return output_path
