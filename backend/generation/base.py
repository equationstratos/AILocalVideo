"""Interface commune à tous les moteurs de génération vidéo.

Toute intégration de modèle (AnimateDiff, CogVideoX, et plus tard des modèles
GPU comme LTX-Video) implémente :class:`VideoBackend`. Cela isole le reste de
l'application des détails de chaque pipeline ``diffusers``.

Chaque backend produit des *frames* via :meth:`generate_frames`. La méthode
:meth:`generate` (déjà implémentée ici) encode ces frames en MP4. Le découpage
frames/écriture permet à l'orchestrateur de chaînage (``chaining.py``) de
concaténer plusieurs segments en une seule vidéo longue.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

# Callback de progression : reçoit une fraction dans [0.0, 1.0].
ProgressCallback = Callable[[float], None]


@dataclass
class GenerationParams:
    """Paramètres résolus passés à un backend pour une génération."""

    prompt: str
    negative_prompt: str
    num_frames: int
    fps: int
    steps: int
    guidance: float
    width: int
    height: int
    seed: Optional[int] = None


class VideoBackend(ABC):
    """Classe de base pour un moteur de génération vidéo.

    Le pipeline lourd est chargé paresseusement via :meth:`load` pour éviter de
    télécharger/charger un modèle tant qu'aucune génération n'est demandée.
    """

    #: Identifiant unique du backend (référencé dans styles.yaml).
    name: str = "base"

    #: Styles que ce backend sait gérer (purement informatif pour l'API).
    supports_styles: list[str] = []

    #: True si le backend sait conditionner une génération sur une image de
    #: départ (``init_image``), permettant une continuité fluide entre segments.
    supports_init_image: bool = False

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Charge le pipeline (idempotent). Doit positionner ``_loaded``."""

    @abstractmethod
    def generate_frames(
        self,
        params: GenerationParams,
        init_image: Optional[Any] = None,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> List[Any]:
        """Génère et renvoie la liste des frames (images PIL).

        ``init_image`` : image de départ optionnelle pour la continuité entre
        segments. Les backends qui ne la gèrent pas (``supports_init_image`` =
        False) doivent l'ignorer.
        """

    def generate(
        self,
        params: GenerationParams,
        output_path: Path,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Path:
        """Génère une vidéo d'un seul segment et l'écrit en MP4."""
        from .video_io import export_to_video

        if not self._loaded:
            self.load()
        frames = self.generate_frames(params, progress_cb=progress_cb)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, output_path, fps=params.fps)
        if progress_cb is not None:
            progress_cb(1.0)
        return output_path
