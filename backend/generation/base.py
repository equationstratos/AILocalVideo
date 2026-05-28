"""Interface commune à tous les moteurs de génération vidéo.

Toute intégration de modèle (AnimateDiff, CogVideoX, et plus tard des modèles
GPU comme LTX-Video) implémente :class:`VideoBackend`. Cela isole le reste de
l'application des détails de chaque pipeline ``diffusers``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

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
    def generate(
        self,
        params: GenerationParams,
        output_path: Path,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Path:
        """Génère une vidéo, l'écrit dans ``output_path`` et renvoie son chemin."""
