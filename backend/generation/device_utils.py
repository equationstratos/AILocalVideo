"""Utilitaires de gestion du matériel (dtype, optimisations mémoire).

Centralise les décisions dépendantes du device pour éviter de les dupliquer
dans chaque backend.
"""

from __future__ import annotations

from ..config import get_settings


def resolve_dtype():
    """Renvoie le dtype torch adapté au device et au réglage de précision."""
    import torch

    settings = get_settings()
    if settings.device == "cpu":
        return torch.float32
    # GPU : bfloat16 par défaut (recommandé pour CogVideoX/LTX), sinon float16.
    if settings.precision == "float16":
        return torch.float16
    return torch.bfloat16


def apply_memory_optimizations(pipe) -> None:
    """Applique les optimisations mémoire adaptées au device.

    - GPU : ``enable_model_cpu_offload`` (déplace les sous-modèles inactifs en RAM)
      pour faire tenir de gros modèles dans la VRAM, + tiling/slicing du VAE.
    - CPU : ``enable_sequential_cpu_offload`` n'a pas de sens (pas de GPU) ; on
      garde simplement le modèle en RAM avec tiling pour limiter l'empreinte.
    """
    settings = get_settings()

    if settings.device == "cpu":
        pipe.to("cpu")
    elif settings.enable_offload:
        # Compromis VRAM/vitesse : idéal pour gros modèles (CogVideoX-5B…).
        pipe.enable_model_cpu_offload()
    else:
        # Tout en VRAM : le plus rapide si la mémoire suffit.
        pipe.to(settings.device)

    # Tiling/slicing du VAE : réduit fortement la VRAM lors du décodage vidéo.
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
