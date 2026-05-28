"""Registre des backends de génération disponibles.

Les classes de backend sont enregistrées par nom. Les instances sont créées
paresseusement et mises en cache afin qu'un même modèle ne soit pas rechargé à
chaque génération.
"""

from __future__ import annotations

from .base import VideoBackend

# nom -> classe de backend
_BACKEND_CLASSES: dict[str, type[VideoBackend]] = {}

# nom -> instance (cache des backends instanciés)
_INSTANCES: dict[str, VideoBackend] = {}


def register_backend(cls: type[VideoBackend]) -> type[VideoBackend]:
    """Décorateur enregistrant une classe de backend par son attribut ``name``."""
    if not getattr(cls, "name", None) or cls.name == "base":
        raise ValueError(f"{cls.__name__} doit définir un attribut 'name' unique")
    _BACKEND_CLASSES[cls.name] = cls
    return cls


def list_backends() -> list[str]:
    return sorted(_BACKEND_CLASSES)


def get_backend(name: str, device: str = "cpu") -> VideoBackend:
    """Renvoie une instance (mise en cache) du backend nommé."""
    if name not in _BACKEND_CLASSES:
        raise KeyError(f"Backend inconnu: {name!r}. Connus: {list_backends()}")
    if name not in _INSTANCES:
        _INSTANCES[name] = _BACKEND_CLASSES[name](device=device)
    return _INSTANCES[name]


def backend_class(name: str) -> type[VideoBackend]:
    return _BACKEND_CLASSES[name]


# Import des backends concrets pour déclencher leur enregistrement.
# Placé en fin de module pour éviter les imports circulaires.
from . import animatediff_backend  # noqa: E402,F401
from . import cogvideox_backend  # noqa: E402,F401
