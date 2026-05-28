"""Chargement et validation des presets de styles depuis ``styles.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .config import get_settings


@dataclass(frozen=True)
class Style:
    """Un preset de style résolu."""

    name: str
    label: str
    backend: str
    prompt_prefix: str
    prompt_suffix: str
    negative_prompt: str
    defaults: dict

    def build_prompt(self, user_prompt: str) -> str:
        """Fusionne le prompt utilisateur avec le préfixe/suffixe du style."""
        parts = [self.prompt_prefix.strip(), user_prompt.strip()]
        merged = " ".join(p for p in parts if p)
        return f"{merged}{self.prompt_suffix}"


def _parse_styles(raw: dict) -> dict[str, Style]:
    styles_section = raw.get("styles")
    if not isinstance(styles_section, dict) or not styles_section:
        raise ValueError("styles.yaml: clé racine 'styles' manquante ou vide")

    result: dict[str, Style] = {}
    required_defaults = {"num_frames", "fps", "steps", "guidance", "width", "height"}
    for name, cfg in styles_section.items():
        if "backend" not in cfg:
            raise ValueError(f"Style '{name}': champ 'backend' requis")
        defaults = cfg.get("defaults", {})
        missing = required_defaults - defaults.keys()
        if missing:
            raise ValueError(
                f"Style '{name}': défauts manquants: {sorted(missing)}"
            )
        result[name] = Style(
            name=name,
            label=cfg.get("label", name),
            backend=cfg["backend"],
            prompt_prefix=cfg.get("prompt_prefix", ""),
            prompt_suffix=cfg.get("prompt_suffix", ""),
            negative_prompt=cfg.get("negative_prompt", ""),
            defaults=defaults,
        )
    return result


def load_styles(path: Path | None = None) -> dict[str, Style]:
    """Charge les styles depuis le fichier YAML (chemin par défaut: config)."""
    path = path or get_settings().styles_file
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _parse_styles(raw)


@lru_cache
def get_styles() -> dict[str, Style]:
    """Renvoie les styles chargés une seule fois (mis en cache)."""
    return load_styles()


def get_style(name: str) -> Style:
    """Renvoie un style par nom, ou lève KeyError si introuvable."""
    styles = get_styles()
    if name not in styles:
        raise KeyError(name)
    return styles[name]
