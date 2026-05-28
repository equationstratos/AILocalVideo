"""Configuration globale de l'application.

Les chemins sont calculés relativement à la racine du dépôt afin que l'app
fonctionne quel que soit le répertoire de lancement. Les réglages peuvent être
surchargés via des variables d'environnement préfixées par ``AILV_``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du dépôt : .../AILocalVideo (deux niveaux au-dessus de ce fichier).
REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Réglages de l'application, surchargeable via variables d'environnement."""

    model_config = SettingsConfigDict(env_prefix="AILV_", extra="ignore")

    # Matériel : "cpu" par défaut (cible du projet). Mettre "cuda" si GPU dispo.
    device: str = "cpu"

    # Chemins
    styles_file: Path = REPO_ROOT / "config" / "styles.yaml"
    outputs_dir: Path = REPO_ROOT / "outputs"

    # Limites de sécurité pour éviter des générations interminables sur CPU.
    max_num_frames: int = 49
    max_steps: int = 50
    max_resolution: int = 1024

    # Serveur
    host: str = "127.0.0.1"
    port: int = 8000

    def ensure_dirs(self) -> None:
        """Crée les répertoires nécessaires s'ils n'existent pas."""
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Renvoie l'instance unique de configuration (mise en cache)."""
    return Settings()
