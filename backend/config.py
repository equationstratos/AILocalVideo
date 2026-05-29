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
    # Précision sur GPU : "bfloat16" (défaut, recommandé) ou "float16".
    precision: str = "bfloat16"
    # Sur GPU, décharge les sous-modèles inactifs en RAM (enable_model_cpu_offload)
    # pour faire tenir de gros modèles. Mettre False si VRAM abondante = + rapide.
    enable_offload: bool = True

    # Identifiants des modèles, surchargeables par variable d'environnement.
    # CogVideoX : 2B est léger (CPU) ; 5B est bien meilleur (GPU 16Go+).
    cogvideox_model: str = "THUDM/CogVideoX-2b"
    cogvideox_i2v_model: str = "THUDM/CogVideoX-5b-I2V"
    ltx_model: str = "Lightricks/LTX-Video"

    # Chemins
    styles_file: Path = REPO_ROOT / "config" / "styles.yaml"
    outputs_dir: Path = REPO_ROOT / "outputs"

    # Limites de sécurité. Élevées pour le GPU ; sur CPU, gardez de petites
    # valeurs dans les requêtes (les défauts de styles.yaml restent bas).
    max_num_frames: int = 257
    max_steps: int = 60
    max_resolution: int = 1280
    # Nombre max de segments enchaînés (durée = num_segments * num_frames / fps).
    max_segments: int = 120

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
