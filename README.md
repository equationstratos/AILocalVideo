# AILocalVideo

Système **local** de génération de vidéos par IA, multi-styles (animation,
anime, cartoon, réaliste, cinématique), basé sur des modèles open-source
HuggingFace `diffusers`. Architecture **API d'abord** (FastAPI) avec une Web UI
par-dessus.

## ⚠️ Attentes réalistes (CPU)

Ce projet cible une machine **sans GPU NVIDIA (CPU seul)**. Aucun système local
de cette taille ne peut égaler des services cloud type Gemini/Sora, qui tournent
sur d'énormes fermes GPU.

Sur CPU, la génération vidéo par diffusion est **très lente** : de plusieurs
minutes à plusieurs heures pour quelques secondes de vidéo. Le premier lancement
**télécharge** aussi les modèles (plusieurs Go). C'est utilisable pour
expérimenter, pas pour produire en volume.

L'architecture est **modulaire** : on peut brancher des modèles GPU plus
puissants (LTX-Video, HunyuanVideo, Wan2.1) en ajoutant un seul fichier de
backend, sans réécrire l'application.

## Architecture

```
backend/
  config.py                 réglages (device=cpu, chemins, limites)
  schemas.py                modèles Pydantic (API)
  styles.py                 chargement des presets de styles
  jobs.py                   file de jobs + worker en arrière-plan
  main.py                   application FastAPI
  generation/
    base.py                 interface VideoBackend (abstraction)
    registry.py             registre des backends
    animatediff_backend.py  AnimateDiff (SD1.5) — styles animation
    cogvideox_backend.py    CogVideoX-2B — styles réaliste/cinématique
config/styles.yaml          presets de styles (modèle + prompts + défauts)
frontend/                   Web UI (à venir, phase 2)
outputs/                    vidéos générées (.mp4)
tests/                      tests pytest
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

> `torch` en version CPU. Si vous avez un GPU NVIDIA, installez la build CUDA de
> torch et passez `AILV_DEVICE=cuda`.

## Lancer l'API

```bash
ailocalvideo            # ou : uvicorn backend.main:app --reload
```

Puis ouvrez http://127.0.0.1:8000/docs pour la documentation interactive.

### Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET  | `/api/health` | vérification de vie |
| GET  | `/api/styles` | styles disponibles + défauts |
| GET  | `/api/backends` | moteurs disponibles + device |
| POST | `/api/generate` | lance un job → `{job_id}` |
| GET  | `/api/jobs/{id}` | statut + progression |
| GET  | `/api/jobs/{id}/video` | récupère le `.mp4` |

### Exemple

```bash
# Lancer une génération
curl -X POST http://127.0.0.1:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat surfing a wave", "style": "animation", "num_frames": 16}'
# => {"job_id": "..."}

# Suivre la progression
curl http://127.0.0.1:8000/api/jobs/<job_id>

# Télécharger la vidéo une fois "status": "done"
curl http://127.0.0.1:8000/api/jobs/<job_id>/video -o out.mp4
```

## Styles

Les styles sont définis dans `config/styles.yaml`. Chaque style choisit un
backend, ajoute un suffixe de prompt, un prompt négatif et des paramètres par
défaut (frames, steps, fps, résolution…). Ajoutez vos propres styles en éditant
ce fichier — aucun code à modifier.

## Configuration

Variables d'environnement (préfixe `AILV_`) :

- `AILV_DEVICE` : `cpu` (défaut) ou `cuda`
- `AILV_PORT`, `AILV_HOST`
- `AILV_MAX_NUM_FRAMES`, `AILV_MAX_STEPS`, `AILV_MAX_RESOLUTION` : garde-fous

## Tests

```bash
pytest -q
```

Les tests n'effectuent **aucun** téléchargement ni génération réelle : les
backends lourds sont remplacés par des doublures.

## Feuille de route

- [ ] Web UI (Vite + React) : formulaire, suivi de progression, lecteur vidéo
- [ ] Backends GPU : LTX-Video, HunyuanVideo, Wan2.1
- [ ] Persistance des jobs (SQLite)
- [ ] Image-to-video, audio, upscaling, interpolation de frames
```

## Ajouter un backend

1. Créez `backend/generation/mon_backend.py`
2. Sous-classez `VideoBackend`, implémentez `load()` et `generate()`
3. Décorez la classe avec `@register_backend`
4. Importez-le dans `registry.py`
5. Référencez son `name` dans un style de `config/styles.yaml`
