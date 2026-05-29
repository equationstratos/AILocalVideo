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

## Vidéos longues (chaînage de segments)

Les modèles ne génèrent qu'un court clip à la fois (≈ 2-6 s). Pour des vidéos
plus longues, le paramètre **`num_segments`** enchaîne plusieurs clips et les
assemble en une seule vidéo.

> Durée totale ≈ `num_segments * num_frames / fps`

```bash
# ~8 s d'animation : 4 segments de 16 frames à 8 fps
curl -X POST http://127.0.0.1:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cat surfing","style":"animation","num_frames":16,"num_segments":4}'
```

- **Continuité** : pour les styles d'animation (backend AnimateDiff), la
  dernière frame d'un segment sert d'image de départ au suivant
  (`supports_init_image`), pour des transitions fluides. Les styles réaliste/
  cinématique (CogVideoX) ne gèrent pas encore cette continuité : leurs segments
  sont mis bout à bout (coupures possibles).
- **Coût** : le temps de rendu est ~proportionnel au nombre de segments. Sur
  CPU, une vidéo longue peut prendre beaucoup de temps.
- Plafond : `AILV_MAX_SEGMENTS` (60 par défaut), `AILV_MAX_NUM_FRAMES`.

## Styles

Les styles sont définis dans `config/styles.yaml`. Chaque style choisit un
backend, ajoute un suffixe de prompt, un prompt négatif et des paramètres par
défaut (frames, steps, fps, résolution…). Ajoutez vos propres styles en éditant
ce fichier — aucun code à modifier.

## GPU (recommandé pour de vrais résultats)

L'app détecte le device via `AILV_DEVICE`. Sur GPU, les modèles se chargent en
`bfloat16` avec offload mémoire automatique.

```bash
# torch CUDA (adapter cu121/cu118 selon votre driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# Lancer en GPU, en utilisant les gros modèles
AILV_DEVICE=cuda \
AILV_HOST=0.0.0.0 \
AILV_COGVIDEOX_MODEL=THUDM/CogVideoX-5b \
ailocalvideo
```

### Modèles et styles disponibles

| Style | Backend | Idéal pour | VRAM conseillée |
|-------|---------|-----------|-----------------|
| `animation`, `anime`, `cartoon` | AnimateDiff (SD1.5) | dessins animés, continuité | 6-8 Go |
| `realistic`, `cinematic` | CogVideoX (2B/5B) | réaliste + **vraie continuité** (I2V) | 16-24 Go |
| `ltx_realistic`, `ltx_cinematic` | LTX-Video | rapide, haute qualité, clips longs | 16 Go+ |

- **Continuité réelle** : pour CogVideoX et LTX, la dernière frame d'un segment
  conditionne le segment suivant (`image-to-video`) → enchaînement fluide pour
  des vidéos longues.
- Réglages GPU : `AILV_PRECISION` (`bfloat16`/`float16`), `AILV_ENABLE_OFFLOAD`
  (`true` = économe en VRAM ; `false` = tout en VRAM, plus rapide si ça tient).

### Reco pour 32 Go de VRAM

Tout en VRAM, modèle 5B + LTX, segments longs :

```bash
AILV_DEVICE=cuda AILV_HOST=0.0.0.0 \
AILV_COGVIDEOX_MODEL=THUDM/CogVideoX-5b \
AILV_ENABLE_OFFLOAD=false \
ailocalvideo
```

Pour une vidéo réaliste de ~30 s avec continuité : style `ltx_cinematic`,
`num_segments` élevé (chaque segment ≈ 6-7 s à 24 fps).

## Configuration

Variables d'environnement (préfixe `AILV_`) :

- `AILV_DEVICE` : `cpu` (défaut) ou `cuda`
- `AILV_PRECISION` : `bfloat16` (défaut GPU) ou `float16`
- `AILV_ENABLE_OFFLOAD` : `true` (défaut) économe en VRAM, `false` = plus rapide
- `AILV_COGVIDEOX_MODEL`, `AILV_COGVIDEOX_I2V_MODEL`, `AILV_LTX_MODEL` : modèles
- `AILV_PORT`, `AILV_HOST`
- `AILV_MAX_NUM_FRAMES`, `AILV_MAX_STEPS`, `AILV_MAX_RESOLUTION`,
  `AILV_MAX_SEGMENTS` : garde-fous

## Tests

```bash
pytest -q
```

Les tests n'effectuent **aucun** téléchargement ni génération réelle : les
backends lourds sont remplacés par des doublures.

## Feuille de route

- [ ] Web UI (Vite + React) : formulaire, suivi de progression, lecteur vidéo
- [x] Backends GPU : LTX-Video, CogVideoX-5B (+I2V)
- [x] Chaînage de segments avec continuité image-to-video
- [ ] Autres modèles : HunyuanVideo, Wan2.1
- [ ] Persistance des jobs (SQLite)
- [ ] Audio, upscaling, interpolation de frames (RIFE)
```

## Ajouter un backend

1. Créez `backend/generation/mon_backend.py`
2. Sous-classez `VideoBackend`, implémentez `load()` et `generate()`
3. Décorez la classe avec `@register_backend`
4. Importez-le dans `registry.py`
5. Référencez son `name` dans un style de `config/styles.yaml`
