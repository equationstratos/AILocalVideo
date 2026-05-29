"""Backend AnimateDiff (Stable Diffusion 1.5) — styles d'animation.

AnimateDiff ajoute un "motion adapter" à un checkpoint SD1.5 pour produire de
courtes animations. C'est l'option la plus raisonnable sur CPU (modèle léger),
même si cela reste lent.

Les imports lourds (torch, diffusers) sont volontairement faits dans
:meth:`load` pour que le module soit importable sans ces dépendances (utile en
test / pour exposer les métadonnées via l'API).
"""

from __future__ import annotations

from .base import VideoBackend
from .registry import register_backend

# Modèles HuggingFace utilisés (téléchargés au 1er usage).
MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
BASE_MODEL = "emilianJR/epiCRealism"  # checkpoint SD1.5 polyvalent


@register_backend
class AnimateDiffBackend(VideoBackend):
    name = "animatediff"
    supports_styles = ["animation", "anime", "cartoon"]
    # Continuité activée : on initialise le 1er latent du segment à partir de la
    # dernière frame du segment précédent (img2img léger sur la frame initiale).
    supports_init_image = True

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device=device)
        self._pipe = None
        self._v2v = None

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        from diffusers import (
            AnimateDiffPipeline,
            DDIMScheduler,
            MotionAdapter,
        )

        dtype = torch.float32 if self.device == "cpu" else torch.float16

        adapter = MotionAdapter.from_pretrained(MOTION_ADAPTER, torch_dtype=dtype)
        pipe = AnimateDiffPipeline.from_pretrained(
            BASE_MODEL, motion_adapter=adapter, torch_dtype=dtype
        )
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        pipe.to(self.device)
        # Réduit l'empreinte mémoire (important sur CPU/RAM limitée).
        pipe.enable_vae_slicing()

        self._pipe = pipe
        self._loaded = True

    # Force de régénération en mode continuité : plus c'est bas, plus la frame
    # de départ est préservée (continuité forte) ; plus c'est haut, plus le
    # mouvement est libre.
    INIT_STRENGTH = 0.7

    def _vid2vid_pipe(self):
        """Pipeline video-to-video partageant les poids déjà chargés."""
        if self._v2v is None:
            from diffusers import AnimateDiffVideoToVideoPipeline

            self._v2v = AnimateDiffVideoToVideoPipeline(**self._pipe.components)
            self._v2v.to(self.device)
        return self._v2v

    def generate_frames(self, params, init_image=None, progress_cb=None):
        if not self._loaded:
            self.load()

        import torch

        generator = None
        if params.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(params.seed)

        total = max(params.steps, 1)

        def _on_step(pipe, step_index, timestep, callback_kwargs):
            if progress_cb is not None:
                progress_cb(min((step_index + 1) / total, 1.0))
            return callback_kwargs

        if init_image is None:
            # Premier segment : texte -> vidéo classique.
            result = self._pipe(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt or None,
                num_frames=params.num_frames,
                num_inference_steps=params.steps,
                guidance_scale=params.guidance,
                width=params.width,
                height=params.height,
                generator=generator,
                callback_on_step_end=_on_step,
            )
        else:
            # Segment suivant : on part de la dernière frame précédente en
            # construisant une "vidéo" qui la répète, puis on la ré-anime.
            init = init_image.resize((params.width, params.height))
            seed_video = [init] * params.num_frames
            result = self._vid2vid_pipe()(
                video=seed_video,
                prompt=params.prompt,
                negative_prompt=params.negative_prompt or None,
                strength=self.INIT_STRENGTH,
                num_inference_steps=params.steps,
                guidance_scale=params.guidance,
                generator=generator,
                callback_on_step_end=_on_step,
            )

        return result.frames[0]
