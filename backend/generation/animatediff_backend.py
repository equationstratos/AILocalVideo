"""Backend AnimateDiff (Stable Diffusion 1.5) — styles d'animation.

AnimateDiff ajoute un "motion adapter" à un checkpoint SD1.5 pour produire de
courtes animations. C'est l'option la plus raisonnable sur CPU (modèle léger),
même si cela reste lent.

Les imports lourds (torch, diffusers) sont volontairement faits dans
:meth:`load` pour que le module soit importable sans ces dépendances (utile en
test / pour exposer les métadonnées via l'API).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import GenerationParams, ProgressCallback, VideoBackend
from .registry import register_backend

# Modèles HuggingFace utilisés (téléchargés au 1er usage).
MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-2"
BASE_MODEL = "emilianJR/epiCRealism"  # checkpoint SD1.5 polyvalent


@register_backend
class AnimateDiffBackend(VideoBackend):
    name = "animatediff"
    supports_styles = ["animation", "anime", "cartoon"]

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device=device)
        self._pipe = None

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

    def generate(
        self,
        params: GenerationParams,
        output_path: Path,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Path:
        if not self._loaded:
            self.load()

        import torch
        from diffusers.utils import export_to_video

        generator = None
        if params.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(params.seed)

        total = max(params.steps, 1)

        def _on_step(pipe, step_index, timestep, callback_kwargs):
            if progress_cb is not None:
                # Réserve les derniers % à l'encodage/export.
                progress_cb(min((step_index + 1) / total * 0.95, 0.95))
            return callback_kwargs

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

        frames = result.frames[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(output_path), fps=params.fps)
        if progress_cb is not None:
            progress_cb(1.0)
        return output_path
