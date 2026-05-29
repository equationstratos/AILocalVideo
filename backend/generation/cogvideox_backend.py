"""Backend CogVideoX — styles réaliste / cinématique (texte -> vidéo).

- Premier segment : texte -> vidéo (``CogVideoXPipeline``).
- Segments suivants : image -> vidéo (``CogVideoXImageToVideoPipeline``) en
  partant de la dernière frame, ce qui assure une **vraie continuité** lors du
  chaînage de segments.

Le modèle est configurable : 2B (léger, CPU) ou 5B (bien meilleur, GPU). Voir
``cogvideox_model`` / ``cogvideox_i2v_model`` dans la config.

Imports lourds (torch/diffusers) faits dans ``load`` pour garder le module
importable sans ces dépendances (tests, métadonnées API).
"""

from __future__ import annotations

from .base import VideoBackend
from .device_utils import apply_memory_optimizations, resolve_dtype
from .registry import register_backend


@register_backend
class CogVideoXBackend(VideoBackend):
    name = "cogvideox"
    supports_styles = ["realistic", "cinematic"]
    supports_init_image = True

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device=device)
        self._pipe = None  # texte -> vidéo
        self._i2v = None  # image -> vidéo (continuité)

    def load(self) -> None:
        if self._loaded:
            return

        from diffusers import CogVideoXPipeline
        from ..config import get_settings

        dtype = resolve_dtype()
        pipe = CogVideoXPipeline.from_pretrained(
            get_settings().cogvideox_model, torch_dtype=dtype
        )
        apply_memory_optimizations(pipe)
        self._pipe = pipe
        self._loaded = True

    def _i2v_pipe(self):
        """Charge paresseusement le pipeline image->vidéo (continuité)."""
        if self._i2v is None:
            from diffusers import CogVideoXImageToVideoPipeline
            from ..config import get_settings

            pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                get_settings().cogvideox_i2v_model, torch_dtype=resolve_dtype()
            )
            apply_memory_optimizations(pipe)
            self._i2v = pipe
        return self._i2v

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

        common = dict(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt or None,
            num_frames=params.num_frames,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            num_videos_per_prompt=1,
            generator=generator,
            callback_on_step_end=_on_step,
        )

        if init_image is None:
            result = self._pipe(**common)
        else:
            image = init_image.resize((params.width, params.height))
            result = self._i2v_pipe()(image=image, **common)

        return result.frames[0]
