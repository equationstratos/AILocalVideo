"""Backend LTX-Video (Lightricks) — génération rapide et haute qualité.

LTX-Video est un modèle DiT vidéo rapide, adapté aux GPU (idéal 16 Go+), qui
gère des clips relativement longs et le conditionnement par image. Combiné au
chaînage de segments, c'est la meilleure option locale pour des vidéos longues
et fluides.

Contraintes du modèle :
- ``num_frames`` doit être de la forme 8k + 1 (ex. 121, 161, 257).
- largeur/hauteur multiples de 32.
On "snappe" automatiquement ces valeurs.
"""

from __future__ import annotations

from .base import VideoBackend
from .device_utils import apply_memory_optimizations, resolve_dtype
from .registry import register_backend


def _snap_frames(n: int) -> int:
    """Arrondit au plus proche 8k+1 (>= 9)."""
    n = max(n, 9)
    k = round((n - 1) / 8)
    return int(k * 8 + 1)


def _snap_dim(x: int) -> int:
    """Arrondit à un multiple de 32 (>= 32)."""
    return max(32, (int(x) // 32) * 32)


@register_backend
class LTXVideoBackend(VideoBackend):
    name = "ltx"
    supports_styles = ["ltx_realistic", "ltx_cinematic"]
    supports_init_image = True

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device=device)
        self._pipe = None  # texte -> vidéo
        self._i2v = None  # image -> vidéo

    def load(self) -> None:
        if self._loaded:
            return

        from diffusers import LTXPipeline
        from ..config import get_settings

        pipe = LTXPipeline.from_pretrained(
            get_settings().ltx_model, torch_dtype=resolve_dtype()
        )
        apply_memory_optimizations(pipe)
        self._pipe = pipe
        self._loaded = True

    def _i2v_pipe(self):
        if self._i2v is None:
            from diffusers import LTXImageToVideoPipeline
            from ..config import get_settings

            # Partage les poids déjà chargés (pas de second téléchargement).
            if self._pipe is not None:
                self._i2v = LTXImageToVideoPipeline(**self._pipe.components)
                apply_memory_optimizations(self._i2v)
            else:
                self._i2v = LTXImageToVideoPipeline.from_pretrained(
                    get_settings().ltx_model, torch_dtype=resolve_dtype()
                )
                apply_memory_optimizations(self._i2v)
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

        width = _snap_dim(params.width)
        height = _snap_dim(params.height)
        num_frames = _snap_frames(params.num_frames)

        common = dict(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt or None,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            generator=generator,
            callback_on_step_end=_on_step,
        )

        if init_image is None:
            result = self._pipe(**common)
        else:
            image = init_image.resize((width, height))
            result = self._i2v_pipe()(image=image, **common)

        return result.frames[0]
