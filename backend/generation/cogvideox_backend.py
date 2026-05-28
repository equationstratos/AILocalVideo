"""Backend CogVideoX-2B — styles réaliste / cinématique (texte -> vidéo).

CogVideoX produit des vidéos plus réalistes mais est nettement plus lourd.
Sur CPU, on active l'offload séquentiel et le tiling du VAE pour limiter la RAM.
Attention : la génération peut prendre **très** longtemps sans GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import GenerationParams, ProgressCallback, VideoBackend
from .registry import register_backend

MODEL_ID = "THUDM/CogVideoX-2b"


@register_backend
class CogVideoXBackend(VideoBackend):
    name = "cogvideox"
    supports_styles = ["realistic", "cinematic"]

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device=device)
        self._pipe = None

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        from diffusers import CogVideoXPipeline

        dtype = torch.float32 if self.device == "cpu" else torch.float16
        pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)

        # Stratégie mémoire : offload séquentiel + tiling VAE.
        if self.device == "cpu":
            pipe.to("cpu")
        else:
            pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

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
                progress_cb(min((step_index + 1) / total * 0.95, 0.95))
            return callback_kwargs

        result = self._pipe(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt or None,
            num_frames=params.num_frames,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            num_videos_per_prompt=1,
            generator=generator,
            callback_on_step_end=_on_step,
        )

        frames = result.frames[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(output_path), fps=params.fps)
        if progress_cb is not None:
            progress_cb(1.0)
        return output_path
