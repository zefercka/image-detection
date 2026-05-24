import gc
import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image

_logger = logging.getLogger(__name__)

_TASK = "<MORE_DETAILED_CAPTION>"


def _get_device() -> str:
    if torch.cuda.is_available():
        device_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        return f"cuda:{device_id}"
    return "cpu"


class CaptionExtractor:
    """Florence-2 для генерации текстовых описаний кадров."""

    _model = None
    _processor = None

    @classmethod
    def load(cls, model_path: str) -> None:
        if cls._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: PLC0415

        device = _get_device()
        dtype = torch.float16 if "cuda" in device else torch.float32

        _logger.info("[Caption] loading %s ...", model_path)
        cls._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        cls._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(device).eval()

        if "cuda" in device:
            _logger.info(
                "[Caption] ready - VRAM %d MB",
                torch.cuda.memory_allocated() // 1024**2,
            )
        else:
            _logger.info("[Caption] ready (CPU)")

    @classmethod
    def unload(cls) -> None:
        cls._model = None
        cls._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    @torch.no_grad()
    def caption_frames(cls, frames: list[np.ndarray]) -> list[str]:
        """Генерирует описания для кадров в формате BGR."""
        if cls._model is None or cls._processor is None:
            raise RuntimeError("CaptionExtractor не загружен, вызовите load() сначала.")

        device = _get_device()
        captions: list[str] = []

        for frame in frames:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = cls._processor(text=_TASK, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if "cuda" in device:
                inputs["pixel_values"] = inputs["pixel_values"].half()

            generated_ids = cls._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                num_beams=3,
                use_cache=False,
            )
            generated_text = cls._processor.batch_decode(
                generated_ids, skip_special_tokens=False,
            )[0]
            result = cls._processor.post_process_generation(
                generated_text,
                task=_TASK,
                image_size=(image.width, image.height),
            )
            captions.append(result.get(_TASK, "").strip())

        return captions
