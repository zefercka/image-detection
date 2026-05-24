import gc
import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    ImageClassificationPipeline,
    Owlv2ForObjectDetection,
    Owlv2Processor,
)
from transformers import pipeline as hf_pipeline

from app.config import settings

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_logger = logging.getLogger(__name__)


@dataclass
class SigLIPResult:
    """Результат классификации SigLIP2 для одного кадра.

    Маппинг на категории и промпты c их уверенностью. Уверенности в диапазоне 0.0-1.0.
    """

    by_category: dict[str, float]
    by_prompt: dict[str, float]


@dataclass
class OwlResult:
    """Результат детекции OWLv2 для одного кадра.

    Маппинг на категории и промпты c максимальной уверенностью.
    Уверенности в диапазоне 0.0-1.0.
    """

    by_category: dict[str, float]
    by_prompt: dict[str, float]
    detections: list[dict]  # [{prompt, score, box:[x1,y1,x2,y2]}]


def _get_device() -> str:
    if torch.cuda.is_available():
        device_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        return f"cuda:{device_id}"
    return "cpu"


class OwlDetector:
    """OWLv2 классификатор объектов на кадрах."""

    _model: Owlv2ForObjectDetection | None = None
    _processor: Owlv2Processor | None = None

    @classmethod
    def get_model(cls) -> tuple[Owlv2ForObjectDetection, Owlv2Processor]:
        if cls._model is None or cls._processor is None:
            device = _get_device()
            dtype = torch.float16 if "cuda" in device else torch.float32

            _logger.info("[OWLv2] loading %s ...", settings.OWL_MODEL_PATH)
            cls._processor = Owlv2Processor.from_pretrained(settings.OWL_MODEL_PATH)
            cls._model = Owlv2ForObjectDetection.from_pretrained(
                settings.OWL_MODEL_PATH,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device).eval()

            if "cuda" in device:
                _logger.info(
                    "[OWLv2] ready - VRAM %d MB",
                    torch.cuda.memory_allocated() // 1024**2,
                )
            else:
                _logger.info("[OWLv2] ready")

        return cls._model, cls._processor

    @classmethod
    @torch.no_grad()
    def detect_frames(
        cls,
        frames: list[np.ndarray],
        categories: dict[str, list[str]],
        threshold: float | None = None,
    ) -> list[OwlResult]:
        """Детектит объекты на кадрах по маппингу {категория: [промпты]} через OWLv2.

        Промпты разбиваются на чанки размером OWL_PROMPT_CHUNK_SIZE, что позволяет
        обрабатывать больше кадров за раз при том же объёме VRAM.

        Args:
            frames (list[np.ndarray]): Список кадров в формате BGR.
            categories (dict[str, list[str]]): Маппинг категория -> список промптов.
            threshold (float | None): Порог уверенности. По умолчанию OWL_THRESHOLD.

        Returns:
            list[OwlResult]: Результаты детекции для каждого кадра.

        """
        flat_prompts: list[str] = [p for ps in categories.values() for p in ps]
        prompt_to_cat: dict[str, str] = {
            p: cat for cat, ps in categories.items() for p in ps
        }

        if not flat_prompts:
            return [
                OwlResult(by_category={}, by_prompt={}, detections=[])
                for _ in frames
            ]

        model, processor = cls.get_model()
        device = _get_device()
        thr = threshold if threshold is not None else settings.OWL_THRESHOLD

        pil_images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for f in frames
        ]
        target_sizes = torch.tensor(
            [(img.height, img.width) for img in pil_images],
            device=device,
        )

        # Аккумуляторы результатов — по одному на кадр
        acc_by_prompt: list[dict[str, float]] = [
            dict.fromkeys(flat_prompts, 0.0) for _ in frames
        ]
        acc_by_category: list[dict[str, float]] = [
            dict.fromkeys(categories, 0.0) for _ in frames
        ]
        acc_detections: list[list[dict]] = [[] for _ in frames]

        chunk_size = settings.OWL_PROMPT_CHUNK_SIZE
        prompt_chunks = [
            flat_prompts[i: i + chunk_size]
            for i in range(0, len(flat_prompts), chunk_size)
        ]

        for chunk_prompts in prompt_chunks:
            inputs = processor(
                text=[chunk_prompts] * len(pil_images),
                images=pil_images,
                return_tensors="pt",
                padding=True,
            ).to(device)
            if "cuda" in device:
                inputs["pixel_values"] = inputs["pixel_values"].half()

            outputs = model(**inputs)
            raw = processor.post_process_object_detection(
                outputs=outputs,
                threshold=thr,
                target_sizes=target_sizes,
            )

            for frame_idx, r in enumerate(raw):
                for score, label_idx, box in zip(
                    r["scores"].cpu(),
                    r["labels"].cpu().tolist(),
                    r["boxes"].cpu().tolist(),
                    strict=True,
                ):
                    prompt = chunk_prompts[label_idx]
                    s = float(score)
                    acc_by_prompt[frame_idx][prompt] = max(acc_by_prompt[frame_idx][prompt], s)
                    cat = prompt_to_cat[prompt]
                    acc_by_category[frame_idx][cat] = max(acc_by_category[frame_idx][cat], s)
                    acc_detections[frame_idx].append({
                        "prompt": prompt,
                        "score": s,
                        "box": [round(c) for c in box],
                    })

            del inputs, outputs, raw
            if "cuda" in device:
                torch.cuda.empty_cache()

        del pil_images, target_sizes

        return [
            OwlResult(
                by_category=acc_by_category[i],
                by_prompt=acc_by_prompt[i],
                detections=acc_detections[i],
            )
            for i in range(len(frames))
        ]

    @classmethod
    def unload(cls) -> None:
        cls._model = None
        cls._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SigLIPDetector:
    """SigLIP2 классификатор сцен-контекста."""

    _model: AutoModel | None = None
    _processor: AutoProcessor | None = None

    @classmethod
    def get_model(cls) -> tuple[AutoModel, AutoProcessor]:
        if cls._model is None or cls._processor is None:
            device = _get_device()
            dtype = torch.float16 if "cuda" in device else torch.float32
            _logger.info("[SigLIP2] loading %s ...", settings.CLIP_MODEL_PATH)
            cls._processor = AutoProcessor.from_pretrained(settings.CLIP_MODEL_PATH)
            cls._model = AutoModel.from_pretrained(
                settings.CLIP_MODEL_PATH,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device).eval()
            if "cuda" in device:
                _logger.info(
                    "[CLIP] ready - VRAM %d MB",
                    torch.cuda.memory_allocated() // 1024**2,
                )
            else:
                _logger.info("[CLIP] ready")

        return cls._model, cls._processor

    @classmethod
    @torch.no_grad()
    def detect_frames(
        cls,
        frames: list[np.ndarray],
        categories: dict[str, list[str]],
    ) -> list[SigLIPResult]:
        """Классифицировать батч кадров по маппингу {категория: [промпты]}.

        Args:
            frames (list[np.ndarray]): Список кадров в формате BGR.
            categories (dict[str, list[str]]): Маппинг категория -> список промптов.

        Returns:
            list[SigLIPResult]: Результаты классификации для каждого кадра.

        """
        flat_prompts: list[str] = [p for ps in categories.values() for p in ps]
        prompt_to_cat: dict[str, str] = {
            p: cat for cat, ps in categories.items() for p in ps
        }

        if not flat_prompts:
            return [SigLIPResult(by_category={}, by_prompt={}) for _ in frames]

        model, processor = cls.get_model()
        device = _get_device()
        pil_images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]
        inputs = processor(
            text=flat_prompts,
            images=pil_images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(device)  # ty:ignore[call-non-callable]
        out = model(**inputs)  # ty:ignore[call-non-callable]
        # logits_per_image: [num_images, num_prompts] -> sigmoid per prompt
        probs_batch = out.logits_per_image.sigmoid().cpu().tolist()
        del inputs, out, pil_images
        if "cuda" in device:
            torch.cuda.empty_cache()

        results = []

        for probs in probs_batch:
            by_prompt = dict(zip(flat_prompts, probs, strict=True))
            by_category: dict[str, float] = dict.fromkeys(categories, 0.0)
            for prompt, score in by_prompt.items():
                cat = prompt_to_cat[prompt]
                by_category[cat] = max(by_category[cat], score)
            results.append(SigLIPResult(by_category=by_category, by_prompt=by_prompt))

        return results

    @classmethod
    def unload(cls) -> None:
        cls._model = None
        cls._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class NSFWClassifier:
    """NSFW классификатор.

    Используется Falconsai/nsfw_image_detection бинарный классификатор.
    """

    _pipe: ImageClassificationPipeline | None = None

    @classmethod
    def get_model(cls) -> ImageClassificationPipeline:
        if cls._pipe is None:
            device = _get_device()
            _logger.info("[NSFW] loading %s ...", settings.NSFW_MODEL_PATH)
            cls._pipe = hf_pipeline(
                "image-classification",
                model=settings.NSFW_MODEL_PATH,
                device=0 if "cuda" in device else -1,
            )
            _logger.info("[NSFW] ready")
        return cls._pipe

    @classmethod
    def detect_frames(
        cls,
        frames: list[np.ndarray],
    ) -> list[float]:
        """Возвращает уверенность наличия NSFW-контента для батча кадров.

        Args:
            frames (list[np.ndarray]): Список кадров в формате BGR.

        Returns:
            list[float]: Уверенность наличия NSFW-контента для каждого кадра.

        """
        pipe = cls.get_model()
        pil_images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]
        results = pipe(pil_images)
        return [
            next(
                (float(r["score"]) for r in frame_results
                 if r["label"].lower() == "nsfw"),
                0.0,
            )
            for frame_results in results
        ]

    @classmethod
    def unload(cls) -> None:
        cls._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
