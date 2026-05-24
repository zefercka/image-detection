import asyncio
import json
import logging
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from logging import Logger
from pathlib import Path

import cv2
import dramatiq
import numpy as np

import app.dependencies.rabbitmq.broker  # noqa: F401
from app.config import settings
from app.database.core import get_db_context
from app.database.declarations.detection_models import DetectionModelsEnum
from app.database.declarations.pipeline import PipelineStatusesEnum
from app.database.models import Pipeline, Query
from app.database.repositories import (
    DetectionClassRepository,
    PipelineRepository,
    QueryRepository,
)
from app.dependencies.s3.client import S3Client, get_s3_client
from app.helpers.text_analysis import LLMTextAnalyzer
from app.helpers.video_preprocessing.caption_extractor import CaptionExtractor
from app.helpers.video_preprocessing.frames_detector import (
    NSFWClassifier,
    OwlDetector,
    SigLIPDetector,
)
from app.helpers.video_preprocessing.ocr_extractor import OcrExtractor
from app.tasks._loop import get_loop

_MAX_AGE_S = 60 * 60 * 8   # 8 часов в секундах (для future.result timeout)
_MAX_AGE_MS = _MAX_AGE_S * 1000  # в миллисекундах (для Dramatiq max_age)


class DetectFramesTask(dramatiq.GenericActor):
    """Задача для детекции фреймов."""

    class Meta:
        queue_name = "detect_frames_queue"
        max_retries = 5
        max_age = _MAX_AGE_MS

    def __init__(self) -> None:
        self.logger: Logger = logging.getLogger(__name__)

    def perform(self, pipeline_id: str) -> None:
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._perform_async(pipeline_id),
                loop,
            )
            future.result(timeout=_MAX_AGE_S)
        except RuntimeError:
            get_loop().run_until_complete(self._perform_async(pipeline_id))

    async def _download_frames_to_disk(
        self,
        s3: S3Client,
        bucket_name: str,
        frames_keys: list[str],
        pipeline_id: str,
    ) -> Path:
        """Скачивает все кадры из S3 в temp-каталог параллельно (один раз)."""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"frames_{pipeline_id}_"))
        sem = asyncio.Semaphore(settings.FRAME_UPLOAD_CONCURRENCY)

        async def _save(key: str) -> None:
            async with sem:
                raw = await s3.download_file(bucket_name, key)
                (temp_dir / Path(key).name).write_bytes(raw)

        await asyncio.gather(*[_save(k) for k in frames_keys])
        self.logger.info(
            "[PIPELINE - %s] Скачано %d кадров в %s.",
            pipeline_id, len(frames_keys), temp_dir,
        )
        return temp_dir

    @staticmethod
    def _read_batch_local(
        temp_dir: Path,
        batch_keys: list[str],
    ) -> tuple[list[np.ndarray], list[str]]:
        """Читает батч кадров с локального диска."""
        frames: list[np.ndarray] = []
        valid_keys: list[str] = []
        for key in batch_keys:
            raw = (temp_dir / Path(key).name).read_bytes()
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
                valid_keys.append(key)
        return frames, valid_keys

    async def _get_active_pipeline(self, pipeline_id: str) -> Pipeline | None:
        async with get_db_context() as db:
            pipeline = await PipelineRepository.get(db, pipeline_id)
            if pipeline is None:
                self.logger.warning(
                    "[PIPELINE - %s][Detection] Задача не найдена.",
                    pipeline_id,
                )
                return None

            if pipeline.status == PipelineStatusesEnum.FAILED:
                self.logger.warning(
                    "[PIPELINE - %s][Detection] Задача помечена как FAILED, пропуск.",
                    pipeline_id,
                )
                return None

        return pipeline

    async def _get_frames_list_from_s3(
        self, s3: S3Client, bucket_name: str, pipeline_id: str,
    ) -> list[str] | None:
        frames = await s3.list_files(bucket_name, f"frames/{pipeline_id}/")
        if not frames:
            self.logger.warning(
                "[PIPELINE - %s][Detection] Кадры не найдены в S3.",
                pipeline_id,
            )

            async with get_db_context() as db:
                await PipelineRepository.update_status(
                    db, pipeline_id, PipelineStatusesEnum.FAILED,
                )
            return None

        return frames

    async def _get_active_queries(
        self, pipeline_id: str,
    ) -> tuple[
        list[Query], dict[str, list[str]],
        list[Query], dict[str, list[str]],
        dict[str, int], list[str],
    ]:
        async with get_db_context() as db:
            active_owl_queries = await QueryRepository.get_active(
                db, model_id=DetectionModelsEnum.OWL,
            )
            active_clip_queries = await QueryRepository.get_active(
                db, model_id=DetectionModelsEnum.CLIP,
            )
            detection_classes = await DetectionClassRepository.get_all(db)

        owl_queries = defaultdict(list)
        for query in active_owl_queries:
            owl_queries[query.class_name].append(query.query)

        clip_queries = defaultdict(list)
        for query in active_clip_queries:
            clip_queries[query.class_name].append(query.query)

        min_prompts: dict[str, int] = {
            dc.name: dc.min_prompts_confirm for dc in detection_classes
        }
        class_names: list[str] = [dc.name for dc in detection_classes]

        if not owl_queries:
            self.logger.warning(
                "[PIPELINE - %s][Detection] "
                "Активные OWL-запросы не найдены, OWLv2 пропущен.",
                pipeline_id,
            )

        if not clip_queries:
            self.logger.warning(
                "[PIPELINE - %s][Detection] "
                "Активные CLIP-запросы не найдены, CLIP пропущен.",
                pipeline_id,
            )

        return (
            active_owl_queries, owl_queries,
            active_clip_queries, clip_queries,
            min_prompts, class_names,
        )

    def _batches(
        self, frames_keys: list[str], size: int | None = None,
    ) -> Iterator[list[str]]:
        size = size or settings.IMAGE_PROCESSING_BATCH_SIZE
        for start in range(0, len(frames_keys), size):
            yield frames_keys[start:start + size]

    async def _clip_pass(  # noqa: PLR0913
        self,
        temp_dir: Path,
        frames_keys: list[str],
        pipeline_id: str,
        active_queries: list[Query],
        queries: dict[str, list[str]],
        detections: dict,
    ) -> None:
        prompt_to_query = {q.query: q for q in active_queries}
        processed = 0
        for batch_keys in self._batches(frames_keys):
            batch_frames, valid_keys = self._read_batch_local(temp_dir, batch_keys)
            clip_raw = SigLIPDetector.detect_frames(batch_frames, queries)
            for frame_key, frame in zip(valid_keys, clip_raw, strict=True):
                frame_id = Path(frame_key).stem
                for prompt, score in frame.by_prompt.items():
                    q = prompt_to_query.get(prompt)
                    if q is not None and score >= q.threshold:
                        detections["clip"].setdefault(frame_id, []).append({
                            "class": q.class_name,
                            "prompt": prompt,
                            "score": score,
                        })
            del batch_frames, clip_raw
            processed += len(valid_keys)
            self.logger.info(
                "[PIPELINE - %s][CLIP] Обработано %d/%d кадров.",
                pipeline_id, processed, len(frames_keys),
            )
        SigLIPDetector.unload()

    async def _owl_pass(  # noqa: PLR0913
        self,
        temp_dir: Path,
        frames_keys: list[str],
        pipeline_id: str,
        active_queries: list[Query],
        queries: dict[str, list[str]],
        detections: dict,
    ) -> None:
        prompt_to_query = {q.query: q for q in active_queries}
        processed = 0
        for batch_keys in self._batches(frames_keys, size=settings.OWL_BATCH_SIZE):
            batch_frames, valid_keys = self._read_batch_local(temp_dir, batch_keys)
            owl_raw = OwlDetector.detect_frames(batch_frames, queries)
            for frame_key, frame in zip(valid_keys, owl_raw, strict=True):
                frame_id = Path(frame_key).stem
                for prompt, score in frame.by_prompt.items():
                    q = prompt_to_query.get(prompt)
                    if q is not None and score >= q.threshold:
                        detections["owlv2"].setdefault(frame_id, []).append({
                            "class": q.class_name,
                            "prompt": prompt,
                            "score": score,
                        })
            del batch_frames, owl_raw
            processed += len(valid_keys)
            self.logger.info(
                "[PIPELINE - %s][OWL] Обработано %d/%d кадров.",
                pipeline_id, processed, len(frames_keys),
            )
        OwlDetector.unload()

    async def _nsfw_pass(
        self,
        temp_dir: Path,
        frames_keys: list[str],
        pipeline_id: str,
        detections: dict,
    ) -> None:
        processed = 0
        for batch_keys in self._batches(frames_keys):
            batch_frames, valid_keys = self._read_batch_local(temp_dir, batch_keys)
            nsfw_raw = NSFWClassifier.detect_frames(batch_frames)
            for frame_key, score in zip(valid_keys, nsfw_raw, strict=True):
                if score >= settings.NUDE_THRESHOLD:
                    frame_id = Path(frame_key).stem
                    detections["nsfw"][frame_id] = score
            del batch_frames, nsfw_raw
            processed += len(valid_keys)
            self.logger.info(
                "[PIPELINE - %s][NSFW] Обработано %d/%d кадров.",
                pipeline_id, processed, len(frames_keys),
            )
        NSFWClassifier.unload()

    async def _ocr_pass(  # noqa: PLR0913
        self,
        temp_dir: Path,
        frames_keys: list[str],
        pipeline_id: str,
        class_names: list[str],
        fps: int,
        frames_interval: int,
    ) -> dict:
        if not settings.GEMINI_API_KEY:
            self.logger.warning(
                "[PIPELINE - %s][OCR] GEMINI_API_KEY не задан, OCR-анализ пропущен.",
                pipeline_id,
            )
            return {}

        OcrExtractor.load()
        frame_texts: list[tuple[int, str]] = []
        processed = 0

        for batch_keys in self._batches(frames_keys):
            batch_frames, valid_keys = self._read_batch_local(temp_dir, batch_keys)
            texts = OcrExtractor.extract_frames(batch_frames)
            for frame_key, text in zip(valid_keys, texts, strict=True):
                stem = Path(frame_key).stem
                try:
                    frame_idx = int(stem.split("_", 1)[1])
                except (ValueError, IndexError):
                    continue
                frame_texts.append((frame_idx, text))
            del batch_frames
            processed += len(valid_keys)
            self.logger.info(
                "[PIPELINE - %s][OCR] Обработано %d/%d кадров.",
                pipeline_id, processed, len(frames_keys),
            )

        OcrExtractor.unload()

        frame_texts.sort(key=lambda x: x[0])
        unique = OcrExtractor.deduplicate(frame_texts)

        if not unique:
            self.logger.info(
                "[PIPELINE - %s][OCR] Текст на экране не обнаружен.",
                pipeline_id,
            )
            return {}

        step = max(frames_interval, 1)
        segments = [
            {
                "start": round(frame_idx / fps, 2),
                "end": round((frame_idx + step) / fps, 2),
                "text": text,
            }
            for frame_idx, text in unique
        ]
        self.logger.info(
            "[PIPELINE - %s][OCR] Найдено %d уникальных фрагментов текста, "
            "передаю в LLM.",
            pipeline_id, len(segments),
        )

        analyzer = LLMTextAnalyzer(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            base_url=settings.GEMINI_BASE_URL,
        )
        analysis = await analyzer.analyze({"segments": segments}, class_names)
        self.logger.info(
            "[PIPELINE - %s][OCR] LLM-анализ завершён: %s.",
            pipeline_id, analysis.get("summary", {}),
        )
        return analysis

    async def _caption_pass(
        self,
        temp_dir: Path,
        flagged_keys: list[str],
        pipeline_id: str,
        detections: dict,
    ) -> None:
        if not flagged_keys:
            return

        CaptionExtractor.load(settings.CAPTION_MODEL_PATH)
        processed = 0
        for batch_keys in self._batches(flagged_keys, size=settings.CAPTION_BATCH_SIZE):
            batch_frames, valid_keys = self._read_batch_local(temp_dir, batch_keys)
            captions = CaptionExtractor.caption_frames(batch_frames)
            for frame_key, caption in zip(valid_keys, captions, strict=True):
                if caption:
                    detections["captions"][Path(frame_key).stem] = caption
            del batch_frames
            processed += len(valid_keys)
            self.logger.info(
                "[PIPELINE - %s][Caption] Описание %d/%d кадров.",
                pipeline_id, processed, len(flagged_keys),
            )
        CaptionExtractor.unload()

    async def _caption_analysis_pass(
        self,
        pipeline_id: str,
        class_names: list[str],
        detections: dict,
    ) -> None:
        captions = detections.get("captions", {})
        if not captions or not settings.GEMINI_API_KEY:
            return
        analyzer = LLMTextAnalyzer(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            base_url=settings.GEMINI_BASE_URL,
        )
        caption_classes = await analyzer.analyze_captions(captions, class_names)
        detections["caption_classes"] = caption_classes
        self.logger.info(
            "[PIPELINE - %s][Caption] LLM классифицировал %d кадров.",
            pipeline_id, len(caption_classes),
        )

    async def _process_frames(  # noqa: PLR0913
        self,
        s3: S3Client,
        bucket_name: str,
        pipeline_id: str,
        active_owl_queries: list[Query],
        owl_queries: dict[str, list[str]],
        active_clip_queries: list[Query],
        clip_queries: dict[str, list[str]],
        min_prompts: dict[str, int],
        class_names: list[str],
        fps: int,
        frames_interval: int,
    ) -> dict[str, dict]:
        frames_keys = await s3.list_files(bucket_name, f"frames/{pipeline_id}/")
        detections: dict = {"owlv2": {}, "clip": {}, "nsfw": {}, "captions": {}}

        temp_dir = await self._download_frames_to_disk(
            s3, bucket_name, frames_keys, pipeline_id,
        )
        try:
            if clip_queries:
                await self._clip_pass(
                    temp_dir, frames_keys, pipeline_id,
                    active_clip_queries, clip_queries, detections,
                )

            if owl_queries:
                await self._owl_pass(
                    temp_dir, frames_keys, pipeline_id,
                    active_owl_queries, owl_queries, detections,
                )

            flagged_ids = set(detections["clip"]) | set(detections["owlv2"])
            if settings.CAPTION_ENABLED and flagged_ids:
                flagged_keys = [k for k in frames_keys if Path(k).stem in flagged_ids]
                await self._caption_pass(temp_dir, flagged_keys, pipeline_id, detections)
                await self._caption_analysis_pass(pipeline_id, class_names, detections)

            await self._nsfw_pass(temp_dir, frames_keys, pipeline_id, detections)

            detections["frames"] = self._aggregate_frames(detections, min_prompts)
            detections["ocr_analysis"] = await self._ocr_pass(
                temp_dir, frames_keys, pipeline_id,
                class_names, fps, frames_interval,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.logger.info(
                "[PIPELINE - %s] Временный каталог кадров удалён: %s.",
                pipeline_id, temp_dir,
            )

        return detections

    @staticmethod
    def _count_prompts(matches: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in matches:
            cls = m.get("class")
            if cls:
                counts[cls] = counts.get(cls, 0) + 1
        return counts

    @staticmethod
    def _is_confirmed(
        category: str,
        owl: dict[str, int],
        clip: dict[str, int],
        caption: dict[str, int],
        threshold: int,
    ) -> bool:
        votes = sum([category in owl, category in clip, category in caption])
        enough_prompts = (
            owl.get(category, 0) >= threshold
            or clip.get(category, 0) >= threshold
        )
        return votes >= 2 or enough_prompts

    @staticmethod
    def _aggregate_frames(
        detections: dict,
        min_prompts: dict[str, int],
    ) -> dict[str, dict]:
        frames: dict[str, set[str]] = {}
        owl_counts: dict[str, dict[str, int]] = {}
        clip_counts: dict[str, dict[str, int]] = {}
        caption_counts: dict[str, dict[str, int]] = {}

        for frame_id, matches in detections["owlv2"].items():
            counts = DetectFramesTask._count_prompts(matches)
            frames.setdefault(frame_id, set()).update(counts)
            owl_counts[frame_id] = counts

        for frame_id, matches in detections["clip"].items():
            counts = DetectFramesTask._count_prompts(matches)
            frames.setdefault(frame_id, set()).update(counts)
            clip_counts[frame_id] = counts

        for frame_id, classes in detections.get("caption_classes", {}).items():
            caption_counts[frame_id] = {cls: 1 for cls in classes}
            frames.setdefault(frame_id, set()).update(classes)

        for frame_id in detections["nsfw"]:
            frames.setdefault(frame_id, set()).add("nsfw")

        result = {}
        for fid, all_classes in frames.items():
            owl = owl_counts.get(fid, {})
            clip = clip_counts.get(fid, {})
            caption = caption_counts.get(fid, {})
            confirmed = {
                cat for cat in all_classes
                if cat != "nsfw" and DetectFramesTask._is_confirmed(
                    cat, owl, clip, caption, min_prompts.get(cat, 2),
                )
            }
            result[fid] = {
                "classes": sorted(all_classes),
                "confirmed_classes": sorted(confirmed),
            }
        return result

    async def _perform_async(self, pipeline_id: str) -> None:
        self.logger.info(
            "[PIPELINE - %s][Detection] Начало распознавания кадров.",
            pipeline_id,
        )

        pipeline = await self._get_active_pipeline(pipeline_id)
        if pipeline is None:
            return

        s3 = get_s3_client()
        bucket_name = settings.S3_PIPELINE_BUCKET_NAME

        frames_keys = await self._get_frames_list_from_s3(s3, bucket_name, pipeline_id)
        if frames_keys is None:
            return

        (
            active_owl_queries, owl_queries,
            active_clip_queries, clip_queries,
            min_prompts, class_names,
        ) = await self._get_active_queries(pipeline_id)

        detections = await self._process_frames(
            s3, bucket_name, pipeline_id, active_owl_queries, owl_queries,
            active_clip_queries, clip_queries, min_prompts, class_names,
            pipeline.fps or 1, pipeline.frames_interval or 1,
        )

        detections_key = f"detections/{pipeline_id}.json"
        await s3.upload_file(
            bucket_name,
            detections_key,
            json.dumps(detections, ensure_ascii=False).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )
        self.logger.info(
            "[PIPELINE - %s][Detection] Результаты сохранены в S3.",
            pipeline_id,
        )

        async with get_db_context() as db:
            await PipelineRepository.update_detection_status(
                db, pipeline_id, finished=True,
            )

        self.logger.info(
            "[PIPELINE - %s][Detection] Распознавание кадров завершено.",
            pipeline_id,
        )

# python app/run_dramatiq.py app.tasks.detect_frames_task --queue detect_frames_queue --processes 1 --threads 1
