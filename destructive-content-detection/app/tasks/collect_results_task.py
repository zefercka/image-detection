import asyncio
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from logging import Logger
from pathlib import Path

import dramatiq

import app.dependencies.rabbitmq.broker  # noqa: F401
from app.config import settings
from app.database.core import get_db_context
from app.database.declarations.pipeline import PipelineStatusesEnum
from app.database.models import Pipeline
from app.database.repositories import (
    FileRegistryRepository,
    PipelineRepository,
    VideoRepository,
)
from app.dependencies.s3.client import S3Client, get_s3_client
from app.tasks._loop import get_loop

_MAX_AGE_S = 60 * 60 * 9   # 9 часов
_MAX_AGE_MS = _MAX_AGE_S * 1000
_TIME_LIMIT_MS = 60 * 60 * 24 * 1000  # 24 часа
_POLL_INTERVAL_S = 30
_AUDIO_CONFIDENCE = 0.98


class CollectResultsTask(dramatiq.GenericActor):
    """Задача для сборки итогового отчёта детекции."""

    class Meta:
        queue_name = "collect_results_queue"
        max_retries = 5
        max_age = _MAX_AGE_MS
        time_limit = _TIME_LIMIT_MS

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

    async def _perform_async(self, pipeline_id: str) -> None:
        self.logger.info(
            "[PIPELINE - %s][Report] Ожидание завершения детекции и транскрибации.",
            pipeline_id,
        )

        pipeline = await self._wait_for_tasks(pipeline_id)
        if pipeline is None:
            return

        self.logger.info(
            "[PIPELINE - %s][Report] Начало сборки итогового отчёта.",
            pipeline_id,
        )

        s3 = get_s3_client()
        bucket_name = settings.S3_PIPELINE_BUCKET_NAME

        detections = await self._load_json(
            s3, bucket_name, f"detections/{pipeline_id}.json",
        )
        text_analysis = await self._load_json(
            s3, bucket_name, f"text_analysis/{pipeline_id}.json",
        )

        if detections is None:
            self.logger.error(
                "[PIPELINE - %s][Report] Файл детекций не найден в S3.",
                pipeline_id,
            )
            return

        frame_keys = await s3.list_files(bucket_name, f"frames/{pipeline_id}/")
        frame_count = self._estimate_frame_count(frame_keys, pipeline.frames_interval)
        video_path = await self._get_video_path(pipeline_id, s3)

        report = self._build_report(
            pipeline, detections, text_analysis or {}, frame_count, video_path,
        )

        await s3.upload_file(
            bucket_name,
            f"reports/{pipeline_id}.json",
            json.dumps(report, ensure_ascii=False).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )

        async with get_db_context() as db:
            await PipelineRepository.update_status(
                db, pipeline_id, PipelineStatusesEnum.COMPLETED,
            )

        self.logger.info(
            "[PIPELINE - %s][Report] Итоговый отчёт сохранён в S3.",
            pipeline_id,
        )

    async def _wait_for_tasks(self, pipeline_id: str) -> "Pipeline | None":
        """Опрашивает БД каждые _POLL_INTERVAL_S секунд до завершения обеих задач."""
        import time  # noqa: PLC0415

        deadline = time.monotonic() + _MAX_AGE_S
        while time.monotonic() < deadline:
            async with get_db_context() as db:
                pipeline = await PipelineRepository.get(db, pipeline_id)

            if pipeline is None:
                self.logger.warning(
                    "[PIPELINE - %s][Report] Задача не найдена.",
                    pipeline_id,
                )
                return None

            if pipeline.status == PipelineStatusesEnum.FAILED:
                self.logger.warning(
                    "[PIPELINE - %s][Report] Пайплайн завершился с ошибкой, пропуск.",
                    pipeline_id,
                )
                return None

            if pipeline.detection_done and pipeline.transcription_done:
                return pipeline

            self.logger.info(
                "[PIPELINE - %s][Report] detection_done=%s, transcription_done=%s — "
                "следующая проверка через %d с.",
                pipeline_id,
                pipeline.detection_done,
                pipeline.transcription_done,
                _POLL_INTERVAL_S,
            )
            await asyncio.sleep(_POLL_INTERVAL_S)

        self.logger.error(
            "[PIPELINE - %s][Report] Превышено время ожидания (%d с).",
            pipeline_id,
            _MAX_AGE_S,
        )
        return None

    async def _load_json(
        self,
        s3: S3Client,
        bucket_name: str,
        key: str,
    ) -> dict | None:
        try:
            raw = await s3.download_file(bucket_name, key)
            return json.loads(raw.decode("utf-8"))
        except Exception:
            self.logger.warning("[Report] Не удалось загрузить %s из S3.", key)
            return None

    @staticmethod
    def _estimate_frame_count(frame_keys: list[str], frames_interval: int) -> int:
        """Оценивает общее число кадров по именам файлов в S3."""
        max_idx = 0
        for key in frame_keys:
            stem = Path(key).stem  # "frame_173"
            try:
                idx = int(stem.split("_", 1)[1])
                if idx > max_idx:
                    max_idx = idx
            except (ValueError, IndexError):
                pass
        return max_idx + max(frames_interval, 1)

    async def _get_video_path(self, pipeline_id: str, s3: S3Client) -> str:
        try:
            async with get_db_context() as db:
                pipeline = await PipelineRepository.get(db, pipeline_id)
                if pipeline is None:
                    return ""
                video = await VideoRepository.get(db, pipeline.video_id)
                if video is None:
                    return ""
                file_reg = await FileRegistryRepository.get_by_sha256(
                    db, video.file_registry_hash,
                )
                if file_reg is None:
                    return ""
            return await s3.get_presigned_url(
                settings.S3_VIDEO_BUCKET_NAME, file_reg.s3_key,
            )
        except Exception:
            self.logger.warning(
                "[PIPELINE - %s][Report] Не удалось получить ссылку на видеофайл.",
                pipeline_id, exc_info=True,
            )
            return ""

    def _build_report(
        self,
        pipeline: Pipeline,
        detections: dict,
        text_analysis: dict,
        frame_count: int,
        video_path: str = "",
    ) -> dict:
        fps = pipeline.fps or 1
        frames_data = detections.get("frames", {})

        video_duration_s = frame_count / fps
        finished_at = datetime.now(UTC)
        processing_s = (finished_at - pipeline.created_at).total_seconds()
        efficiency = (
            round(processing_s / video_duration_s, 2) if video_duration_s > 0 else 0.0
        )

        video_detections = self._collect_video_detections(
            frames_data, detections, fps, pipeline.frames_interval,
        )
        audio_detections = self._collect_timeline_detections(
            text_analysis, fps, "audio",
        )
        ocr_detections = self._collect_timeline_detections(
            detections.get("ocr_analysis", {}), fps, "video",
        )

        all_detections = sorted(
            video_detections + audio_detections + ocr_detections,
            key=lambda d: (d["startFrame"], d["type"]),
        )

        return {
            "report_type": "TIME_BASED_REPORT",
            "detections": all_detections,
            "sourceInfo": {
                "frameCount": frame_count,
                "fps": float(fps),
                "video_path": video_path,
                "video_duration_seconds": round(video_duration_s, 1),
                "processing_time_seconds": round(processing_s, 2),
                "processing_efficiency": efficiency,
                "video_duration_formatted": self._format_duration(video_duration_s),
                "processing_time_formatted": self._format_duration(processing_s),
                "analysis_timestamp": finished_at.isoformat(),
            },
        }

    def _collect_video_detections(
        self,
        frames_data: dict,
        detections: dict,
        fps: int,
        frames_interval: int,
    ) -> list[dict]:
        class_frames: dict[str, list[tuple[int, float]]] = defaultdict(list)

        for frame_id, frame_info in frames_data.items():
            confirmed = [
                cls for cls in frame_info.get("confirmed_classes", [])
                if cls != "safe"
            ]
            if not confirmed:
                continue
            try:
                frame_idx = int(frame_id.split("_", 1)[1])
            except (ValueError, IndexError):
                continue

            for cls in confirmed:
                confidence = self._get_frame_confidence(frame_id, cls, detections)
                class_frames[cls].append((frame_idx, confidence))

        step = max(frames_interval, 1)
        result = []
        for cls, frames in class_frames.items():
            frames.sort(key=lambda x: x[0])
            for start_f, end_f, conf in self._group_into_intervals(frames, fps):
                result.append(
                    self._make_detection(
                        max(start_f - step, 0), end_f + step,
                        cls, conf, fps, "video",
                    ),
                )

        return result

    def _collect_timeline_detections(
        self,
        analysis: dict,
        fps: int,
        detection_type: str,
    ) -> list[dict]:
        result = []
        for entry in analysis.get("timeline", []):
            start_s = float(entry.get("start", 0))
            end_s = float(entry.get("end", start_s))
            start_frame = round(start_s * fps)
            end_frame = round(end_s * fps)
            for cls in entry.get("classes", {}):
                if cls == "safe":
                    continue
                result.append(
                    self._make_detection(
                        start_frame, end_frame, cls,
                        _AUDIO_CONFIDENCE, fps, detection_type,
                    ),
                )
        return result

    @staticmethod
    def _get_frame_confidence(
        frame_id: str,
        class_name: str,
        detections: dict,
    ) -> float:
        max_score = 0.0
        for det in detections.get("owlv2", {}).get(frame_id, []):
            if det.get("class") == class_name:
                max_score = max(max_score, float(det.get("score", 0)))
        for det in detections.get("clip", {}).get(frame_id, []):
            if det.get("class") == class_name:
                max_score = max(max_score, float(det.get("score", 0)))
        return max_score

    @staticmethod
    def _group_into_intervals(
        frames: list[tuple[int, float]],
        fps: int,
    ) -> list[tuple[int, int, float]]:
        """Группирует кадры одного класса в интервалы по временной близости."""
        if not frames:
            return []

        max_gap = max(fps, 1)  # максимальный разрыв — 1 секунда

        intervals: list[tuple[int, int, float]] = []
        start_idx, start_conf = frames[0]
        end_idx = start_idx
        max_conf = start_conf

        for frame_idx, confidence in frames[1:]:
            if frame_idx - end_idx <= max_gap:
                end_idx = frame_idx
                max_conf = max(max_conf, confidence)
            else:
                intervals.append((start_idx, end_idx, round(max_conf, 3)))
                start_idx = frame_idx
                end_idx = frame_idx
                max_conf = confidence

        intervals.append((start_idx, end_idx, round(max_conf, 3)))
        return intervals

    @staticmethod
    def _format_time(seconds: float) -> str:
        total = int(seconds)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        from datetime import timedelta  # noqa: PLC0415
        return str(timedelta(seconds=int(seconds)))

    def _make_detection(
        self,
        start_frame: int,
        end_frame: int,
        subclass: str,
        confidence: float,
        fps: int,
        detection_type: str,
    ) -> dict:
        start_time = self._format_time(start_frame / fps)
        end_time = self._format_time(end_frame / fps)
        return {
            "startFrame": start_frame,
            "endFrame": end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "time_interval": f"{start_time} - {end_time}",
            "subclass": subclass,
            "confidence": round(confidence, 3),
            "type": detection_type,
        }
