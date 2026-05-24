import asyncio
import json
import logging
from logging import Logger

import dramatiq

import app.dependencies.rabbitmq.broker  # noqa: F401
from app.config import settings
from app.database.core import get_db_context
from app.database.declarations.pipeline import PipelineStatusesEnum
from app.database.repositories import DetectionClassRepository, PipelineRepository
from app.dependencies.s3.client import S3Client, get_s3_client
from app.dependencies.s3.helpers import download_to_file
from app.helpers.text_analysis import LLMTextAnalyzer
from app.helpers.video_preprocessing.audio_transcriber import AudioTranscriber
from app.helpers.video_preprocessing.vocal_separator import VocalSeparator
from app.tasks._loop import get_loop

_MAX_AGE_S = 60 * 60 * 2
_MAX_AGE_MS = _MAX_AGE_S * 1000
_TIME_LIMIT_MS = 60 * 60 * 12 * 1000  # 12 часов
_DETECTION_POLL_S = 30
_CLIP_TOP_K = 3


class TranscribeAudioTask(dramatiq.GenericActor):
    """Задача для транскрибации аудио."""

    class Meta:
        queue_name = "transcribe_audio_queue"
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
            "[PIPELINE - %s][Audio] Начало задачи транскрибации аудио.",
            pipeline_id,
        )

        async with get_db_context() as db:
            pipeline = await PipelineRepository.get(db, pipeline_id)
            if pipeline is None:
                self.logger.warning(
                    "[PIPELINE - %s][Audio] Задача не найдена.",
                    pipeline_id,
                )
                return
            s3 = get_s3_client()
            bucket_name = settings.S3_PIPELINE_BUCKET_NAME

            is_file_exist = await s3.file_exists(
                bucket_name, f"audio/{pipeline_id}.mp3",
            )
            if not is_file_exist:
                self.logger.warning(
                    "[PIPELINE - %s][Audio] Аудио не найдено в S3.",
                    pipeline_id,
                )
                await PipelineRepository.update_status(
                    db,
                    pipeline_id,
                    PipelineStatusesEnum.FAILED,
                )
                return

            detection_classes = await DetectionClassRepository.get_all_names(db)

        # Не параллельный режим: ждём детекцию ДО Whisper
        if not settings.PARALLEL_PROCESS_VIDEO:
            clip_context = await self._wait_and_get_clip_context(
                s3, bucket_name, pipeline_id,
            )

        file_path = await self._download_audio(s3, bucket_name, pipeline_id)
        with VocalSeparator() as separator:
            audio_path = self._separate_vocals(pipeline_id, separator, file_path)
            transcription = self._transcribe_audio(pipeline_id, audio_path)
        AudioTranscriber.unload()

        # Параллельный режим: Whisper уже отработал, теперь ждём детекцию
        if settings.PARALLEL_PROCESS_VIDEO:
            clip_context = await self._wait_and_get_clip_context(s3, bucket_name, pipeline_id)

        analysis = await self._analyze_text(
            pipeline_id, transcription, detection_classes, clip_context,
        )
        await self._upload_transcription(s3, bucket_name, pipeline_id, transcription)
        await self._upload_text_analysis(s3, bucket_name, pipeline_id, analysis)

        async with get_db_context() as db:
            await PipelineRepository.update_transcription_status(
                db, pipeline_id, finished=True,
            )

        self.logger.info(
            "[PIPELINE - %s][Audio] Транскрибация аудио завершена.",
            pipeline_id,
        )

    async def _download_audio(
        self,
        s3: S3Client,
        bucket_name: str,
        pipeline_id: str,
    ) -> str:
        try:
            file_path = await download_to_file(
                s3, bucket_name, f"audio/{pipeline_id}.mp3",
            )
            self.logger.info(
                "[PIPELINE - %s][Audio] Файл скачен на диск из S3.",
                pipeline_id,
            )
        except Exception:
            self.logger.exception(
                "[PIPELINE - %s][Audio] Ошибка при загрузке аудио из S3.",
                pipeline_id,
            )
            raise

        return file_path

    def _separate_vocals(
        self,
        pipeline_id: str,
        separator: VocalSeparator,
        file_path: str,
    ) -> str:
        if not settings.WHISPER_VOCAL_SEPARATION:
            self.logger.info(
                "[PIPELINE - %s][Audio] Сепарация вокала отключена (WHISPER_VOCAL_SEPARATION=false).",
                pipeline_id,
            )
            return file_path
        try:
            vocals_path = separator.separate(file_path)
            self.logger.info(
                "[PIPELINE - %s][Audio] Вокал отделён: %s",
                pipeline_id,
                vocals_path,
            )
            return vocals_path
        except Exception:
            self.logger.warning(
                "[PIPELINE - %s][Audio] Ошибка отделения вокала, используется оригинал.",
                pipeline_id,
                exc_info=True,
            )
            return file_path

    def _transcribe_audio(self, pipeline_id: str, file_path: str) -> dict:
        self.logger.info(
            "[PIPELINE - %s][Audio] Начало транскрибации аудио.",
            pipeline_id,
        )
        result = AudioTranscriber.transcribe(
            settings.WHISPER_MODEL,
            file_path,
            beam_size=settings.WHISPER_BEAM_SIZE,
            temperature=tuple(settings.WHISPER_TEMPERATURE),
            language=settings.WHISPER_LANGUAGE,
            no_speech_threshold=settings.WHISPER_NO_SPEECH_THRESHOLD,
            logprob_threshold=settings.WHISPER_LOGPROB_THRESHOLD,
            compression_ratio_threshold=settings.WHISPER_COMPRESSION_RATIO_THRESHOLD,
            condition_on_previous_text=settings.WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        )
        self.logger.info(
            "[PIPELINE - %s][Audio] Транскрибация аудио завершена.",
            pipeline_id,
        )
        return result

    async def _wait_for_detection(self, pipeline_id: str) -> bool:
        import time  # noqa: PLC0415

        deadline = time.monotonic() + _MAX_AGE_S
        while time.monotonic() < deadline:
            async with get_db_context() as db:
                pipeline = await PipelineRepository.get(db, pipeline_id)
            if pipeline is None or pipeline.status == PipelineStatusesEnum.FAILED:
                return False
            if pipeline.detection_done:
                self.logger.info(
                    "[PIPELINE - %s][Audio] Детекция фреймов завершена, "
                    "получаю CLIP-контекст.",
                    pipeline_id,
                )
                return True
            self.logger.info(
                "[PIPELINE - %s][Audio] Ожидание детекции фреймов, "
                "следующая проверка через %d с.",
                pipeline_id, _DETECTION_POLL_S,
            )
            await asyncio.sleep(_DETECTION_POLL_S)
        self.logger.warning(
            "[PIPELINE - %s][Audio] Превышено время ожидания детекции, "
            "LLM будет запущен без визуального контекста.",
            pipeline_id,
        )
        return False

    @staticmethod
    def _build_clip_context(
        clip_data: dict,
        fps: int,
        captions: dict[str, str] | None = None,
    ) -> dict[float, dict]:
        time_prompts: dict[float, list[tuple[str, float]]] = {}
        time_captions: dict[float, list[str]] = {}

        for frame_id, matches in clip_data.items():
            try:
                frame_idx = int(frame_id.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            time_sec = round(frame_idx / max(fps, 1), 1)
            for m in matches:
                prompt = m.get("prompt", "")
                score = float(m.get("score", 0.0))
                if prompt:
                    time_prompts.setdefault(time_sec, []).append((prompt, score))
            if captions:
                caption = captions.get(frame_id, "")
                if caption:
                    time_captions.setdefault(time_sec, []).append(caption)

        all_times = sorted(set(time_prompts) | set(time_captions))
        return {
            t: {
                "prompts": [
                    p for p, _ in sorted(
                        time_prompts.get(t, []), key=lambda x: x[1], reverse=True,
                    )[:_CLIP_TOP_K]
                ],
                "caption": next(iter(time_captions.get(t, [])), None),
            }
            for t in all_times
        }

    async def _wait_and_get_clip_context(
        self,
        s3: S3Client,
        bucket_name: str,
        pipeline_id: str,
    ) -> dict[float, dict]:
        if not await self._wait_for_detection(pipeline_id):
            return {}
        try:
            raw = await s3.download_file(bucket_name, f"detections/{pipeline_id}.json")
            detections = json.loads(raw.decode("utf-8"))
        except Exception:
            self.logger.warning(
                "[PIPELINE - %s][Audio] Ошибка загрузки результатов детекции, "
                "LLM будет запущен без визуального контекста.",
                pipeline_id,
                exc_info=True,
            )
            return {}
        clip_data = detections.get("clip", {})
        if not clip_data:
            return {}
        async with get_db_context() as db:
            pipeline = await PipelineRepository.get(db, pipeline_id)
        fps = (pipeline.fps or 1) if pipeline else 1
        captions = detections.get("captions") or None
        context = self._build_clip_context(clip_data, fps, captions)
        self.logger.info(
            "[PIPELINE - %s][Audio] CLIP-контекст: %d временных точек, "
            "описания: %d.",
            pipeline_id, len(context),
            sum(1 for v in context.values() if v.get("caption")),
        )
        return context

    async def _analyze_text(
        self,
        pipeline_id: str,
        transcription: dict,
        classes: list[str],
        clip_context: dict[float, dict] | None = None,
    ) -> dict:
        if not settings.GEMINI_API_KEY:
            self.logger.warning(
                "[PIPELINE - %s][Audio] GEMINI_API_KEY не задан, "
                "анализ текста пропущен.",
                pipeline_id,
            )
            return {}

        self.logger.info(
            "[PIPELINE - %s][Audio] Начало LLM-анализа текста (%d классов, "
            "визуальный контекст: %s).",
            pipeline_id,
            len(classes),
            f"{len(clip_context)} точек" if clip_context else "нет",
        )
        analyzer = LLMTextAnalyzer(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            base_url=settings.GEMINI_BASE_URL,
        )
        analysis = await analyzer.analyze(transcription, classes, clip_context=clip_context)
        self.logger.info(
            "[PIPELINE - %s][Audio] LLM-анализ завершён: %s.",
            pipeline_id,
            analysis.get("summary", {}),
        )
        return analysis

    async def _upload_transcription(
        self,
        s3: S3Client,
        bucket_name: str,
        pipeline_id: str,
        result: dict,
    ) -> None:
        transcription_key = f"transcription/{pipeline_id}.json"
        await s3.upload_file(
            bucket_name,
            transcription_key,
            json.dumps(result, ensure_ascii=False).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )
        self.logger.info(
            "[PIPELINE - %s][Audio] Результат транскрибации загружен в S3.",
            pipeline_id,
        )

    async def _upload_text_analysis(
        self,
        s3: S3Client,
        bucket_name: str,
        pipeline_id: str,
        analysis: dict,
    ) -> None:
        analysis_key = f"text_analysis/{pipeline_id}.json"
        await s3.upload_file(
            bucket_name,
            analysis_key,
            json.dumps(analysis, ensure_ascii=False).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )
        self.logger.info(
            "[PIPELINE - %s][Audio] Результат анализа текста загружен в S3.",
            pipeline_id,
        )
