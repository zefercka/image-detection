import asyncio
import logging
import time
from contextlib import suppress
from logging import Logger
from pathlib import Path

import aiofiles
import cv2
import dramatiq

import app.dependencies.rabbitmq.broker  # noqa: F401
from app.config import settings
from app.database.core import get_db_context
from app.database.declarations.pipeline import PipelineStatusesEnum
from app.database.repositories import FileRegistryRepository, PipelineRepository
from app.database.repositories.video import VideoRepository
from app.dependencies.s3.client import S3Client, get_s3_client
from app.dependencies.s3.helpers import download_to_file
from app.helpers.video_preprocessing.audio_extractor import AudioExtractor
from app.tasks._loop import get_loop

_LOG_EVERY_N_FRAMES = 100
_MAX_AGE_S = 60 * 60 * 2   # 2 часа в секундах (для future.result timeout)
_MAX_AGE_MS = _MAX_AGE_S * 1000  # в миллисекундах (для Dramatiq max_age)


class ProcessVideoTask(dramatiq.GenericActor):
    """Задача для обработки видео."""

    class Meta:
        queue_name = "video_process_queue"
        max_retries = 5
        max_age = _MAX_AGE_MS
        time_limit = _MAX_AGE_MS

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

    async def _is_data_exists(
        self,
        s3: S3Client,
        pipeline_id: str,
    ) -> tuple[str, int] | tuple[None, None]:
        async with get_db_context() as db:
            pipeline = await PipelineRepository.get(db, pipeline_id)
            if pipeline is None:
                self.logger.warning("[PIPELINE - %s] Задача не найдена.", pipeline_id)
                return None, None

            video = await VideoRepository.get(db, pipeline.video_id)
            if video is None:
                self.logger.warning("[PIPELINE - %s] Видео не найдено.", pipeline_id)
                return None, None

            s3_file = await FileRegistryRepository.get_by_sha256(
                db, video.file_registry_hash,
            )
            if s3_file is None:
                self.logger.error(
                    "[PIPELINE - %s] Файл не найден в реестре файлов.",
                    pipeline_id,
                )
                return None, None

            s3_key = s3_file.s3_key
            is_file_exists = await s3.file_exists(settings.S3_VIDEO_BUCKET_NAME, s3_key)
            if not is_file_exists:
                self.logger.error(
                    "[PIPELINE - %s] Файл не найден в S3.",
                    pipeline_id,
                )
                return None, None

        return s3_key, pipeline.frames_interval

    async def _perform_async(self, pipeline_id: str) -> None:
        start_time = time.time()
        self.logger.info("[PIPELINE - %s] Начало обработки задачи.", pipeline_id)

        s3 = get_s3_client()

        s3_key, frames_interval = await self._is_data_exists(s3, pipeline_id)

        async with get_db_context() as db:
            if s3_key is None:
                await PipelineRepository.update_status(
                    db, pipeline_id, PipelineStatusesEnum.FAILED,
                )
                return

            await PipelineRepository.update_status(
                db, pipeline_id, PipelineStatusesEnum.IN_PROGRESS,
            )

        pipeline_bucket = settings.S3_PIPELINE_BUCKET_NAME

        await self._process_video(
            s3,
            s3_key,
            pipeline_id,
            pipeline_bucket,
            frames_interval,
        )

        self.logger.info(
            "[PIPELINE - %s] Подготовка данных завершена за %f секунд.",
            pipeline_id,
            round(time.time() - start_time, 2),
        )

        from app.tasks.collect_results_task import CollectResultsTask  # noqa: PLC0415
        from app.tasks.detect_frames_task import DetectFramesTask  # noqa: PLC0415
        from app.tasks.transcribe_audio_task import TranscribeAudioTask  # noqa: PLC0415

        TranscribeAudioTask.send(pipeline_id)
        DetectFramesTask.send(pipeline_id)
        CollectResultsTask.send(pipeline_id)

    async def _process_video(
        self,
        s3: S3Client,
        s3_key: str,
        pipeline_id: str,
        pipeline_bucket: str,
        frame_interval: int,
    ) -> None:
        path_to_video = await download_to_file(
            s3,
            bucket_name=settings.S3_VIDEO_BUCKET_NAME,
            key=s3_key,
            suffix=".mp4",
        )
        path_to_video = Path(path_to_video)

        self.logger.info(
            "[PIPELINE - %s] Видео сохранено: %s",
            pipeline_id,
            path_to_video,
        )

        try:
            await asyncio.gather(
                self._split_video_into_frames(
                    s3,
                    pipeline_id,
                    pipeline_bucket,
                    path_to_video,
                    frame_interval,
                ),
                self._extract_audio(
                    s3,
                    pipeline_id,
                    pipeline_bucket,
                    path_to_video,
                ),
            )
        finally:
            with suppress(FileNotFoundError):
                path_to_video.unlink()
                self.logger.info(
                    "[PIPELINE - %s] Удалён временный файл: %s",
                    pipeline_id,
                    path_to_video,
                )

    async def _split_video_into_frames(
        self,
        s3: S3Client,
        pipeline_id: str,
        pipeline_bucket: str,
        path_to_video: Path,
        frame_interval: int = 1,
    ) -> None:

        # Читаем фреймы итеративно через очередь, чтобы не загружать всё видео
        # в память сразу. Декодирование идёт в потоке executor'а, загрузка в S3 -
        # в event loop. Буфер очереди ограничивает пиковое потребление памяти.
        queue = asyncio.Queue(maxsize=32)
        loop = asyncio.get_running_loop()
        fps: float | int = 0.0

        async def _producer() -> None:
            """Декодирует фреймы в executor и кладёт нужные в очередь."""
            nonlocal fps

            def _decode_frames() -> None:
                nonlocal fps

                cap = cv2.VideoCapture(str(path_to_video))

                if not cap.isOpened():
                    msg = f"Cannot open {path_to_video}"
                    raise ValueError(msg)

                fps = cap.get(cv2.CAP_PROP_FPS)

                try:
                    idx = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if idx % frame_interval == 0:
                            ok, buffer = cv2.imencode(
                                ".jpg", frame,
                                [cv2.IMWRITE_JPEG_QUALITY, settings.FRAME_JPEG_QUALITY],
                            )
                            if not ok:
                                self.logger.warning(
                                    "Не удалось закодировать фрейм %d, пропускаем.", idx,
                                )
                            else:
                                asyncio.run_coroutine_threadsafe(
                                    queue.put((idx, buffer.tobytes())),
                                    loop,
                                ).result()
                        idx += 1
                finally:
                    cap.release()

            await loop.run_in_executor(None, _decode_frames)
            await queue.put(None)

        async def _consumer() -> None:
            """Загружает фреймы из очереди в S3 параллельно."""
            sem = asyncio.Semaphore(settings.FRAME_UPLOAD_CONCURRENCY)
            pending: set[asyncio.Task] = set()
            uploaded = 0

            async def _upload(frame_index: int, data: bytes) -> None:
                async with sem:
                    await s3.upload_file(
                        bucket_name=pipeline_bucket,
                        file_name=f"frames/{pipeline_id}/frame_{frame_index}.jpg",
                        file_content=data,
                    )

            while True:
                item = await queue.get()
                if item is None:
                    break
                frame_index, data = item
                task = asyncio.create_task(_upload(frame_index, data))
                pending.add(task)
                task.add_done_callback(pending.discard)
                uploaded += 1
                if uploaded % _LOG_EVERY_N_FRAMES == 0:
                    self.logger.info(
                        "Загружено %d фреймов в бакет %s.", uploaded, pipeline_bucket,
                    )

            if pending:
                await asyncio.gather(*pending)

        await asyncio.gather(_producer(), _consumer())
        await self._save_fps_to_db(pipeline_id, fps)

    async def _save_fps_to_db(self, pipeline_id: str, fps: float) -> None:
        async with get_db_context() as db:
            await PipelineRepository.update(db, pipeline_id, fps=int(fps))

    async def _extract_audio(
        self,
        s3: S3Client,
        pipeline_id: str,
        pipeline_bucket: str,
        path_to_video: Path,
    ) -> None:
        loop = asyncio.get_running_loop()

        try:
            with AudioExtractor() as extractor:
                path_to_audio = await loop.run_in_executor(
                    None, extractor.extract_audio, path_to_video,
                )

                async with aiofiles.open(path_to_audio, "rb") as f:
                    audio_bytes = await f.read()

                await s3.upload_file(
                    bucket_name=pipeline_bucket,
                    file_name=f"audio/{pipeline_id}.mp3",
                    file_content=audio_bytes,
                    content_type="audio/mpeg",
                )

                self.logger.info(
                    "[PIPELINE - %s] Аудио загружено в S3.",
                    pipeline_id,
                )
        except Exception:
            self.logger.exception(
                "[PIPELINE - %s] Ошибка при извлечении/загрузке аудио.",
                pipeline_id,
            )
            raise
