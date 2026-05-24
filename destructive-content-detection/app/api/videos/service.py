import io
import logging
import uuid
from http import HTTPStatus

from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database.models import Video
from app.database.repositories import (
    FileRegistryRepository,
    PipelineRepository,
    VideoRepository,
)
from app.dependencies.s3.client import S3Client
from app.helpers.helpers import calculate_file_hash
from app.tasks.process_video_task import ProcessVideoTask

logger = logging.getLogger(__name__)


async def upload_video(
    db: AsyncSession,
    s3: S3Client,
    file: UploadFile,
) -> Video:
    """Загружает видео в S3."""

    file_hash = calculate_file_hash(file.file)

    existing = await FileRegistryRepository.get_by_sha256(db, file_hash)
    if existing and existing.upload_count > 0:
        await FileRegistryRepository.increment_upload_count(db, file_hash)

        video = await VideoRepository.create(
            db=db,
            title=file.filename or "Без названия",
            description=None,
            file_registry_hash=file_hash,
        )

        logger.info(
            "Файл с хешем %s уже существует.",
            file_hash,
        )

        return video

    file.file.seek(0, io.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)

    if not existing:
        filename = f"{uuid.uuid4().hex}"

        await FileRegistryRepository.create(
            db=db,
            sha256=file_hash,
            s3_key=filename,
            content_type=file.content_type or "application/octet-stream",
            size=size,
        )
    else:
        filename = existing.s3_key
        await FileRegistryRepository.increment_upload_count(db, file_hash)

    video = await VideoRepository.create(
        db=db,
        title=file.filename or "Без названия",
        description=None,
        file_registry_hash=file_hash,
    )

    try:
        await s3.upload_file(
            bucket_name=settings.S3_VIDEO_BUCKET_NAME,
            file_name=filename,
            file_content=file.file,
            content_type=file.content_type or "application/octet-stream",
        )
        logger.info("Файл %s загружен в S3 и зарегистрирован в БД.", filename)
    except Exception:
        try:
            await s3.delete_file(
                bucket_name=settings.S3_VIDEO_BUCKET_NAME,
                file_name=filename,
            )
        except Exception:
            logger.exception("Не удалось удалить файл %s из S3 при откате.", filename)

        raise

    return video


async def delete_video(db: AsyncSession, video_id: str) -> None:
    """Удаляет видео.

    Помечает видео как удалённое в БД и уменьшает счётчик загрузок в
    реестре файлов.
    """

    video = await VideoRepository.get(db, video_id)
    if not video:
        logger.warning("Видео с ID %s не найдено для удаления.", video_id)
        return

    await VideoRepository.delete(db, video_id)
    logger.info("Видео с ID %s помечено как удалённое.", video_id)

    await FileRegistryRepository.decrement_upload_count(db, video.file_registry_hash)

    file_registry = await FileRegistryRepository.get_by_sha256(
        db, video.file_registry_hash,
    )
    if file_registry and file_registry.upload_count == 0:
        try:
            await S3Client(
                endpoint_url=settings.S3_ENDPOINT_URL,
                access_key=settings.S3_ACCESS_KEY,
                secret_key=settings.S3_SECRET_KEY,
            ).delete_file(
                bucket_name=settings.S3_VIDEO_BUCKET_NAME,
                file_name=file_registry.s3_key,
            )
            logger.info("Файл с ключом %s удалён из S3.", file_registry.s3_key)
        except Exception:
            logger.exception(
                "Не удалось удалить файл с ключом %s из S3.",
                file_registry.s3_key,
            )
