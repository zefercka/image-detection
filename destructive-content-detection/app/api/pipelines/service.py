import logging
from http import HTTPStatus

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.repositories import PipelineRepository, VideoRepository
from app.tasks.process_video_task import ProcessVideoTask

from .schemas import PipelineResponse, PipelineStatusResponse

logger = logging.getLogger(__name__)


async def start_video_processing(
    db: AsyncSession, video_id: str, frames_interval: int,
) -> PipelineResponse:
    """Запускает фоновую задачу по обработке видео."""

    video = await VideoRepository.get(db, video_id)
    if not video:
        logger.warning("Видео с ID %s не найдено для обработки.", video_id)
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Видео не найдено.",
        )

    pipeline = await PipelineRepository.create(db, video_id, frames_interval)

    ProcessVideoTask.send_with_options(args=(pipeline.id, ), delay=5000)
    return PipelineResponse.model_validate(pipeline)


async def get_pipeline_status(
    db: AsyncSession, video_id: str, pipeline_id: str,
) -> PipelineStatusResponse:
    """Получает статус пайплайна по ID видео и ID пайплайна."""

    pipeline = await PipelineRepository.get_by_video_and_pipeline_id(
        db, video_id, pipeline_id,
    )
    if not pipeline:
        logger.warning(
            "Пайплайн с ID %s для видео с ID %s не найден.", pipeline_id, video_id,
        )
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Пайплайн не найден.",
        )

    return PipelineStatusResponse.model_validate(pipeline)
