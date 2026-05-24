from http import HTTPStatus

from fastapi import APIRouter

from app.database.core import AsyncDbSession

from . import service
from .schemas import PipelineResponse, PipelineStatusResponse

router = APIRouter(prefix="/videos", tags=["pipelines"])


@router.post(
    "/{video_id}/pipelines",
    summary="Запуск анализа видео по ID",
    status_code=HTTPStatus.OK,
)
async def analyze_video(  # noqa: D103
    db: AsyncDbSession,
    video_id: str,
    frames_interval: int = 30,
) -> PipelineResponse:
    return await service.start_video_processing(db, video_id, frames_interval)

@router.get(
    "/{video_id}/pipelines/{pipeline_id}/status",
    summary="Получение статуса пайплайна по ID видео и ID пайплайна",
    status_code=HTTPStatus.OK,
)
async def get_pipeline_status(  # noqa: D103
    db: AsyncDbSession,
    video_id: str,
    pipeline_id: str,
) -> PipelineStatusResponse:
    return await service.get_pipeline_status(db, video_id, pipeline_id)
