from http import HTTPStatus

from fastapi import APIRouter

from app.database.core import AsyncDbSession
from app.dependencies.s3.client import S3ClientDependency

from . import service
from .schemas import PipelineResponse, PipelineResultResponse, PipelineStatusResponse

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


@router.get(
    "/{video_id}/pipelines/{pipeline_id}",
    summary="Получение итогового отчёта обработки видео по ID пайплайна",
    status_code=HTTPStatus.OK,
)
async def get_pipeline_result(  # noqa: D103
    db: AsyncDbSession,
    video_id: str,
    pipeline_id: str,
    s3: S3ClientDependency,
) -> PipelineResultResponse:
    return await service.get_pipeline_result(db, video_id, pipeline_id, s3)
