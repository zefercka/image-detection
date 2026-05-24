from http import HTTPStatus

from fastapi import APIRouter, HTTPException, UploadFile

from app.database.core import AsyncDbSession
from app.dependencies.s3.client import S3ClientDependency

from . import service
from .schemas import VideoCreateResponse

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post(
    "",
    summary="Загрузка видео",
    status_code=HTTPStatus.CREATED,
)
async def upload_video(  # noqa: D103
    db: AsyncDbSession,
    s3: S3ClientDependency,
    file: UploadFile,
) -> VideoCreateResponse:
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail="Разрешены только видеофайлы.",
        )
    video = await service.upload_video(db, s3, file)

    return VideoCreateResponse(
        id=video.id,
        title=video.title,
        description=video.description,
        created_at=video.created_at,
        updated_at=video.updated_at,
    )

@router.delete(
    "/{video_id}",
    summary="Удаление видео по ID",
    status_code=HTTPStatus.OK,
)
async def delete_video(  # noqa: D103
    db: AsyncDbSession,
    video_id: str,
) -> dict:
    await service.delete_video(db, video_id)

    return {"detail": "Видео успешно удалено."}
