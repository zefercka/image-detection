from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.declarations.pipeline import PipelineStatusesEnum
from app.database.models import Pipeline


class PipelineRepository:
    """Репозиторий для работы с пайплайнами."""

    @staticmethod
    async def get(db: AsyncSession, id_: str) -> Pipeline | None:
        """Получает запись по ID пайплайна."""

        stmt = select(Pipeline).where(Pipeline.id == id_)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_video_and_pipeline_id(
        db: AsyncSession, video_id: str, pipeline_id: str,
    ) -> Pipeline | None:
        """Получает запись по ID видео и ID пайплайна."""

        stmt = (
            select(Pipeline)
            .where(Pipeline.id == pipeline_id)
            .where(Pipeline.video_id == video_id)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def create(
        db: AsyncSession,
        video_id: str,
        frames_interval: int,
        description: str | None = None,
    ) -> Pipeline:
        """Создаёт новую запись в таблице пайплайнов."""

        record = Pipeline(
            video_id=video_id,
            frames_interval=frames_interval,
            description=description,
            status=PipelineStatusesEnum.PENDING,
        )
        db.add(record)
        await db.flush()

        return record

    @staticmethod
    async def update(
        db: AsyncSession,
        id_: str,
        **kwargs: dict,
    ) -> None:
        """Обновляет запись в таблице пайплайнов."""

        stmt = (
            update(Pipeline)
            .where(Pipeline.id == id_)
            .values(**kwargs)
        )
        await db.execute(stmt)

    @staticmethod
    async def update_status(
        db: AsyncSession,
        id_: str,
        status: PipelineStatusesEnum,
    ) -> None:
        """Обновляет статус пайплайна."""

        stmt = (
            update(Pipeline)
            .where(Pipeline.id == id_)
            .values(status=status)
        )
        await db.execute(stmt)

    @staticmethod
    async def update_transcription_status(
        db: AsyncSession,
        id_: str,
        finished: bool,
    ) -> None:
        """Обновляет статус транскрибации пайплайна."""

        stmt = (
            update(Pipeline)
            .where(Pipeline.id == id_)
            .values(transcription_done=finished)
        )
        await db.execute(stmt)

    @staticmethod
    async def update_detection_status(
        db: AsyncSession,
        id_: str,
        finished: bool,
    ) -> None:
        """Обновляет статус детекции пайплайна."""

        stmt = (
            update(Pipeline)
            .where(Pipeline.id == id_)
            .values(detection_done=finished)
        )
        await db.execute(stmt)
