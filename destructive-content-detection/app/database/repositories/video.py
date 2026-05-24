from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Video


class VideoRepository:
    """Репозиторий для работы с моделью Video."""

    @staticmethod
    async def get(
        db: AsyncSession,
        id_: str,
        show_deleted: bool = False,
    ) -> Video | None:
        """Получает запись по ID видео."""

        stmt = (
            select(Video)
            .where(Video.id == id_)
        )
        if not show_deleted:
            stmt = stmt.where(Video.deleted_at.is_(None))

        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def create(
        db: AsyncSession,
        title: str,
        description: str | None,
        file_registry_hash: str,
    ) -> Video:
        """Создаёт новую запись в таблице видео."""

        record = Video(
            title=title,
            description=description,
            file_registry_hash=file_registry_hash,
        )
        db.add(record)
        await db.flush()

        return record

    @staticmethod
    async def delete(db: AsyncSession, id_: str) -> None:
        """Удаляет видео, устанавливая deleted_at."""

        stmt = (
            update(Video)
            .where(Video.id == id_)
            .values(deleted_at=Video.updated_at)
        )
        await db.execute(stmt)
