from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.detection_class import DetectionClass


class DetectionClassRepository:
    """Репозиторий для работы с классами детекции."""

    @staticmethod
    async def get_all(db: AsyncSession) -> list[DetectionClass]:
        """Возвращает все классы детекции."""
        result = await db.execute(select(DetectionClass))
        return list(result.scalars().all())

    @staticmethod
    async def get_all_names(db: AsyncSession) -> list[str]:
        """Возвращает список названий всех классов детекции."""
        result = await db.execute(select(DetectionClass.name))
        return list(result.scalars().all())
