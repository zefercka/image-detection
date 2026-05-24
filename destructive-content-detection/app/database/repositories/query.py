from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.query import Query


class QueryRepository:
    """Репозиторий для управления запросами OWLv2."""

    @staticmethod
    async def get_active(db: AsyncSession, model_id: int | None = None) -> list[Query]:
        """Возвращает активные запросы с маппингом на класс."""
        stmt = select(Query).where(Query.is_active.is_(True))
        if model_id is not None:
            stmt = stmt.where(Query.model_type == model_id)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_all(db: AsyncSession) -> list[Query]:
        """Возвращает все запросы."""
        result = await db.execute(select(Query))
        return list(result.scalars().all())

    @staticmethod
    async def create(db: AsyncSession, query: str) -> Query:
        """Создаёт новый запрос."""
        record = Query(query=query)
        db.add(record)
        await db.flush()
        return record

    @staticmethod
    async def set_active(db: AsyncSession, query: str, is_active: bool) -> None:
        """Включает или отключает запрос."""
        stmt = update(Query).where(Query.query == query).values(is_active=is_active)
        await db.execute(stmt)

    @staticmethod
    async def delete(db: AsyncSession, query: str) -> None:
        """Удаляет запрос."""
        stmt = delete(Query).where(Query.query == query)
        await db.execute(stmt)
