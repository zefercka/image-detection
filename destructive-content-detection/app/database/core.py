from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class BaseModel(DeclarativeBase):  # noqa: D101
    def dict(self) -> dict:  # noqa: D102
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


engine = create_async_engine(settings.SQLALCHEMY_DATABASE_URL)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession]:
    """Возвращает асинхронную сессию для работы с базой данных.

    При ошибке выполняет откат транзакции и закрывает сессию.

    Yields:
        AsyncGenerator[AsyncSession]: Асинхронная сессия для работы с базой данных.

    """

    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession]:
    """Контекстный менеджер для работы с базой данных.

    При ошибке выполняет откат транзакции и закрывает сессию.

    Yields:
        AsyncGenerator[AsyncSession, None]: Асинхронная сессия для работы
        с базой данных.

    """

    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


AsyncDbSession = Annotated[AsyncSession, Depends(get_db)]
