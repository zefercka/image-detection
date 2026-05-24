from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.file_registry import FileRegistry


class FileRegistryRepository:
    """Репозиторий для работы с моделью FileRegistry."""

    @staticmethod
    async def get_by_sha256(db: AsyncSession, sha256: str) -> FileRegistry | None:
        """Получает запись по SHA256 хэшу."""

        stmt = select(FileRegistry).where(FileRegistry.sha256 == sha256)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def create(
        db: AsyncSession,
        sha256: str,
        s3_key: str,
        size: int,
        content_type: str = "application/octet-stream",
    ) -> FileRegistry:
        """Создаёт новую запись в реестре файлов."""

        record = FileRegistry(
            sha256=sha256,
            s3_key=s3_key,
            content_type=content_type,
            upload_count=1,
            size=size,
        )
        db.add(record)
        await db.flush()

        return record

    @staticmethod
    async def increment_upload_count(db: AsyncSession, sha256: str) -> None:
        """Атомарно увеличивает upload_count на 1."""

        stmt = (
            update(FileRegistry)
            .where(FileRegistry.sha256 == sha256)
            .values(upload_count=FileRegistry.upload_count + 1)
        )
        await db.execute(stmt)

    @staticmethod
    async def decrement_upload_count(db: AsyncSession, sha256: str) -> None:
        """Атомарно уменьшает upload_count на 1."""

        stmt = (
            update(FileRegistry)
            .where(FileRegistry.sha256 == sha256)
            .values(upload_count=FileRegistry.upload_count - 1)
        )
        await db.execute(stmt)
