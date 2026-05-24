
from datetime import UTC, datetime

from sqlalchemy import BigInteger, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.core import BaseModel


class FileRegistry(BaseModel):
    """Модель для регистрации файлов."""

    __tablename__ = "file_registry"

    sha256: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )
    s3_key: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )
    content_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )
    size: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    upload_count: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )
