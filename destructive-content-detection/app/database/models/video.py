from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.core import BaseModel
from app.helpers.helpers import generate_flake_id


class Video(BaseModel):
    """Модель для хранения информации о видео."""

    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(
        String(20),
        primary_key=True,
        default=generate_flake_id,
        nullable=False,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    description: Mapped[str] = mapped_column(
        String(1024),
        nullable=True,
    )
    file_registry_hash: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("file_registry.sha256"),
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
    deleted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
