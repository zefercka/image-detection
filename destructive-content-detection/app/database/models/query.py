from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.core import BaseModel


class Query(BaseModel):
    """Текстовые запросы для zero-shot детекции."""

    __tablename__ = "queries"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default="true",
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
    model_type: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("detection_models.id"),
        nullable=False,
    )
    threshold: Mapped[float] = mapped_column(
        default=0.0,
        nullable=False,
    )
    query: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
    )
    class_name: Mapped[str | None] = mapped_column(
        String(100),
        ForeignKey("detection_classes.name"),
        nullable=True,
    )
