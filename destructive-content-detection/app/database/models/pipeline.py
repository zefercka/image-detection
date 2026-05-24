from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.core import BaseModel
from app.helpers.helpers import generate_flake_id


class PipelineStatuses(BaseModel):
    """Справочник статусов конвейера."""

    __tablename__ = "pipeline_statuses"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
    )
    description: Mapped[str] = mapped_column(
        String(255),
        nullable=True,
    )


class Pipeline(BaseModel):
    """Модель для хранения информации о конвейере обработки видео."""

    __tablename__ = "pipelines"

    id: Mapped[str] = mapped_column(
        String(20),
        primary_key=True,
        default=generate_flake_id,
        nullable=False,
    )
    video_id: Mapped[str] = mapped_column(
        String(20),
        ForeignKey("videos.id"),
        nullable=False,
    )
    frames_interval: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    status: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(f"{PipelineStatuses.__tablename__}.id"),
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
    detection_done: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        server_default="false",
    )
    transcription_done: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        server_default="false",
    )
    fps: Mapped[int] = mapped_column(
        Integer,
        nullable=True,
    )
    description: Mapped[str] = mapped_column(
        String(1023),
        nullable=True,
    )

    status_obj: Mapped["PipelineStatuses"] = relationship(
        "PipelineStatuses", lazy="joined", foreign_keys=[status],
    )
