from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.core import BaseModel


class DetectionModel(BaseModel):
    """Модели детекции, используемые в пайплайне."""

    __tablename__ = "detection_models"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )
