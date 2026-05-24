from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.core import BaseModel


class DetectionClass(BaseModel):
    """Справочник классов детекции — к ним маппятся запросы OWLv2."""

    __tablename__ = "detection_classes"

    name: Mapped[str] = mapped_column(
        String(100),
        primary_key=True,
        nullable=False,
    )
    min_prompts_confirm: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=2,
        server_default="2",
    )
