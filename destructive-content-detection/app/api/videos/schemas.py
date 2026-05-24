from datetime import datetime

from pydantic import BaseModel, ConfigDict


class VideoCreateResponse(BaseModel):
    """Ответ при успешной загрузке видео."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    title: str
    description: str | None
    created_at: datetime
    updated_at: datetime
