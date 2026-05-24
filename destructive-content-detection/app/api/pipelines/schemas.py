from pydantic import BaseModel, ConfigDict, Field


class PipelineStatus(BaseModel):
    """Схема для статуса пайплайна."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None = None


class PipelineResponse(BaseModel):
    """Схема ответа при получении или запуске пайплайна."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    pipeline_id: str = Field(..., alias="id")
    video_id: str
    frames_interval: int = 30
    description: str | None = None


class PipelineStatusResponse(BaseModel):
    """Схема ответа при получении статуса пайплайна."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    pipeline_id: str = Field(..., alias="id")
    video_id: str
    status: PipelineStatus = Field(..., alias="status_obj")
