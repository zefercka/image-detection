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


class DetectionItem(BaseModel):
    startFrame: int
    endFrame: int
    start_time: str
    end_time: str
    time_interval: str
    subclass: str
    confidence: float
    type: str


class SourceInfo(BaseModel):
    frameCount: int
    fps: float
    video_path: str
    video_duration_seconds: float
    processing_time_seconds: float
    processing_efficiency: float
    video_duration_formatted: str | None = None
    processing_time_formatted: str | None = None
    analysis_timestamp: str | None = None


class PipelineResultResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    report_type: str
    detections: list[DetectionItem] = Field(default_factory=list)
    sourceInfo: SourceInfo | None = None
