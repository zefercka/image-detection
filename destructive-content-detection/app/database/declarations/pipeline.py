from enum import IntEnum


class PipelineStatusesEnum(IntEnum):
    """Перечисление для статусов пайплайна."""

    PENDING = 0
    IN_PROGRESS = 1
    COMPLETED = 2
    FAILED = 3


PIPELINE_STATUSES_DESCRIPTION = {
    PipelineStatusesEnum.PENDING: "В очереди на обработку.",
    PipelineStatusesEnum.IN_PROGRESS: "В процессе обработки.",
    PipelineStatusesEnum.COMPLETED: "Обработка завершена.",
    PipelineStatusesEnum.FAILED: "Обработка завершилась с ошибкой.",
}
