from pydantic_settings import BaseSettings


class Settings(BaseSettings):  # noqa: D101

    # ##############################
    # Настройки S3
    # ##############################
    S3_ENDPOINT_URL: str
    S3_ACCESS_KEY: str
    S3_SECRET_KEY: str
    S3_VIDEO_BUCKET_NAME: str = "videos"
    S3_PIPELINE_BUCKET_NAME: str = "pipelines"

    # ##############################
    # Настройки RabbitMQ
    # ##############################
    RABBITMQ_DEFAULT_USER: str
    RABBITMQ_DEFAULT_PASS: str
    RABBITMQ_HOST: str
    RABBITMQ_PORT: int
    RABBITMQ_VHOST: str

    @property
    def RABBITMQ_URL(self) -> str:
        return (
            "amqp://"
            f"{self.RABBITMQ_DEFAULT_USER}:{self.RABBITMQ_DEFAULT_PASS}@"
            f"{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}/{self.RABBITMQ_VHOST}"
        )

    # ##############################
    # Настройки подключения к БД
    # ##############################
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str
    DB_DRIVER: str

    @property
    def SQLALCHEMY_DATABASE_URL(self) -> str:
        return (
            f"{self.DB_DRIVER}://{self.DB_USER}:{self.DB_PASSWORD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # ##############################
    # Настройки приложения
    # ##############################
    LOG_LEVEL: str = "INFO"

    # Настройки параллельной обработки. Если запускается на одном устройстве,
    # рекомендуется отключить для экономии ресурсов. Так задачи на GPU не будут
    # конкурировать за ресурсы.
    PARALLEL_PROCESS_VIDEO: bool = True

    # ##############################
    # Настройки транскрибации (Whisper)
    # ##############################
    WHISPER_MODEL: str = "turbo"
    WHISPER_BEAM_SIZE: int = 10
    WHISPER_LANGUAGE: str = "ru"
    WHISPER_NO_SPEECH_THRESHOLD: float = 0.6
    WHISPER_LOGPROB_THRESHOLD: float = -0.7
    WHISPER_COMPRESSION_RATIO_THRESHOLD: float = 2.4
    WHISPER_CONDITION_ON_PREVIOUS_TEXT: bool = False
    WHISPER_TEMPERATURE: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    WHISPER_VOCAL_SEPARATION: bool = False

    # ##############################
    # Настройки обработки изображений
    # ##############################
    FRAME_JPEG_QUALITY: int = 80
    FRAME_UPLOAD_CONCURRENCY: int = 8
    IMAGE_PROCESSING_BATCH_SIZE: int = 16
    OWL_BATCH_SIZE: int = 4
    OWL_PROMPT_CHUNK_SIZE: int = 10
    OWL_MODEL_PATH: str = "google/owlv2-base-patch16-ensemble"
    OWL_THRESHOLD: float = 0.1
    CLIP_MODEL_PATH: str = "google/siglip2-so400m-patch16-naflex"
    NSFW_MODEL_PATH: str = "Falconsai/nsfw_image_detection"
    NUDE_THRESHOLD: float = 0.997
    CAPTION_ENABLED: bool = True
    CAPTION_MODEL_PATH: str = "microsoft/Florence-2-base"
    CAPTION_BATCH_SIZE: int = 16
    OCR_RECOGNITION_BATCH_SIZE: int = 16

    # ##############################
    # Настройки LLM анализа текста
    # ##############################
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-3.1-flash-lite"
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"

    class Config:
        env_file = ".env"


settings = Settings()
