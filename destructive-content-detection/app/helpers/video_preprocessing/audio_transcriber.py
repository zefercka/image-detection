import gc
import logging

import torch
import whisper

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Класс для транскрибирования аудио с помощью модели Whisper."""

    _model: whisper.Whisper | None = None

    @classmethod
    def get_model(cls, model_name: str) -> whisper.Whisper:
        """Возвращает модель Whisper, загружая её при первом вызове."""
        if cls._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(
                "Загрузка Whisper модели '%s' на устройство: %s",
                model_name,
                device,
            )
            cls._model = whisper.load_model(model_name, device=device)
        return cls._model

    @classmethod
    def transcribe(
        cls,
        model: str,
        audio_input: str,
        beam_size: int = 5,
        temperature: tuple[float, ...] = (0.4, 0.6),
        language: str = "ru",
        no_speech_threshold: float = 0.8,
        logprob_threshold: float = -1.0,
        compression_ratio_threshold: float = 2.0,
        condition_on_previous_text: bool = False,
    ) -> dict:
        """Транскрибирует аудио с помощью модели Whisper."""
        return cls.get_model(model).transcribe(
            audio_input,
            beam_size=beam_size,
            temperature=temperature,
            language=language,
            no_speech_threshold=no_speech_threshold,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            condition_on_previous_text=condition_on_previous_text,
        )

    @classmethod
    def unload(cls) -> None:
        cls._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
