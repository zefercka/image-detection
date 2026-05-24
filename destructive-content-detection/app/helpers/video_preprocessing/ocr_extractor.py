import logging
from difflib import SequenceMatcher

import easyocr
import numpy as np

logger = logging.getLogger(__name__)

_SIMILARITY_THRESHOLD = 0.85
_LATIN_TO_CYR = str.maketrans(
    "aeopcxylmkivusnrABEHKMOPCTXYLIVUSNR",
    "аеорсхулмкивиснрАВЕНКМОРСТХУЛИВИСНР",
)


class OcrExtractor:
    """Извлекает текст с кадров видео с помощью EasyOCR."""

    _model: easyocr.Reader | None = None

    @classmethod
    def load(cls) -> None:
        if cls._model is None:
            logger.info("Загрузка EasyOCR модели...")
            cls._model = easyocr.Reader(["ru", "en"], gpu=True)

    @classmethod
    def unload(cls) -> None:
        cls._model = None

    @classmethod
    def extract_frames(cls, frames: list[np.ndarray]) -> list[str]:
        """Возвращает список распознанных текстов для каждого кадра."""
        if cls._model is None:
            msg = "OcrExtractor не загружен. Вызовите load() перед использованием."
            raise RuntimeError(msg)
        results = []
        for frame in frames:
            pred = cls._model.readtext(frame)
            # EasyOCR format: [(bbox, text, confidence), ...]
            text = " ".join(item[1] for item in pred if item[1]).strip()
            results.append(text)
        return results

    @staticmethod
    def deduplicate(
        frame_texts: list[tuple[int, str]],
        threshold: float = _SIMILARITY_THRESHOLD,
    ) -> list[tuple[int, str]]:
        """Убирает кадры с текстом, слишком похожим на предыдущий."""
        result: list[tuple[int, str]] = []
        prev_norm = ""
        for frame_idx, text in frame_texts:
            if not text:
                continue
            norm = text.translate(_LATIN_TO_CYR).lower()
            if SequenceMatcher(None, prev_norm, norm).ratio() < threshold:
                result.append((frame_idx, text))
                prev_norm = norm
        return result
