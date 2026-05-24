import asyncio
import json
import logging

import httpx

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_BASE = 2.0  # seconds

_SYSTEM_PROMPT = (
    "Ты — система автоматической модерации видеоконтента. "
    "Анализируй транскрибацию аудиодорожки и находи фрагменты "
    "с потенциально опасным или запрещённым содержимым. "
    "Транскрибация — это связный диалог или монолог: "
    "более поздние фрагменты могут ссылаться на то, что обсуждалось раньше. "
    "Учитывай весь предшествующий контекст при классификации каждого фрагмента."
)

_USER_PROMPT_CAPTIONS = """\
Для каждого кадра определи, к каким категориям относится описание изображения.

Категории:
{classes}

Описания кадров (формат: [frame_id] описание):
{captions}

Верни JSON, где каждый ключ — frame_id, значение — массив подходящих категорий из списка выше.
Если категорий нет — пустой массив [].
Включай только очевидные совпадения, не домысливай.
"""

_USER_PROMPT = """\
Проанализируй транскрибацию и найди фрагменты, относящиеся к следующим категориям:
{classes}

Транскрибация (формат: [start_сек - end_сек] текст):
{segments}
{clip_section}
Правила классификации:
1. Читай весь текст как единый диалог перед тем как выносить решение по каждому фрагменту.
2. Если тема установлена раньше по контексту — последующие фрагменты о том же предмете \
(эффекты, ощущения, последствия, действие вещества и т.п.) тоже относятся к этой категории, \
даже если ключевое слово в них не повторяется.
3. Разрешай анафорические ссылки: местоимения и слова «эффект», «действие», «оно», «это», \
«после него», «от этого» — определяй, к чему они отсылают по контексту.
4. Основание для классификации — слова в аудио (включая косвенно выраженный смысл). \
Визуальный контекст — вспомогательная подсказка, но не самостоятельное основание.

Верни JSON, где каждый ключ — название категории из списка выше, \
значение — массив найденных совпадений.
Формат каждого совпадения: {{"start": <float>, "end": <float>, "quote": "<точная цитата>"}}
Если совпадений нет — пустой массив [].
"""


class LLMTextAnalyzer:
    """Анализирует транскрибацию через LLM (OpenAI-совместимый API)."""

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")

    async def analyze(
        self,
        transcription: dict,
        classes: list[str],
        clip_context: dict[float, dict] | None = None,
    ) -> dict:
        """Возвращает найденные совпадения по категориям с временными метками.

        Формат результата:
        {
            "<class>": [{"start": float, "end": float, "quote": str}],
            "summary": {"<class>": int}
        }
        """
        segments = transcription.get("segments", [])
        empty = {cls: [] for cls in classes} | {"summary": {cls: 0 for cls in classes}}

        if not classes or not segments:
            return empty

        segments_text = self._format_segments(segments)
        clip_section = self._format_clip_context(clip_context or {})
        prompt = _USER_PROMPT.format(
            classes="\n".join(f"- {c}" for c in classes),
            segments=segments_text,
            clip_section=clip_section,
        )

        try:
            raw = await self._call_api(prompt)
        except Exception:
            logger.exception("Ошибка при обращении к LLM API.")
            return empty

        return self._build_result(raw, classes)

    async def analyze_captions(
        self,
        captions: dict[str, str],
        classes: list[str],
    ) -> dict[str, list[str]]:
        """Классифицирует описания кадров по категориям.

        Возвращает {frame_id: [class1, class2]}.
        """
        if not classes or not captions:
            return {}

        captions_text = "\n".join(
            f"[{fid}] {text}" for fid, text in captions.items() if text
        )
        if not captions_text:
            return {}

        prompt = _USER_PROMPT_CAPTIONS.format(
            classes="\n".join(f"- {c}" for c in classes),
            captions=captions_text,
        )
        try:
            raw = await self._call_api(prompt)
        except Exception:
            logger.exception("Ошибка при анализе описаний кадров через LLM.")
            return {}

        result: dict[str, list[str]] = {}
        classes_set = set(classes)
        for frame_id, frame_classes in raw.items():
            if isinstance(frame_classes, list):
                valid = [c for c in frame_classes if c in classes_set]
                if valid:
                    result[str(frame_id)] = valid
        return result

    async def _call_api(self, user_message: str) -> dict:
        last_exc: Exception | None = None
        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                try:
                    response = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json={
                            "model": self._model,
                            "messages": [
                                {"role": "system", "content": _SYSTEM_PROMPT},
                                {"role": "user", "content": user_message},
                            ],
                            "response_format": {"type": "json_object"},
                            "temperature": 0.0,
                        },
                    )
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                    return json.loads(content)
                except httpx.HTTPStatusError as exc:
                    # 4xx (кроме 429) — не ретраить, это ошибка запроса
                    if exc.response.status_code < 500 and exc.response.status_code != 429:
                        raise
                    last_exc = exc
                    logger.warning(
                        "LLM API вернул %d, попытка %d/%d.",
                        exc.response.status_code, attempt, _MAX_ATTEMPTS,
                    )
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    last_exc = exc
                    logger.warning(
                        "LLM API недоступен (%s), попытка %d/%d.",
                        type(exc).__name__, attempt, _MAX_ATTEMPTS,
                    )

                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(_RETRY_BACKOFF_BASE ** (attempt - 1))

        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _format_clip_context(clip_context: dict[float, dict]) -> str:
        if not clip_context:
            return ""
        lines: list[str] = [
            "",
            "Контекст происходящего на экране (визуальные маркеры по временным меткам):",
        ]
        for time_sec, data in clip_context.items():
            prompts = data.get("prompts", [])
            caption = data.get("caption")
            parts = []
            if prompts:
                parts.append(", ".join(f'"{p}"' for p in prompts))
            if caption:
                parts.append(f"описание: {caption}")
            if parts:
                lines.append(f"[{time_sec:.1f} с] {' | '.join(parts)}")
        lines.append("Используй визуальный контекст как дополнительную подсказку.")
        return "\n".join(lines)

    @staticmethod
    def _format_segments(segments: list[dict]) -> str:
        lines = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if text:
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                lines.append(f"[{start:.1f} - {end:.1f}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _build_result(raw: dict, classes: list[str]) -> dict:
        # Collect all matches keyed by (start, end) so overlapping segments merge.
        timeline: dict[tuple[float, float], dict[str, str]] = {}
        summary: dict[str, int] = {cls: 0 for cls in classes}

        for cls in classes:
            matches = raw.get(cls, [])
            for m in matches:
                if not isinstance(m, dict):
                    continue
                start = float(m.get("start", 0))
                end = float(m.get("end", 0))
                key = (start, end)
                timeline.setdefault(key, {})[cls] = str(m.get("quote", ""))
                summary[cls] += 1

        sorted_entries = sorted(timeline.items(), key=lambda x: (x[0][0], x[0][1]))
        return {
            "timeline": [
                {"start": start, "end": end, "classes": classes_map}
                for (start, end), classes_map in sorted_entries
            ],
            "summary": summary,
        }
