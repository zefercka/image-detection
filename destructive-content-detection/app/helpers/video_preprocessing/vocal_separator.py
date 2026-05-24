import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class VocalSeparator:
    """Отделяет вокал от музыки с помощью Demucs."""

    def __init__(self) -> None:
        self._temp_dirs: list[Path] = []

    def separate(self, audio_path: str) -> str:
        output_dir = Path(tempfile.mkdtemp(prefix="demucs_"))
        self._temp_dirs.append(output_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        stem = Path(audio_path).stem

        logger.info("Demucs: отделение вокала из '%s' на %s", audio_path, device)
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "demucs",
                    "--two-stems=vocals",
                    "--device", device,
                    "--out", str(output_dir),
                    audio_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Demucs stderr: %s", e.stderr)
            logger.error("Demucs stdout: %s", e.stdout)
            raise
        logger.debug("Demucs stdout: %s", result.stdout)

        vocals_path = output_dir / "htdemucs" / stem / "vocals.wav"
        if not vocals_path.exists():
            raise FileNotFoundError(f"Demucs: файл вокала не найден: {vocals_path}")

        return str(vocals_path)

    def __enter__(self) -> "VocalSeparator":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        self._temp_dirs.clear()
        return False
