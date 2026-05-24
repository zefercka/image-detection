from pathlib import Path
from abc import ABC, abstractmethod


class VideoDownloader(ABC):
    def __init__(self, save_path: str = "./"):
        self._save_path = Path(save_path).expanduser()
        self._save_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def download_file(self, url: str, resolution: int | None = None) -> str: ...
