from pathlib import Path
from abc import ABC, abstractmethod


class VideoDownloader(ABC):
    def __init__(self, save_path: str = './'):
        self._save_path = Path(save_path.rstrip('/'))
    
    @abstractmethod
    async def download_file(self, url: str, resolution: int | None = None) -> str:
        ...
