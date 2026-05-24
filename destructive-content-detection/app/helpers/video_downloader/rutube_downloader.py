import asyncio
from functools import partial
from pathlib import Path
from typing import Any

import yt_dlp

from .abstract_downloader import VideoDownloader


class RutubeDownloader(VideoDownloader):
    def __init__(self, save_path: str = "./"):
        super().__init__(save_path)

    def _resolve_downloaded_path(
        self, ydl: yt_dlp.YoutubeDL, info: dict[str, Any]
    ) -> Path:
        requested_downloads = info.get("requested_downloads") or []
        for item in requested_downloads:
            file_path = item.get("filepath") or item.get("_filename")
            if file_path:
                return Path(file_path)

        file_path = info.get("_filename")
        if file_path:
            return Path(file_path)

        prepared_path = Path(ydl.prepare_filename(info))
        if prepared_path.exists():
            return prepared_path

        mp4_path = prepared_path.with_suffix(".mp4")
        if mp4_path.exists():
            return mp4_path

        return prepared_path

    def _download_sync(self, url: str, resolution: int | None = None) -> str:
        ydl_opts = {
            "outtmpl": str(self._save_path / "%(title)s.%(ext)s"),
            "merge_output_format": "mp4",
        }

        if resolution:
            ydl_opts["format"] = (
                f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]"
            )

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = self._resolve_downloaded_path(ydl, info)

        return str(file_path.absolute())

    async def download_file(self, url: str, resolution: int | None = None) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self._download_sync, url, resolution)
        )
