import asyncio
from functools import partial
from pathlib import Path

import yt_dlp

from .abstract_downloader import VideoDownloader


class RutubeDownloader(VideoDownloader):
    def __init__(self, save_path: str = './'):
        super().__init__(save_path)
    
    def _download_sync(self, url: str, resolution: int | None = None) -> str:
        ydl_opts = {
            'outtmpl': str(self._save_path / '%(title)s.%(ext)s'),
            'merge_output_format': 'mp4',
        }

        if resolution:
            ydl_opts['format'] = f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]'

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            filename = ydl.prepare_filename(info)
            file_path = Path(filename).with_suffix('.mp4')

            ydl.download([url])

        return str(file_path.absolute())

    async def download_file(self, url: str, resolution: int | None = None) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._download_sync, url, resolution))
