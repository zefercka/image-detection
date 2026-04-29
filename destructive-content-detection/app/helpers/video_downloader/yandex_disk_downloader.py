import uuid
from urllib.parse import urlencode

import magic
from aiofile import async_open
from httpx import AsyncClient

from .abstract_downloader import VideoDownloader

_BASE_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"


class YandexDiskDownloader(VideoDownloader):
    def __init__(
        self,
        base_url: str = _BASE_URL,
        save_path: str = "./",
    ):
        super().__init__(save_path)
        self._base_url = base_url

    async def _get_download_link(self, url: str) -> str:
        get_url = self._base_url + urlencode(dict(public_key=url))
        async with AsyncClient() as client:
            response = await client.get(get_url)
            response.raise_for_status()
        return response.json()["href"]

    async def download_file(self, url: str, resolution: int | None = None) -> str:
        download_link = await self._get_download_link(url)

        async with AsyncClient() as client:
            download_response = await client.get(download_link)
            download_response.raise_for_status()

        filename = self._get_file_name(download_response.content)
        file_path = self._save_path / filename

        async with async_open(str(file_path), "wb") as f:
            await f.write(download_response.content)

        return str(file_path.absolute())

    def _get_file_name(self, content: bytes) -> str:
        file_extension = magic.from_buffer(content, mime=True).split("/")[-1]
        return f"{uuid.uuid4()}.{file_extension}"
