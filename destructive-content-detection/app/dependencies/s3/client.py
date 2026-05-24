import io
import logging
from collections.abc import AsyncIterator
from typing import Annotated, BinaryIO

import aioboto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from fastapi import Depends

from app.config import settings


class S3Client:
    """Асинхронный клиент для работы с S3-совместимыми хранилищами."""

    def __init__(self, endpoint_url: str, access_key: str, secret_key: str) -> None:
        self.session = aioboto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self.endpoint_url = endpoint_url
        self.logger = logging.getLogger(__name__)

    async def create_bucket(self, bucket_name: str) -> None:
        """Создаёт бакет, если он не существует."""

        async with self.session.client("s3", endpoint_url=self.endpoint_url) as s3:
            try:
                await s3.head_bucket(Bucket=bucket_name)
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code in ("404", "NoSuchBucket"):
                    await s3.create_bucket(Bucket=bucket_name)
                else:
                    raise

    async def upload_file(
        self,
        bucket_name: str,
        file_name: str,
        file_content: bytes | BinaryIO,
        content_type: str = "application/octet-stream",
        chunk_size: int = 8 * 1024 * 1024,
    ) -> None:
        """Загружает файл в S3.

        Автоматически переключается на запись через стрим, если на вход
        подаётся `BinaryIO`

        Args:
            bucket_name (str): Имя бакета, в который будет загружен файл.
            file_name (str): Имя файла в бакете.
            file_content (bytes | BinaryIO): Содержимое файла.
            content_type (str, optional): MIME-тип файла. По умолчанию
            "application/octet-stream".
            chunk_size (int, optional): Размер чанка для многопоточной
            загрузки. По умолчанию 8 МБ.

        """

        file_obj = io.BytesIO(file_content) if isinstance(file_content, bytes) else file_content
        config = TransferConfig(
            multipart_threshold=chunk_size,
            multipart_chunksize=chunk_size,
            max_concurrency=4,
        )

        async with self.session.client("s3", endpoint_url=self.endpoint_url) as s3:
            await s3.upload_fileobj(
                file_obj,
                bucket_name,
                file_name,
                ExtraArgs={"ContentType": content_type},
                Config=config,
            )

    async def download_file(self, bucket_name: str, file_name: str) -> bytes:
        """Скачивает файл из S3 и возвращает его содержимое в виде байтов."""

        async with self.session.client("s3", endpoint_url=self.endpoint_url) as s3:
            response = await s3.get_object(Bucket=bucket_name, Key=file_name)
            return await response["Body"].read()

    async def download_file_stream(
        self,
        bucket_name: str,
        file_name: str,
        chunk_size: int = 8 * 1024 * 1024,
    ) -> AsyncIterator[bytes]:
        """Стримит файлы с S3. Удобно для больших файлов.

        Args:
            bucket_name (str): Имя бакета, из которого будет загружен файл.
            file_name (str): Имя файла в бакете.
            chunk_size (int, optional): Размер чанка для стриминга. По умолчанию 8 МБ.

        Yields:
            Iterator[AsyncIterator[bytes]]: Итератор асинхронных итераторов байтов файла.

        """

        async with self.session.client("s3", endpoint_url=self.endpoint_url) as s3:
            response = await s3.get_object(Bucket=bucket_name, Key=file_name)
            body = response["Body"]
            try:
                while chunk := await body.read(chunk_size):
                    yield chunk
            finally:
                body.close()

    async def delete_file(self, bucket_name: str, file_name: str) -> None:
        """Удаляет файл с S3.

        Args:
            bucket_name (str): Имя бакета, из которого будет удалён файл.
            file_name (str): Имя файла в бакете.

        """

        async with self.session.client("s3", endpoint_url=self.endpoint_url) as s3:
            await s3.delete_object(Bucket=bucket_name, Key=file_name)

    async def list_files(self, bucket_name: str, prefix: str = "") -> list[str]:
        """Возвращает список файлов в бакете.

        Args:
            bucket_name (str): Имя бакета, из которого будут получены файлы.
            prefix (str, optional): Префикс для фильтрации файлов. По умолчанию "".

        Returns:
            list[str]: Список имен файлов в бакете.

        """

        keys = []
        async with self.session.client("s3", endpoint_url=self.endpoint_url) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys

    async def file_exists(self, bucket_name: str, file_name: str) -> bool:
        """Проверяет, существует ли файл в S3.

        Args:
            bucket_name (str): Имя бакета, в котором будет проверяться файл.
            file_name (str): Имя файла в бакете.

        Returns:
            bool: True, если файл существует, иначе False.

        """

        async with self.session.client("s3", endpoint_url=self.endpoint_url) as s3:
            try:
                await s3.head_object(Bucket=bucket_name, Key=file_name)
            except ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                    return False
                raise

            return True

    async def get_presigned_url(
        self,
        bucket_name: str,
        file_name: str,
        expires_in: int = 3600,
    ) -> str:
        """Генерирует предподписанный URL для доступа к файлу в S3.

        Args:
            bucket_name (str): Имя бакета, в котором находится файл.
            file_name (str): Имя файла в бакете.
            expires_in (int, optional): Время жизни URL в секундах. По умолчанию 3600.

        Returns:
            str: Предподписанный URL для доступа к файлу.

        """
        async with self.session.client("s3", endpoint_url=self.endpoint_url) as client:
            return await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": file_name},
                ExpiresIn=expires_in,
            )


def get_s3_client() -> S3Client:
    """Cоздаёт и возвращает экземпляр S3Client с настройками из конфигурации."""

    return S3Client(
        endpoint_url=settings.S3_ENDPOINT_URL,
        access_key=settings.S3_ACCESS_KEY,
        secret_key=settings.S3_SECRET_KEY,
    )


S3ClientDependency = Annotated[S3Client, Depends(get_s3_client)]
