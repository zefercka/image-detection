import tempfile
from contextlib import suppress

from app.dependencies.s3.client import S3Client


async def download_to_file(
    s3: S3Client,
    bucket_name: str,
    key: str,
    suffix: str | None = None,
) -> str:
    """Скачивает файл из S3 и сохраняет его во временный файл через стриминг.

    Args:
        s3 (S3Client): Клиент для работы с S3.
        bucket_name (str): Имя бакета, из которого будет загружен файл.
        key (str): Имя файла в бакете.
        suffix (str | None): Необязательное расширение для временного файла.

    Returns:
        str: Путь к временному файлу с содержимым из S3.

    """

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            async for chunk in s3.download_file_stream(bucket_name, key):
                tmp_file.write(chunk)

            return tmp_file.name
    except Exception:
        if tmp_file is not None:
            with suppress(FileNotFoundError):
                tmp_file.unlink()
        raise
