import hashlib
from typing import BinaryIO

from snowflake import SnowflakeGenerator

_flake_generator = SnowflakeGenerator(42)


def calculate_file_hash(
    file_obj: BinaryIO | bytes,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """Считает хэш для файла, чтобы не хранить одинаковые файлы.

    Args:
        file_obj (BinaryIO | bytes): Объект файла или его байты для
        которого нужно посчитать хэш
        chunk_size (int, optional): Размер чанка для чтения файла
        (по умолчанию 8 МБ)

    Returns:
        str: Хэш файла в формате SHA-256

    """

    sha256 = hashlib.sha256()
    if isinstance(file_obj, bytes):
        sha256.update(file_obj)

        return sha256.hexdigest()

    # Стриминг для больших файлов
    while chunk := file_obj.read(chunk_size):
        sha256.update(chunk)

    # Возвращаем курсор для последующей загрузки
    file_obj.seek(0)

    return sha256.hexdigest()


def generate_flake_id() -> str:
    """Генерирует уникальный идентификатор в стиле Twitter Snowflake.

    Returns:
        str: Уникальный идентификатор

    """

    return str(next(_flake_generator))
