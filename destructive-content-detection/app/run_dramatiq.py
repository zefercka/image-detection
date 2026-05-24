import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).absolute().parent.parent.as_posix())

from dramatiq.cli import main as dramatiq_main


def parse_queue_name() -> str:
    parser = argparse.ArgumentParser()

    parser.add_argument("broker_module")
    parser.add_argument("--queues", "--queue", dest="queue", default="default")

    args, _ = parser.parse_known_args()

    return args.queue


if __name__ == "__main__":
    from app.config import settings

    queue_name = parse_queue_name()

    # Настройка логирования
    log_level = settings.LOG_LEVEL
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )

    logger = logging.getLogger(__name__)

    dramatiq_main()


# python app/run_dramatiq.py app.src.tasks.create_vm_task --queue default --threads 1
