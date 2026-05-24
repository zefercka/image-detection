import dramatiq
from dramatiq.brokers.rabbitmq import RabbitmqBroker

from app.config import settings

rabbit_mq_broker = RabbitmqBroker(
    url=settings.RABBITMQ_URL,
    confirm_delivery=True,
)

dramatiq.set_broker(rabbit_mq_broker)
rabbit_mq_broker.add_middleware(dramatiq.middleware.AsyncIO())
