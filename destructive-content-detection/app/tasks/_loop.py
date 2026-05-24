import asyncio
import threading

_thread_loop: threading.local = threading.local()


def get_loop() -> asyncio.AbstractEventLoop:
    """Возвращает постоянный event loop для текущего потока.

    asyncio.run() создаёт новый loop и закрывает его после каждого вызова,
    из-за чего asyncpg-соединения становятся невалидными при retry.
    Этот хелпер создаёт loop один раз на поток и переиспользует его.
    """
    loop = getattr(_thread_loop, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _thread_loop.loop = loop
    return loop
