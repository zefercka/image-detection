import time

t_start = time.time()

# ruff: noqa: E402
import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.pipelines.controller import router as pipeline_router
from app.api.videos.controller import router as video_router

from .config import settings

logging.basicConfig(
    level=settings.LOG_LEVEL, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {exc}\n"
        f"Path: {request.method} {request.url}\n"
        f"Traceback:\n{traceback.format_exc()}",
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


app.include_router(video_router)
app.include_router(pipeline_router)

logger.info(f"Application started in {time.time() - t_start} seconds")
