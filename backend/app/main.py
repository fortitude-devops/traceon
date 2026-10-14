"""
Traceon — ML-оценка AML-рисков криптовалютных переводов для банков РК.

Запуск локально (из папки backend):  uvicorn app.main:app --reload
Интерфейс: http://localhost:8000      Документация API: http://localhost:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import config
from .db import init_service_tables
from .routers.api import router
from .services.model import RiskModel

log = logging.getLogger("traceon")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = RiskModel(config.MODELS_DIR)
    if not app.state.model.ready:
        log.warning(app.state.model.error)
    try:
        init_service_tables()
    except Exception as exc:
        log.warning("База данных недоступна: %s", str(exc).splitlines()[0])
    yield


app = FastAPI(
    title="Traceon API",
    description="Оценка AML-рисков криптовалютных переводов: скоринг подграфов Elliptic2, "
                "объяснения, проверка подписи владельца адреса, PDF-отчёты и журнал проверок.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
