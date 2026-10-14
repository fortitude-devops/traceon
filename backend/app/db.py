"""
Подключение к PostgreSQL и схема базы.

  subgraphs   — подграфы Elliptic2: метка, часть выборки, признаки модели (JSONB)
  membership  — какой кластер в какой подграф входит
  sub_edges   — переводы между кластерами внутри подграфа (для графа)
  blacklist   — кластеры из санкционных / внутренних списков
  checks      — журнал всех проверок (для отчётности перед регулятором)
"""
from sqlalchemy import create_engine, text

from .config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
PSYCOPG_URL = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")

DATA_TABLES_SQL = """
DROP TABLE IF EXISTS sub_edges;
DROP TABLE IF EXISTS membership;
DROP TABLE IF EXISTS subgraphs;
CREATE TABLE subgraphs (
    cc_id     TEXT PRIMARY KEY,
    label     SMALLINT NOT NULL,      -- 1 подозрительный, 0 законный (только для оценки модели)
    split     TEXT NOT NULL,          -- train / val / test
    n_nodes   INTEGER,
    n_edges   INTEGER,
    features  JSONB NOT NULL
);
CREATE TABLE membership (
    cl_id  TEXT NOT NULL,
    cc_id  TEXT NOT NULL
);
CREATE TABLE sub_edges (
    cc_id  TEXT NOT NULL,
    src    TEXT NOT NULL,
    dst    TEXT NOT NULL
);
"""

DATA_INDEXES_SQL = """
CREATE INDEX membership_cl_idx ON membership (cl_id);
CREATE INDEX membership_cc_idx ON membership (cc_id);
CREATE INDEX sub_edges_cc_idx ON sub_edges (cc_id);
"""

SERVICE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS blacklist (
    cl_id     TEXT PRIMARY KEY,
    source    TEXT NOT NULL,
    reason    TEXT,
    added_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS checks (
    id                  SERIAL PRIMARY KEY,
    query               TEXT NOT NULL,
    cc_id               TEXT,
    score               INTEGER NOT NULL,
    risk_level          TEXT NOT NULL,
    proba               DOUBLE PRECISION,
    reasons             JSONB,
    conclusion          TEXT,
    conclusion_source   TEXT,
    blacklisted         BOOLEAN NOT NULL DEFAULT false,
    signature_verified  BOOLEAN NOT NULL DEFAULT false,
    signed_address      TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS checks_created_idx ON checks (created_at DESC);
"""


def init_service_tables() -> None:
    with engine.begin() as conn:
        for stmt in SERVICE_TABLES_SQL.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
