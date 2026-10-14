"""
Traceon — шаг 3: загрузка подготовленных данных в PostgreSQL.

  1. Пересоздаёт таблицы данных (subgraphs, membership, sub_edges).
     Журнал проверок (checks) и чёрный список (blacklist) сохраняются.
  2. Загружает подграфы с признаками, состав подграфов и переводы через COPY (быстро).
  3. Если есть файл data/blacklist.csv (колонки cl_id,source,reason) — добавляет его в чёрный список.

Запуск из корня проекта:  python ml/load_db.py
"""
import io
import json
import sys
import time

import numpy as np
import pandas as pd
import psycopg

from common import DATA, PROCESSED, ROOT

sys.path.insert(0, str(ROOT / "backend"))
from app.db import DATA_INDEXES_SQL, DATA_TABLES_SQL, PSYCOPG_URL, SERVICE_TABLES_SQL  # noqa: E402

CHUNK = 100_000


def copy_frame(cur, table: str, df: pd.DataFrame) -> None:
    cols = ", ".join(df.columns)
    for start in range(0, len(df), CHUNK):
        buf = io.StringIO()
        df.iloc[start:start + CHUNK].to_csv(buf, index=False, header=False)
        with cur.copy(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv)") as copy:
            copy.write(buf.getvalue())
        print(f"  {table}: {min(start + CHUNK, len(df)):,} из {len(df):,}", end="\r", flush=True)
    print()


def features_json(df: pd.DataFrame, feats) -> pd.Series:
    values = df[feats].astype(float).replace([np.inf, -np.inf], np.nan)
    records = values.to_dict("records")
    return pd.Series(
        [json.dumps({k: (None if v != v else round(v, 6)) for k, v in r.items()}) for r in records],
        index=df.index,
    )


def main() -> None:
    started = time.time()
    feats = json.loads((PROCESSED / "feature_cols.json").read_text())["features"]
    sub = pd.read_parquet(PROCESSED / "subgraphs.parquet")
    members = pd.read_parquet(PROCESSED / "membership.parquet")
    edges = pd.read_parquet(PROCESSED / "sub_edges.parquet")

    out = pd.DataFrame({
        "cc_id": sub["cc_id"].astype(str),
        "label": sub["label"].astype(int),
        "split": sub["split"],
        "n_nodes": sub["n_nodes"].fillna(0).astype(int),
        "n_edges": sub["n_edges"].fillna(0).astype(int),
    })
    print("Готовлю признаки...")
    out["features"] = features_json(sub, feats)

    with psycopg.connect(PSYCOPG_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(SERVICE_TABLES_SQL)
            cur.execute(DATA_TABLES_SQL)
            copy_frame(cur, "subgraphs", out)
            copy_frame(cur, "membership", members[["cl_id", "cc_id"]].astype(str))
            copy_frame(cur, "sub_edges", edges[["cc_id", "src", "dst"]].astype(str))
            print("Строю индексы...")
            cur.execute(DATA_INDEXES_SQL)

            bl_path = DATA / "blacklist.csv"
            if bl_path.exists():
                bl = pd.read_csv(bl_path, dtype=str).fillna("")
                for r in bl.itertuples(index=False):
                    cur.execute(
                        "INSERT INTO blacklist (cl_id, source, reason) VALUES (%s, %s, %s) "
                        "ON CONFLICT (cl_id) DO UPDATE SET source = EXCLUDED.source, reason = EXCLUDED.reason",
                        (r.cl_id, r.source or "внутренний список", r.reason or None),
                    )
                print(f"Чёрный список: добавлено {len(bl):,} кластеров из {bl_path.name}")
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("ANALYZE")
            print("\nИтог:")
            for table in ("subgraphs", "membership", "sub_edges", "blacklist", "checks"):
                cur.execute(f"SELECT count(*), pg_size_pretty(pg_total_relation_size('{table}')) FROM {table}")
                n, size = cur.fetchone()
                print(f"  {table:11s} {n:>10,} строк, {size}")
    print(f"\nГотово за {time.time() - started:.0f} с.")


if __name__ == "__main__":
    main()
