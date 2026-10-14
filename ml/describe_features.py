"""
Traceon — распределения признаков для понятных объяснений.

Признаки Elliptic2 обезличены (feat#1…feat#43), их смысл не раскрыт. Поэтому объяснение строится
через сравнение: «значение больше, чем у 96% законных схем; обычно у законных — 1, у подозрительных — 4».
Скрипт считает по обучающей выборке распределение каждого признака у законных схем
и медианы у законных и подозрительных, и сохраняет их в models/distributions.json.

Также считает, как часто встречается каждый тип схемы (цепочка, дробление, сбор, цикл)
и какая доля таких схем подозрительная — models/typology_stats.json.

Переобучать модель не нужно. Запуск из корня проекта:
  python ml/describe_features.py
  docker compose run --rm ml ml/describe_features.py
"""
import json
import sys

import numpy as np
import pandas as pd

from common import MODELS, PROCESSED, ROOT

sys.path.insert(0, str(ROOT / "backend"))
from app.services.typology import detect  # noqa: E402

N_QUANTILES = 200


def _num(x):
    return None if x is None or pd.isna(x) else round(float(x), 6)


def build_distributions(train: pd.DataFrame, feats) -> dict:
    licit = train[train["label"] == 0]
    susp = train[train["label"] == 1]
    grid = np.linspace(0, 1, N_QUANTILES + 1)
    out = {}
    for f in feats:
        values = licit[f].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        out[f] = {
            "licit_q": [round(float(v), 6) for v in np.quantile(values, grid)],
            "licit_median": _num(np.median(values)),
            "susp_median": _num(susp[f].replace([np.inf, -np.inf], np.nan).median()),
        }
    return out


def build_typology_stats(sub: pd.DataFrame) -> dict:
    """Сколько схем каждого типа в обучающей выборке и какая доля из них подозрительная."""
    ids = set(sub["cc_id"].astype(str))
    members = pd.read_parquet(PROCESSED / "membership.parquet")
    edges = pd.read_parquet(PROCESSED / "sub_edges.parquet")
    members = members[members["cc_id"].astype(str).isin(ids)]
    edges = edges[edges["cc_id"].astype(str).isin(ids)]
    nodes_by = members.groupby("cc_id")["cl_id"].apply(list).to_dict()
    edges_by = edges.groupby("cc_id")[["src", "dst"]].apply(lambda g: list(zip(g["src"], g["dst"]))).to_dict()
    labels = dict(zip(sub["cc_id"], sub["label"]))
    counts, bad = {}, {}
    for cc, nodes in nodes_by.items():
        for p in detect(nodes, edges_by.get(cc, [])):
            counts[p["code"]] = counts.get(p["code"], 0) + 1
            bad[p["code"]] = bad.get(p["code"], 0) + int(labels.get(cc, 0))
    stats = {code: {"count": n, "suspicious_share": round(bad[code] / n, 4)} for code, n in counts.items()}
    stats["_base_rate"] = round(float(sub["label"].mean()), 4)
    return stats


def main() -> None:
    feats = json.loads((MODELS / "model_meta.json").read_text())["features"]
    df = pd.read_parquet(PROCESSED / "subgraphs.parquet")
    dist = build_distributions(df[df["split"] == "train"], feats)
    (MODELS / "distributions.json").write_text(json.dumps(dist, ensure_ascii=False))
    print(f"Готово: распределения {len(dist)} признаков сохранены в {MODELS / 'distributions.json'}")

    print("Считаю типы схем...")
    stats = build_typology_stats(df[df["split"] == "train"])
    (MODELS / "typology_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    base = stats["_base_rate"]
    print(f"Типы схем в обучающей выборке (в среднем подозрительных {base * 100:.1f}%):")
    for code, s in sorted(((c, v) for c, v in stats.items() if c != "_base_rate"), key=lambda kv: -kv[1]["suspicious_share"]):
        print(f"  {code:15s} {s['count']:>8,} схем, подозрительных {s['suspicious_share'] * 100:5.1f}%")
    print("Перезапустите сервис, чтобы он подхватил файл: docker compose up -d --build api")


if __name__ == "__main__":
    main()
