"""
Traceon — «понятные двойники» скрытых признаков Elliptic2.

Смысл признаков кластеров (node_N_*) и переводов (edge_N_*) Elliptic не раскрывает. Но часть из них
сильно связана с признаками, которые мы считаем сами по сети Bitcoin (например, число входящих связей
кластера). Скрипт считает ранговую корреляцию Спирмена на обучающей выборке и сохраняет для каждого
скрытого признака самый похожий понятный признак, если |r| >= 0,6. Это связь, а не расшифровка:
в интерфейсе пишется «связана с …», а не «это …».

Запуск: python ml/proxy_features.py  →  models/feature_proxies.json
"""
import json

import pandas as pd

from common import MODELS, PROCESSED

THRESHOLD = 0.6
SAMPLE = 40000


def main():
    meta = json.loads((MODELS / "model_meta.json").read_text())
    feats = meta["features"]
    df = pd.read_parquet(PROCESSED / "subgraphs.parquet")
    train = df[df["split"] == "train"]
    train = train.sample(min(SAMPLE, len(train)), random_state=0)
    hidden = [f for f in feats if f.startswith(("node_", "edge_"))]
    readable = [f for f in feats if f not in hidden]
    corr = train[hidden + readable].rank().corr().loc[hidden, readable]
    out = {}
    for f in hidden:
        row = corr.loc[f].dropna()
        if row.empty:
            continue
        best = row.abs().idxmax()
        r = float(row[best])
        if abs(r) >= THRESHOLD:
            out[f] = {"proxy": best, "r": round(r, 2)}
    (MODELS / "feature_proxies.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"Скрытых признаков с понятным двойником (|r| >= {THRESHOLD}): {len(out)} из {len(hidden)}")
    for f, v in sorted(out.items(), key=lambda kv: -abs(kv[1]["r"])):
        print(f"  {f:16s} ~ {v['proxy']:16s} r = {v['r']:+.2f}")


if __name__ == "__main__":
    main()
