"""
Traceon — маленький синтетический датасет в формате Elliptic2.

Нужен, чтобы проверить весь проект (подготовка → обучение → база → сайт) за пару минут,
не дожидаясь обработки настоящего Elliptic2 на десятки гигабайт.
Данные выдуманные: метрики на них ничего не значат.

Запуск из корня проекта:
  python ml/make_sample_data.py              # создаст data/elliptic2/ (если там пусто)
  python ml/make_sample_data.py --force      # перезапишет файлы
"""
import argparse

import numpy as np
import pandas as pd

from common import RAW

N_SUBGRAPHS = 4000
SUSPICIOUS_SHARE = 0.04
N_BACKGROUND_EXTRA = 20000
N_NODE_FEATS, N_EDGE_FEATS = 43, 95


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if (RAW / "background_nodes.csv").exists() and not args.force:
        raise SystemExit(f"В {RAW} уже есть данные. Добавьте --force, чтобы перезаписать их синтетикой.")
    RAW.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    def cl_id():
        return rng.bytes(8).hex()

    cc_rows, node_rows, edge_rows = [], [], []
    for i in range(N_SUBGRAPHS):
        cc = str(100000 + i)
        bad = rng.random() < SUSPICIOUS_SHARE
        n = int(rng.integers(2, 9 if bad else 6))
        ids = [cl_id() for _ in range(n)]
        node_rows += [(c, cc) for c in ids]
        if bad and rng.random() < 0.5:            # веер: один кластер раздаёт средства многим
            edge_rows += [(ids[0], c, cc) for c in ids[1:]]
        elif bad:                                  # цепочка «пилинга»
            edge_rows += [(ids[k], ids[k + 1], cc) for k in range(n - 1)]
        else:                                      # обычное дерево переводов
            edge_rows += [(ids[int(rng.integers(0, k))], ids[k], cc) for k in range(1, n)]
        cc_rows.append((cc, "suspicious" if bad else "licit"))

    cc_df = pd.DataFrame(cc_rows, columns=["ccId", "ccLabel"])
    nodes = pd.DataFrame(node_rows, columns=["clId", "ccId"])
    edges = pd.DataFrame(edge_rows, columns=["clId1", "clId2", "ccId"])
    bad_nodes = set(nodes.loc[nodes["ccId"].isin(cc_df.loc[cc_df["ccLabel"] == "suspicious", "ccId"]), "clId"])

    all_ids = list(nodes["clId"]) + [cl_id() for _ in range(N_BACKGROUND_EXTRA)]
    feats = rng.integers(0, 10, size=(len(all_ids), N_NODE_FEATS)).astype(float)
    is_bad = np.array([c in bad_nodes for c in all_ids])
    feats[is_bad, 2] = np.clip(feats[is_bad, 2] + rng.integers(2, 6, is_bad.sum()), 0, 12)   # сигнал
    feats[is_bad, 7] = np.clip(feats[is_bad, 7] - rng.integers(1, 5, is_bad.sum()), 0, 12)
    bnodes = pd.DataFrame(feats, columns=[f"feat#{k}" for k in range(1, N_NODE_FEATS + 1)])
    bnodes.insert(0, "clId", all_ids)

    extra = pd.DataFrame({"clId1": rng.choice(all_ids, 30000), "clId2": rng.choice(all_ids, 30000)})
    bedges = pd.concat([edges[["clId1", "clId2"]], extra], ignore_index=True)
    efeats = rng.integers(0, 10, size=(len(bedges), N_EDGE_FEATS)).astype(float)
    bedges = pd.concat([bedges, pd.DataFrame(efeats, columns=[f"feat#{k}" for k in range(1, N_EDGE_FEATS + 1)])],
                       axis=1)

    cc_df.to_csv(RAW / "connected_components.csv", index=False)
    nodes.to_csv(RAW / "nodes.csv", index=False)
    edges.to_csv(RAW / "edges.csv", index=False)
    bnodes.to_csv(RAW / "background_nodes.csv", index=False)
    bedges.to_csv(RAW / "background_edges.csv", index=False)
    print(f"Синтетический датасет создан в {RAW}: {len(cc_df):,} подграфов, {len(nodes):,} кластеров")


if __name__ == "__main__":
    main()
