"""
Traceon — шаг 1: подготовка датасета Elliptic2.

Elliptic2 размечает не отдельные адреса, а ПОДГРАФЫ: небольшие группы связанных кластеров
(кластер = группа адресов одного владельца), по которым прошли деньги. Метка подграфа:
licit (законный) или suspicious (подозрительный, похож на отмывание).

Файлы (положите в data/elliptic2/):
  connected_components.csv  — подграфы и их метки (ccId, ccLabel)
  nodes.csv                 — какие кластеры входят в какой подграф (clId, ccId)
  edges.csv                 — переводы внутри подграфов (clId1, clId2, ...)
  background_nodes.csv      — признаки всех кластеров (clId, feat#1..43) — ~49 млн строк
  background_edges.csv      — признаки всех переводов (clId1, clId2, feat#1..95) — ~196 млн строк

Что делает скрипт:
  1. Строит структурные признаки каждого подграфа (размер, плотность, веерность, цепочки).
  2. Потоково читает background_nodes.csv, оставляет только кластеры из подграфов
     и агрегирует их признаки по подграфу (среднее, максимум, минимум).
  3. Окружение подграфа во всей сети: из background_edges.csv читаются только две колонки id,
     считаются связи кластеров со всей сетью и переводы, уходящие из схемы наружу и приходящие извне.
  4. (по флагу --edge-features) агрегаты признаков переводов из background_edges.csv.
  5. Делит подграфы на обучение / валидацию / тест (80/10/10, со стратификацией).
  5. Сохраняет всё в data/processed/.

Запуск из корня проекта:
  python ml/prepare_data.py                       # структура + признаки кластеров + окружение
  python ml/prepare_data.py --no-neighborhood     # без окружения (быстрее, но модель слабее)
  python ml/prepare_data.py --edge-features       # + признаки переводов (долго: 196 млн строк)
  python ml/prepare_data.py --limit-rows 2000000  # быстрый пробный прогон
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from sklearn.model_selection import train_test_split

from common import PROCESSED, RAW, feature_suffix, find_col

STATS = ("mean", "max", "min")


def header(path):
    return pd.read_csv(path, nrows=0).columns.tolist()


def label_to_int(series: pd.Series) -> pd.Series:
    """'suspicious' / 'illicit' / 1 -> 1, 'licit' / 0 -> 0."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0).astype(int)
    s = series.astype(str).str.lower()
    return s.str.contains("susp|illicit").astype(int)


def stream_filter(path, id_cols, keep_ids: set, limit_rows=None, both=False):
    """
    Потоково читает большой CSV и оставляет строки, где id из keep_ids.
    both=True — оба id (для переводов) должны быть в наборе.
    """
    cols = header(path)
    types = {c: pa.string() for c in id_cols}
    for c in cols:
        if c not in id_cols:
            types[c] = pa.float32()
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(column_types=types),
    )
    value_set = pa.array(list(keep_ids), type=pa.string())
    parts, seen, started = [], 0, time.time()
    for batch in reader:
        masks = [pc.is_in(batch.column(c), value_set=value_set) for c in id_cols]
        mask = masks[0]
        for m in masks[1:]:
            mask = pc.and_(mask, m) if both else pc.or_(mask, m)
        kept = batch.filter(mask)
        if kept.num_rows:
            parts.append(kept.to_pandas())
        seen += batch.num_rows
        rate = seen / max(time.time() - started, 1e-6)
        print(f"  {path.name}: прочитано {seen:,} строк ({rate:,.0f} строк/с), найдено {sum(len(p) for p in parts):,}",
              end="\r", flush=True)
        if limit_rows and seen >= limit_rows:
            break
    print()
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def stream_degrees(path, src_col, dst_col, keep_ids: set, limit_rows=None) -> pd.DataFrame:
    """
    Окружение кластеров во ВСЕЙ сети: читает из background_edges.csv только две колонки id
    и считает для каждого нужного кластера число исходящих/входящих связей,
    а также связей с кластерами вне размеченных подграфов (внешний мир).
    """
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(
            include_columns=[src_col, dst_col],
            column_types={src_col: pa.string(), dst_col: pa.string()},
        ),
    )
    value_set = pa.array(list(keep_ids), type=pa.string())
    totals = {k: pd.Series(dtype="int64") for k in ("out_deg", "in_deg", "out_ext", "in_ext")}

    def count(arr) -> pd.Series:
        if len(arr) == 0:
            return pd.Series(dtype="int64")
        vc = pc.value_counts(arr)
        return pd.Series(vc.field("counts").to_numpy(), index=vc.field("values").to_numpy(zero_copy_only=False))

    seen, started = 0, time.time()
    for batch in reader:
        src, dst = batch.column(src_col), batch.column(dst_col)
        src_in = pc.is_in(src, value_set=value_set)
        dst_in = pc.is_in(dst, value_set=value_set)
        parts = {
            "out_deg": pc.filter(src, src_in),
            "in_deg": pc.filter(dst, dst_in),
            "out_ext": pc.filter(src, pc.and_(src_in, pc.invert(dst_in))),
            "in_ext": pc.filter(dst, pc.and_(dst_in, pc.invert(src_in))),
        }
        for key, arr in parts.items():
            totals[key] = totals[key].add(count(arr), fill_value=0)
        seen += batch.num_rows
        rate = seen / max(time.time() - started, 1e-6)
        print(f"  {path.name} (окружение): прочитано {seen:,} строк ({rate:,.0f} строк/с)", end="\r", flush=True)
        if limit_rows and seen >= limit_rows:
            break
    print()
    deg = pd.DataFrame(totals).fillna(0).astype("int64")
    deg.index.name = "cl_id"
    return deg


def aggregate(df: pd.DataFrame, group_col: str, feat_cols, prefix: str) -> pd.DataFrame:
    agg = df.groupby(group_col)[feat_cols].agg(list(STATS))
    agg.columns = [f"{prefix}_{feature_suffix(f)}_{s}" for f, s in agg.columns]
    return agg


def structural_features(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Структура подграфа: сколько узлов и связей, есть ли веер, сбор, цепочка."""
    n_nodes = nodes.groupby("cc_id")["cl_id"].nunique().rename("n_nodes")
    e = edges.drop_duplicates(["cc_id", "src", "dst"])
    n_edges = e.groupby("cc_id").size().rename("n_edges")
    out_deg = e.groupby(["cc_id", "src"]).size()
    in_deg = e.groupby(["cc_id", "dst"]).size()
    max_out = out_deg.groupby(level=0).max().rename("max_out_degree")
    max_in = in_deg.groupby(level=0).max().rename("max_in_degree")
    n_senders = out_deg.groupby(level=0).size().rename("n_senders")
    n_receivers = in_deg.groupby(level=0).size().rename("n_receivers")

    f = pd.concat([n_nodes, n_edges, max_out, max_in, n_senders, n_receivers], axis=1).fillna(0)
    f["n_sources"] = f["n_nodes"] - f["n_receivers"]          # получают деньги только извне подграфа
    f["n_sinks"] = f["n_nodes"] - f["n_senders"]              # дальше деньги не передают
    pairs = f["n_nodes"] * (f["n_nodes"] - 1)
    f["density"] = np.where(pairs > 0, f["n_edges"] / pairs, 0.0)
    f["edges_per_node"] = np.where(f["n_nodes"] > 0, f["n_edges"] / f["n_nodes"], 0.0)
    f["is_chain"] = ((f["max_out_degree"] <= 1) & (f["max_in_degree"] <= 1)).astype(int)
    f["fan_out_ratio"] = np.where(f["n_nodes"] > 0, f["max_out_degree"] / f["n_nodes"], 0.0)
    f["fan_in_ratio"] = np.where(f["n_nodes"] > 0, f["max_in_degree"] / f["n_nodes"], 0.0)
    return f.drop(columns=["n_senders", "n_receivers"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-features", action="store_true", help="добавить признаки переводов (долго)")
    parser.add_argument("--no-neighborhood", action="store_true", help="не считать окружение подграфа в сети")
    parser.add_argument("--limit-rows", type=int, default=None, help="читать не больше N строк больших файлов")
    args = parser.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # 1. Подграфы и метки
    cc_path = RAW / "connected_components.csv"
    cc_cols = header(cc_path)
    cc_id_col = find_col(cc_cols, "ccId", position=0)
    cc_label_col = find_col(cc_cols, "ccLabel", "label", position=1)
    cc = pd.read_csv(cc_path, dtype={cc_id_col: str})
    cc = pd.DataFrame({"cc_id": cc[cc_id_col], "label": label_to_int(cc[cc_label_col])})
    print(f"Подграфов: {len(cc):,}, подозрительных: {cc['label'].sum():,} ({cc['label'].mean() * 100:.1f}%)")

    # 2. Состав подграфов
    n_path = RAW / "nodes.csv"
    n_cols = header(n_path)
    n_cl = find_col(n_cols, "clId", position=0)
    n_cc = find_col(n_cols, "ccId", position=1)
    nodes = pd.read_csv(n_path, usecols=[n_cl, n_cc], dtype=str).rename(columns={n_cl: "cl_id", n_cc: "cc_id"})
    nodes = nodes.drop_duplicates()
    print(f"Кластеров в подграфах: {nodes['cl_id'].nunique():,}")

    # 3. Переводы внутри подграфов
    e_path = RAW / "edges.csv"
    e_cols = header(e_path)
    e_src = find_col(e_cols, "clId1", "src", position=0)
    e_dst = find_col(e_cols, "clId2", "dst", position=1)
    try:
        e_cc = find_col(e_cols, "ccId")
    except KeyError:
        e_cc = None
    use = [e_src, e_dst] + ([e_cc] if e_cc else [])
    edges = pd.read_csv(e_path, usecols=use, dtype=str).rename(columns={e_src: "src", e_dst: "dst"})
    if e_cc:
        edges = edges.rename(columns={e_cc: "cc_id"})
    else:
        first_cc = nodes.drop_duplicates("cl_id").set_index("cl_id")["cc_id"]
        edges["cc_id"] = edges["src"].map(first_cc)
    edges = edges.dropna().drop_duplicates(["cc_id", "src", "dst"])
    print(f"Переводов внутри подграфов: {len(edges):,}")

    # 4. Структурные признаки
    feats = structural_features(nodes, edges)

    # 5. Признаки кластеров (потоково)
    bn_path = RAW / "background_nodes.csv"
    bn_cols = header(bn_path)
    bn_id = find_col(bn_cols, "clId", position=0)
    print("Читаю признаки кластеров (background_nodes.csv)...")
    bn = stream_filter(bn_path, [bn_id], set(nodes["cl_id"]), args.limit_rows)
    bn = bn.rename(columns={bn_id: "cl_id"}).drop_duplicates("cl_id")
    node_feat_cols = [c for c in bn.columns if c != "cl_id"]
    node_level = nodes.merge(bn, on="cl_id", how="left")
    feats = feats.join(aggregate(node_level, "cc_id", node_feat_cols, "node"), how="left")
    coverage = len(bn) / max(nodes["cl_id"].nunique(), 1)
    print(f"Найдены признаки для {len(bn):,} кластеров из {nodes['cl_id'].nunique():,} ({coverage:.0%})")
    if coverage < 0.95:
        print("  ВНИМАНИЕ: у части кластеров нет признаков. Если запускали с --limit-rows, "
              "для итоговых метрик запустите без этого флага.")

    # 6. Окружение подграфа во всей сети
    if not args.no_neighborhood:
        be_path = RAW / "background_edges.csv"
        be_cols = header(be_path)
        be_src = find_col(be_cols, "clId1", position=0)
        be_dst = find_col(be_cols, "clId2", position=1)
        print("Считаю окружение кластеров во всей сети (background_edges.csv, 2 колонки)...")
        deg = stream_degrees(be_path, be_src, be_dst, set(nodes["cl_id"]), args.limit_rows)
        node_deg = nodes.merge(deg, left_on="cl_id", right_index=True, how="left").fillna(
            {"out_deg": 0, "in_deg": 0, "out_ext": 0, "in_ext": 0})
        cols = ["out_deg", "in_deg", "out_ext", "in_ext"]
        nb = node_deg.groupby("cc_id")[cols].agg(["sum", "max"])
        nb.columns = [f"bg_{c}_{s}" for c, s in nb.columns]
        total = nb["bg_out_deg_sum"] + nb["bg_in_deg_sum"]
        nb["bg_ext_share"] = np.where(total > 0, (nb["bg_out_ext_sum"] + nb["bg_in_ext_sum"]) / total, 0.0)
        feats = feats.join(nb, how="left")
        print(f"Окружение найдено для {int((deg.sum(axis=1) > 0).sum()):,} кластеров из {nodes['cl_id'].nunique():,}")

    # 7. Признаки переводов (по желанию)
    if args.edge_features:
        be_path = RAW / "background_edges.csv"
        be_cols = header(be_path)
        be_src = find_col(be_cols, "clId1", position=0)
        be_dst = find_col(be_cols, "clId2", position=1)
        print("Читаю признаки переводов (background_edges.csv) — это надолго...")
        be = stream_filter(be_path, [be_src, be_dst], set(nodes["cl_id"]), args.limit_rows, both=True)
        be = be.rename(columns={be_src: "src", be_dst: "dst"})
        edge_feat_cols = [c for c in be.columns if c not in ("src", "dst")]
        edge_level = edges.merge(be, on=["src", "dst"], how="inner")
        feats = feats.join(aggregate(edge_level, "cc_id", edge_feat_cols, "edge"), how="left")
        print(f"Признаки найдены для {len(edge_level):,} переводов из {len(edges):,}")

    # 8. Таблица подграфов + разбиение
    df = cc.merge(feats, left_on="cc_id", right_index=True, how="inner")
    feature_cols = [c for c in df.columns if c not in ("cc_id", "label")]
    df = pd.concat([df[["cc_id", "label"]], df[feature_cols].astype("float32")], axis=1)

    idx = np.arange(len(df))
    train_idx, rest = train_test_split(idx, test_size=0.2, stratify=df["label"], random_state=42)
    val_idx, test_idx = train_test_split(rest, test_size=0.5, stratify=df["label"].iloc[rest], random_state=42)
    df["split"] = "train"
    df.iloc[val_idx, df.columns.get_loc("split")] = "val"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"

    print("\nРазбиение (стратифицированное, как 80/10/10 у авторов датасета):")
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        print(f"  {name:6s} {len(part):>8,} подграфов, подозрительных {part['label'].mean() * 100:.2f}%")

    df.to_parquet(PROCESSED / "subgraphs.parquet", index=False)
    nodes[nodes["cc_id"].isin(df["cc_id"])].to_parquet(PROCESSED / "membership.parquet", index=False)
    edges[edges["cc_id"].isin(df["cc_id"])][["cc_id", "src", "dst"]].to_parquet(
        PROCESSED / "sub_edges.parquet", index=False)
    (PROCESSED / "feature_cols.json").write_text(json.dumps(
        {"features": feature_cols, "edge_features": bool(args.edge_features)}, ensure_ascii=False, indent=2))

    print(f"\nПризнаков для модели: {len(feature_cols)}")
    print(f"Готово за {time.time() - started:.0f} с: файлы в {PROCESSED}")


if __name__ == "__main__":
    main()
