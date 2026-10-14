"""
Traceon — шаг 2: обучение модели на подграфах Elliptic2.

Что делает скрипт:
  1. Базовая модель — логистическая регрессия.
  2. Основная модель — XGBoost с учётом дисбаланса классов (~2% подозрительных).
  3. Пороги уровней риска по валидации:
       высокий — максимум F1, средний — верхние 10% подграфов по скору (бюджет ручных проверок).
  4. Метрики на тесте + сравнение с опубликованными результатами авторов датасета (GLASS).
  5. Важность признаков (SHAP-вклады, встроенные в XGBoost) и графики для презентации.
  6. Типичные значения законных подграфов (для объяснений) и демо-подграфы для показа.

Запуск из корня проекта:  python ml/train.py
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import MODELS, PAPER_BASELINES, PROCESSED, REPORTS, ROOT
from describe_features import build_distributions, build_typology_stats

sys.path.insert(0, str(ROOT / "backend"))
from app.services.explain import LEVEL_RU, reasons, risk_level  # noqa: E402

MEDIUM_ALERT_SHARE = 0.10   # средний+высокий риск = верхние 10% подграфов по скору (бюджет ручных проверок)


def best_f1_threshold(y, p) -> float:
    precision, recall, thr = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])


def budget_threshold(p, share) -> float:
    """Порог, выше которого оказывается заданная доля подграфов (бюджет ручных проверок)."""
    return float(np.quantile(p, 1 - share))


def metrics(y, p, threshold) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    feats = json.loads((PROCESSED / "feature_cols.json").read_text())["features"]
    df = pd.read_parquet(PROCESSED / "subgraphs.parquet")
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    part = {s: df[df["split"] == s].copy() for s in ("train", "val", "test")}
    X = {s: d[feats] for s, d in part.items()}
    y = {s: d["label"].values for s, d in part.items()}
    print(f"Подграфов: обучение {len(y['train']):,}, валидация {len(y['val']):,}, тест {len(y['test']):,}")

    results = {}

    print("Обучаю логистическую регрессию (базовый уровень)...")
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    lr.fit(X["train"].fillna(0), y["train"])
    lr_val = lr.predict_proba(X["val"].fillna(0))[:, 1]
    lr_test = lr.predict_proba(X["test"].fillna(0))[:, 1]
    results["logistic_regression"] = metrics(y["test"], lr_test, best_f1_threshold(y["val"], lr_val))

    print("Обучаю XGBoost...")
    pos_weight = float((y["train"] == 0).sum() / max((y["train"] == 1).sum(), 1))
    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.03, max_depth=6, min_child_weight=2,
        subsample=0.8, colsample_bytree=0.6, scale_pos_weight=pos_weight,
        tree_method="hist", eval_metric="aucpr", early_stopping_rounds=100,
        n_jobs=-1, random_state=42,
    )
    model.fit(X["train"], y["train"], eval_set=[(X["val"], y["val"])], verbose=False)
    best_it = int(model.best_iteration)
    print(f"  деревьев после ранней остановки: {best_it + 1}")

    p_val = model.predict_proba(X["val"])[:, 1]
    p_test = model.predict_proba(X["test"])[:, 1]
    high = best_f1_threshold(y["val"], p_val)
    medium = min(budget_threshold(p_val, MEDIUM_ALERT_SHARE), high * 0.9)
    levels = {"medium": round(medium, 6), "high": round(high, 6)}
    results["xgboost"] = metrics(y["test"], p_test, high)

    print("\nМетрики на тесте:")
    table = pd.DataFrame(results).T.drop(columns="confusion")
    print(table.to_string())
    print("\nОпубликованные результаты авторов Elliptic2 (тест, без признаков узлов):")
    for name, m in PAPER_BASELINES.items():
        print(f"  {name:8s} PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}")
    print("\nМатрица ошибок XGBoost:", results["xgboost"]["confusion"])

    # Уровни риска на тесте
    lvl = np.where(p_test >= high, "high", np.where(p_test >= medium, "medium", "low"))
    total_bad = max(int(y["test"].sum()), 1)
    level_stats = {}
    print("\nУровни риска на тесте:")
    base_rate = float(y["test"].mean())
    for name in ("low", "medium", "high"):
        mask = lvl == name
        n, bad = int(mask.sum()), int(y["test"][mask].sum())
        share = bad / n if n else 0.0
        level_stats[name] = {"count": n, "suspicious": bad,
                             "suspicious_share": round(share, 4),
                             "share_of_all_suspicious": round(bad / total_bad, 4),
                             "lift": round(share / base_rate, 2) if base_rate else 0.0}
        print(f"  {LEVEL_RU[name]:13s} {n:>7,} подграфов, подозрительных {share * 100:5.1f}% "
              f"(в {share / base_rate:.1f} раза больше среднего), {bad / total_bad * 100:.0f}% всех подозрительных")

    # Точность среди самых рискованных: сколько найдёт комплаенс, проверив верхние N% подграфов
    order = np.argsort(-p_test)
    top_k = {}
    print("\nЕсли проверять вручную самые рискованные подграфы:")
    for share in (0.01, 0.05, 0.10):
        k = max(1, int(round(len(order) * share)))
        hits = int(y["test"][order[:k]].sum())
        top_k[f"{int(share * 100)}%"] = {"checked": k, "found": hits,
                                         "precision": round(hits / k, 4),
                                         "recall": round(hits / total_bad, 4),
                                         "lift": round(hits / k / base_rate, 2) if base_rate else 0.0}
        print(f"  верхние {share * 100:>3.0f}% ({k:,} шт.): подозрительных {hits / k * 100:5.1f}% "
              f"(в {hits / k / base_rate:.1f} раза больше среднего), найдено {hits / total_bad * 100:.0f}% всех")

    # Важность признаков: SHAP-вклады из самого XGBoost
    booster = model.get_booster()
    sample = X["test"].sample(min(5000, len(X["test"])), random_state=42)
    contribs = booster.predict(xgb.DMatrix(sample, missing=np.nan), pred_contribs=True,
                               iteration_range=(0, best_it + 1))[:, :-1]
    importance = pd.Series(np.abs(contribs).mean(axis=0), index=feats).sort_values(ascending=False)
    print("\nТоп-10 признаков по SHAP:")
    print(importance.head(10).round(4).to_string())

    # Графики для презентации
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, p in (("XGBoost (Traceon)", p_test), ("Логистическая регрессия", lr_test)):
        pr, rc, _ = precision_recall_curve(y["test"], p)
        ax.plot(rc, pr, label=f"{name}: PR-AUC {average_precision_score(y['test'], p):.3f}")
    ax.axhline(PAPER_BASELINES["GLASS"]["pr_auc"], ls="--", c="grey", label="GLASS (авторы датасета): PR-AUC 0.208")
    ax.axhline(y["test"].mean(), ls=":", c="lightgrey", label=f"Случайное угадывание: {y['test'].mean():.3f}")
    ax.set_xlabel("Recall (доля найденных подозрительных)")
    ax.set_ylabel("Precision (точность)")
    ax.set_title("Precision–Recall на тесте Elliptic2")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORTS / "pr_curve.png", dpi=150)
    plt.close()
    try:
        import shap
        shap.summary_plot(contribs, sample, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(REPORTS / "shap_summary.png", dpi=150)
        plt.close()
    except Exception as exc:  # shap — только для графика
        print(f"(график SHAP пропущен: {exc})")

    # Типичные значения законных подграфов
    licit = X["train"][y["train"] == 0]
    typical = {f: (None if pd.isna(v) else float(v)) for f, v in licit.median().items()}
    dists = build_distributions(part["train"], feats)

    # Демо-подграфы: высокий, средний и низкий риск
    test = part["test"].assign(proba=p_test, level=lvl)
    mid = test[test["level"] == "medium"]
    picks = {
        "high": test[(test["label"] == 1) & (test["level"] == "high")].nlargest(1, "proba"),
        "medium": mid.loc[(mid["proba"] - (medium + high) / 2).abs().nsmallest(1).index],
        "low": test[(test["label"] == 0) & (test["level"] == "low") & (test["n_nodes"] >= 3)].nsmallest(1, "proba"),
    }
    members = pd.read_parquet(PROCESSED / "membership.parquet")
    demo = []
    for name, row in picks.items():
        if row.empty:
            print(f"\nНе нашёлся подграф для уровня «{LEVEL_RU[name]}»")
            continue
        r = row.iloc[0]
        values = {f: (None if pd.isna(r[f]) else float(r[f])) for f in feats}
        c = booster.predict(xgb.DMatrix(row[feats], missing=np.nan), pred_contribs=True,
                            iteration_range=(0, best_it + 1))[0, :-1]
        score, level = risk_level(float(r["proba"]), levels)
        why = reasons(values, dict(zip(feats, c)), typical, dists)
        cl = members.loc[members["cc_id"] == r["cc_id"], "cl_id"].iloc[0]
        print(f"\n=== {LEVEL_RU[level]} | скор {score}/100 | подграф {r['cc_id']} | кластер {cl}")
        print(f"    метка в датасете: {'подозрительный' if r['label'] == 1 else 'законный'}, "
              f"кластеров {int(r['n_nodes'])}, переводов {int(r['n_edges'])}")
        for x in why["risk"]:
            print(f"      ▲ {x['text']}")
        for x in why["mitigating"]:
            print(f"      ▼ {x['text']}")
        demo.append({"level": name, "cc_id": str(r["cc_id"]), "cl_id": str(cl)})

    # Сохранение
    model.save_model(MODELS / "xgb_model.json")
    meta = {
        "features": feats,
        "levels": levels,
        "best_iteration": best_it,
        "scale_pos_weight": pos_weight,
        "dataset": "Elliptic2",
        "metrics": {"test": results, "levels": level_stats, "top_k": top_k, "paper_baselines": PAPER_BASELINES,
                    "n_test": int(len(y["test"])), "suspicious_share_test": round(float(y["test"].mean()), 4)},
        "top_features": {k: round(float(v), 4) for k, v in importance.head(15).items()},
    }
    (MODELS / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (MODELS / "typical.json").write_text(json.dumps(typical, ensure_ascii=False, indent=2))
    (MODELS / "distributions.json").write_text(json.dumps(dists, ensure_ascii=False))
    (MODELS / "typology_stats.json").write_text(
        json.dumps(build_typology_stats(part["train"]), ensure_ascii=False, indent=2))
    (MODELS / "demo.json").write_text(json.dumps(demo, ensure_ascii=False, indent=2))
    (REPORTS / "metrics.json").write_text(json.dumps(meta["metrics"], ensure_ascii=False, indent=2))
    print(f"\nГотово: модель и файлы в {MODELS}, графики в {REPORTS}")


if __name__ == "__main__":
    main()
