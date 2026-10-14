"""Общие пути и утилиты для скриптов ML."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "elliptic2"          # сюда кладутся CSV из архива Elliptic2
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "ml" / "reports"

# Опубликованные результаты авторов Elliptic2 (тестовая выборка, без признаков узлов)
# Источник: Bellei et al., "The Shape of Money Laundering", 2024, arXiv:2404.19109, табл. 2
PAPER_BASELINES = {
    "GLASS": {"pr_auc": 0.208, "roc_auc": 0.889},
    "Sub2Vec": {"pr_auc": 0.022, "roc_auc": 0.496},
    "GNN-Seg": {"pr_auc": 0.026, "roc_auc": 0.537},
}


def norm(name: str) -> str:
    """Нормализованное имя колонки для поиска: 'clId 1' -> 'clid1'."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_col(columns, *candidates, position=None):
    """Находит колонку по возможным именам (без учёта регистра и символов), иначе по позиции."""
    lookup = {norm(c): c for c in columns}
    for cand in candidates:
        if norm(cand) in lookup:
            return lookup[norm(cand)]
    if position is not None and len(columns) > position:
        return columns[position]
    raise KeyError(f"Не найдена колонка {candidates} среди {list(columns)[:12]}")


def feature_suffix(name: str) -> str:
    """'feat#12' -> '12', 'Feature 3' -> '3', иначе безопасное имя."""
    digits = "".join(ch for ch in name if ch.isdigit())
    return digits if digits else norm(name)
