"""Загрузка обученной модели и расчёт скора с SHAP-вкладами (встроенными в XGBoost)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


class RiskModel:
    def __init__(self, models_dir: Path):
        self.ready = False
        self.error = None
        try:
            meta = json.loads((models_dir / "model_meta.json").read_text())
            self.features: list[str] = meta["features"]
            self.levels: dict = meta["levels"]
            self.iterations = (0, int(meta.get("best_iteration", 0)) + 1)
            self.meta = meta
            self.typical: dict = json.loads((models_dir / "typical.json").read_text())
            dist_path = models_dir / "distributions.json"
            self.dists: dict = json.loads(dist_path.read_text()) if dist_path.exists() else {}
            typ_path = models_dir / "typology_stats.json"
            self.typology_stats: dict = json.loads(typ_path.read_text()) if typ_path.exists() else {}
            prox_path = models_dir / "feature_proxies.json"
            self.proxies: dict = json.loads(prox_path.read_text()) if prox_path.exists() else {}
            demo_path = models_dir / "demo.json"
            self.demo: list = json.loads(demo_path.read_text()) if demo_path.exists() else []
            self.booster = xgb.Booster()
            self.booster.load_model(str(models_dir / "xgb_model.json"))
            self.ready = True
        except FileNotFoundError as exc:
            self.error = f"Модель не обучена: нет файла {Path(exc.filename).name}. Запустите ml/train.py."
        except Exception as exc:  # повреждённые файлы и т.п.
            self.error = f"Не удалось загрузить модель: {exc}"

    def predict(self, features: dict) -> tuple[float, dict]:
        """Вероятность «подозрительный» и вклад каждого признака (в логарифме шансов)."""
        row = pd.DataFrame([[features.get(f) for f in self.features]], columns=self.features, dtype=float)
        dm = xgb.DMatrix(row, missing=np.nan)
        proba = float(self.booster.predict(dm, iteration_range=self.iterations)[0])
        contribs = self.booster.predict(dm, pred_contribs=True, iteration_range=self.iterations)[0, :-1]
        return proba, dict(zip(self.features, map(float, contribs)))
