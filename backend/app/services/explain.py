"""
Понятные объяснения скора: признаки модели -> фразы на русском.

Признаки Elliptic2 обезличены (feat#1, feat#2, ...), поэтому для них мы пишем номер и показываем,
насколько значение необычно по сравнению с законными схемами. Структурные признаки подграфа считаем сами — их можно объяснить словами.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right

STRUCT = {
    "n_nodes": "Кластеров в схеме",
    "n_edges": "Переводов между кластерами",
    "max_out_degree": "Наибольшее число получателей у одного кластера",
    "max_in_degree": "Наибольшее число отправителей у одного кластера",
    "n_sources": "Кластеров-источников средств",
    "n_sinks": "Конечных получателей средств",
    "density": "Плотность связей",
    "edges_per_node": "Переводов на один кластер",
    "is_chain": "Переводы идут цепочкой",
    "fan_out_ratio": "Доля схемы, получившая средства от одного кластера",
    "fan_in_ratio": "Доля схемы, отправившая средства одному кластеру",
    # Окружение схемы во всей сети Bitcoin
    "bg_out_deg_sum": "Исходящих связей кластеров схемы во всей сети",
    "bg_out_deg_max": "Наибольшее число исходящих связей кластера во всей сети",
    "bg_in_deg_sum": "Входящих связей кластеров схемы во всей сети",
    "bg_in_deg_max": "Наибольшее число входящих связей кластера во всей сети",
    "bg_out_ext_sum": "Переводов из схемы наружу",
    "bg_out_ext_max": "Наибольшее число переводов наружу у одного кластера",
    "bg_in_ext_sum": "Переводов в схему извне",
    "bg_in_ext_max": "Наибольшее число переводов извне у одного кластера",
    "bg_ext_share": "Доля внешних связей схемы",
}

# Что может означать рост признака (подсказка для комплаенса)
HINTS = {
    "max_out_degree": "похоже на дробление суммы между многими получателями",
    "fan_out_ratio": "похоже на дробление суммы между многими получателями",
    "max_in_degree": "похоже на сбор средств из многих источников",
    "fan_in_ratio": "похоже на сбор средств из многих источников",
    "is_chain": "похоже на «пилинг» — последовательную передачу через цепочку",
    "n_nodes": "в схеме участвует много кластеров",
    "bg_out_ext_sum": "средства расходятся из схемы во многие стороны",
    "bg_out_ext_max": "один кластер переводит средства многим внешним получателям",
    "bg_in_ext_sum": "средства стекаются в схему из многих внешних источников",
    "bg_in_ext_max": "один кластер собирает средства из многих внешних источников",
}

STAT = {"mean": "в среднем по схеме", "max": "наибольшая в схеме", "min": "наименьшая в схеме"}
LEVEL_RU = {"low": "Низкий риск", "medium": "Средний риск", "high": "Высокий риск"}

# Пояснение для интерфейса и отчёта: почему у части характеристик только номера
HIDDEN_NOTE = ("Названия характеристик кластеров и переводов скрыты владельцем датасета Elliptic "
               "ради конфиденциальности, поэтому они пронумерованы. Если характеристика сильно связана "
               "с показателем, который мы считаем сами по сети Bitcoin, это указано («связана с …»): "
               "это статистическая связь, а не расшифровка. "
               "Процент показывает, насколько значение необычно по сравнению с законными схемами.")

TYPOLOGY_NOTE = ("Форма схемы описывает, как движутся средства, но сама по себе риск почти не определяет: "
                 "доля подозрительных у всех типов 1,5–2,7%. Скор рассчитан моделью по признакам кластеров и переводов.")


def feature_label(name: str) -> str:
    if name in STRUCT:
        return STRUCT[name]
    parts = name.split("_")
    if len(parts) == 3 and parts[0] in ("node", "edge") and parts[2] in STAT:
        kind = "кластеров" if parts[0] == "node" else "переводов"
        return f"Скрытая характеристика {kind} №{parts[1]}, {STAT[parts[2]]}"
    return name


def is_hidden(name: str) -> bool:
    return name.startswith(("node_", "edge_"))


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def fmt(value) -> str:
    """Число без научной записи, с запятой."""
    if _missing(value):
        return "нет данных"
    v = float(value)
    if v == int(v) and abs(v) < 1e9:
        return f"{int(v):,}".replace(",", " ")
    if abs(v) >= 100:
        s = f"{v:,.0f}".replace(",", " ")
    else:
        decimals = min(max(2, -math.floor(math.log10(abs(v))) + 2), 8)
        s = f"{v:.{decimals}f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


def _times(n: float) -> str:
    """в 2 раза, в 5 раз, в 23 раза, в 2,5 раза."""
    if n < 10:
        s = f"{n:.1f}".rstrip("0").rstrip(".").replace(".", ",")
        if "," in s:
            return f"{s} раза"
        k = int(s)
    else:
        k = int(round(n))
        s = str(k)
    return f"{s} раза" if k % 10 in (2, 3, 4) and k % 100 not in (12, 13, 14) else f"{s} раз"


def comparison(value, typical) -> str:
    if _missing(value) or _missing(typical):
        return ""
    v, t = float(value), float(typical)
    if t == 0:
        return "выше типичного" if v > 0 else ("ниже типичного" if v < 0 else "как у типичного")
    if v == 0:
        return "ниже типичного"
    ratio = v / t
    if 0.8 <= ratio <= 1.25:
        return "на уровне типичного"
    return f"в {_times(ratio)} выше" if ratio > 1 else f"в {_times(1 / ratio)} ниже"


def _pct(x: float) -> str:
    return f"{min(99, max(1, round(x * 100)))}%"


def unusualness(value, dist: dict | None) -> str:
    """«больше, чем у 96% законных схем» / «меньше, чем у 80% законных схем» / «как у большинства законных схем»."""
    if _missing(value) or not dist or not dist.get("licit_q"):
        return ""
    q = dist["licit_q"]
    n = len(q) - 1
    v = float(value)
    below = bisect_left(q, v) / n        # доля законных схем со значением меньше
    above = 1 - bisect_right(q, v) / n   # доля законных схем со значением больше
    if max(below, above) < 0.5:
        return "как у большинства законных схем"
    if below >= above:
        return f"больше, чем у {_pct(below)} законных схем"
    return f"меньше, чем у {_pct(above)} законных схем"


def _reason(name, value, typ, contribution, dist=None) -> dict:
    label = feature_label(name)
    usual = unusualness(value, dist)
    if usual:
        text = f"{label}: {fmt(value)} — {usual}"
        lm, sm = dist.get("licit_median"), dist.get("susp_median")
        if not _missing(lm) and not _missing(sm):
            text += f". Обычно у законных схем {fmt(lm)}, у подозрительных {fmt(sm)}"
            # Влияние против «наивного» направления: модель учитывает признак в сочетании с другими
            if (contribution > 0 and float(sm) <= float(lm)) or (contribution < 0 and float(sm) >= float(lm)):
                text += ". Влияет в сочетании с другими признаками"
    else:  # распределений нет — сравнение с типичным значением
        cmp = comparison(value, typ)
        text = f"{label}: {fmt(value)} (у типичной законной схемы {fmt(typ)})" + (f" — {cmp}" if cmp else "")
    # Подсказка уместна, только когда признак повышает риск И сам выше типичного
    above = not _missing(value) and (_missing(typ) or float(value) > float(typ))
    hint = HINTS.get(name) if contribution > 0 and above else None
    return {
        "feature": name,
        "label": label,
        "hidden": is_hidden(name),
        "value": None if _missing(value) else float(value),
        "typical": None if _missing(typ) else float(typ),
        "impact": round(float(contribution), 4),
        "direction": "up" if contribution > 0 else "down",
        "hint": hint,
        "usual": usual or None,
        "text": text,
    }


# Группы факторов: из чего складывается оценка
GROUPS = {
    "hidden": "Скрытые характеристики кластеров",
    "network": "Связи кластеров с остальной сетью",
    "structure": "Форма схемы",
}
# Понятный «двойник» скрытого признака (см. ml/proxy_features.py): направление связи и подсказка
PROXY = {
    "bg_in_deg_max": "in", "bg_in_deg_sum": "in", "bg_in_ext_sum": "in", "bg_in_ext_max": "in",
    "bg_out_deg_max": "out", "bg_out_deg_sum": "out", "bg_out_ext_sum": "out", "bg_out_ext_max": "out",
}
PROXY_TEXT = {
    "in": "связана с тем, сколько разных источников присылают средства кластерам схемы",
    "out": "связана с тем, скольким получателям кластеры схемы отправляют средства",
}
PROXY_HINT = {
    "in": "похоже на сбор средств из многих источников",
    "out": "похоже на раздачу средств многим получателям",
}


def group_of(name: str, proxies: dict) -> str:
    if name.startswith("bg_") or name in proxies:
        return "network"
    if name in STRUCT:
        return "structure"
    return "hidden"


def _strength(share: float) -> str:
    return "сильное" if share >= 0.15 else "заметное" if share >= 0.05 else "слабое"


def _merge_key(name: str) -> str:
    """node_28_max и node_28_mean — одна характеристика №28."""
    parts = name.split("_")
    if len(parts) == 3 and parts[0] in ("node", "edge") and parts[2] in STAT:
        return f"{parts[0]}_{parts[1]}"
    return name


def reasons(features: dict, contributions: dict, typical: dict, dists: dict | None = None,
            proxies: dict | None = None, n_risk: int = 4, n_mitigating: int = 3) -> dict:
    """Факторы риска и смягчающие факторы, объединённые по характеристикам и сгруппированные по смыслу."""
    dists, proxies = dists or {}, proxies or {}
    total = sum(abs(c) for c in contributions.values()) or 1.0

    # 1. Объединяем статистики одной скрытой характеристики (max/mean/min) в один фактор
    merged: dict[str, dict] = {}
    for name, c in contributions.items():
        key = _merge_key(name)
        m = merged.setdefault(key, {"impact": 0.0, "parts": []})
        m["impact"] += c
        m["parts"].append((name, c))

    # 2. Доли групп
    groups = {g: 0.0 for g in GROUPS}
    for name, c in contributions.items():
        groups[group_of(name, proxies)] += abs(c)
    group_list = [{"key": g, "name": GROUPS[g], "share": round(v / total, 4)}
                  for g, v in sorted(groups.items(), key=lambda kv: -kv[1])]

    def build(key, m):
        impact = m["impact"]
        # показываем статистику, которая сильнее всего тянет в ту же сторону
        same = [pc for pc in m["parts"] if (pc[1] > 0) == (impact > 0)] or m["parts"]
        name = max(same, key=lambda pc: abs(pc[1]))[0]
        r = _reason(name, features.get(name), typical.get(name), impact, dists.get(name))
        group = group_of(name, proxies)
        share = abs(impact) / total
        proxy = proxies.get(name) or proxies.get(f"{key}_max") or proxies.get(f"{key}_mean")
        direction = PROXY.get(proxy["proxy"]) if proxy else PROXY.get(name)
        if proxy and direction:
            r["proxy_note"] = f"{PROXY_TEXT[direction]} (корреляция {fmt(proxy['r'])})"
            dist = dists.get(name) or {}
            lm = dist.get("licit_median")
            high = r["value"] is not None and (_missing(lm) or r["value"] > float(lm))
            counter = "в сочетании с другими" in r["text"]  # направление не очевидно — подсказку не даём
            if impact > 0 and high and not counter and not r.get("hint"):
                r["hint"] = PROXY_HINT[direction]
        r.update({"group": group, "group_name": GROUPS[group], "share": round(share, 4),
                  "strength": _strength(share), "impact": round(impact, 4)})
        return r

    ranked = sorted(merged.items(), key=lambda kv: abs(kv[1]["impact"]), reverse=True)
    up = [build(k, m) for k, m in ranked if m["impact"] > 0][:n_risk]
    down = [build(k, m) for k, m in ranked if m["impact"] < 0][:n_mitigating]
    return {"risk": up, "mitigating": down, "groups": group_list}


def groups_sentence(groups: list[dict]) -> str:
    """«Оценка на 78% основана на скрытых характеристиках кластеров, на 18% — на связях…»"""
    parts = [g for g in groups if g["share"] >= 0.005]
    if not parts:
        return ""
    words = [f"{round(g['share'] * 100)}% — {g['name'].lower()}" for g in parts]
    return "Из чего складывается оценка: " + "; ".join(words) + "."


def risk_level(proba: float, levels: dict) -> tuple[int, str]:
    """Вероятность -> скор 0–100: порог среднего риска = 40, порог высокого = 70."""
    med, high = levels["medium"], levels["high"]
    if proba >= high:
        score, level = 70 + 30 * (proba - high) / max(1 - high, 1e-9), "high"
    elif proba >= med:
        score, level = 40 + 30 * (proba - med) / max(high - med, 1e-9), "medium"
    else:
        score, level = 40 * proba / max(med, 1e-9), "low"
    # 100 из 100 читается как «доказано»; модель даёт вероятность, поэтому потолок 99
    return int(round(min(max(score, 0), 99))), level


RECOMMENDATION = {
    "low": "Операции клиента можно продолжать в обычном режиме; результат проверки сохранён в журнале.",
    "medium": "Рекомендуется запросить у клиента документы, подтверждающие источник средств, "
              "и установить временные лимиты до завершения проверки.",
    "high": "Рекомендуется приостановить операцию, провести углублённую проверку клиента и рассмотреть "
            "направление сообщения о подозрительной операции (СПО) в АФМ.",
}


def conclusion_template(level: str, score: int, why: dict, blacklisted: bool = False) -> str:
    """Заключение для комплаенс-офицера без LLM — работает без интернета."""
    if blacklisted:
        return ("Кластер находится в чёрном списке. " + RECOMMENDATION["high"]
                + " Окончательное решение принимает сотрудник банка.")
    head = f"Схема переводов оценена как «{LEVEL_RU[level].lower()}» ({score} из 100)."
    patterns = [p for p in why.get("patterns", []) if p["code"] not in ("simple", "mixed")]
    if patterns:
        head += " Структура: " + ", ".join(p["name"].lower() for p in patterns) + "."
    risk = why.get("risk", [])
    if risk:
        main = risk[0]
        detail = f" Основной фактор: {main['label'][0].lower() + main['label'][1:]}"
        if main.get("usual"):
            detail += f" ({main['usual']})"
        if main.get("hint"):
            detail += f" — {main['hint']}"
        detail += "."
        if len(risk) > 1:
            detail += f" Дополнительно влияют: {'; '.join(r['label'][0].lower() + r['label'][1:] for r in risk[1:3])}."
    else:
        detail = " Существенных факторов риска не выявлено."
    drivers = groups_sentence(why.get("groups") or [])
    if drivers:
        detail += " " + drivers
    return head + detail + " " + RECOMMENDATION[level] + " Окончательное решение принимает сотрудник банка."
