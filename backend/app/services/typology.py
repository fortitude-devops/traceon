"""
Распознавание типовых схем отмывания по структуре переводов.

Работает на реальных переводах внутри схемы (не на скрытых признаках), поэтому
каждый вывод можно объяснить словами и проверить глазами на графе.
Тип схемы — это описание формы движения денег, а не приговор: риск оценивает модель.
На Elliptic2 доля подозрительных почти одинакова у всех типов (1,5–2,7% при среднем 2,3%),
поэтому описания нейтральны, а насколько тип рискован, показывает статистика рядом.
"""
from __future__ import annotations

from collections import defaultdict

# Порядок = порядок показа (самые характерные для отмывания — выше)
PATTERNS = {
    "cycle": "Цикл",
    "gather_scatter": "Сбор и раздача",
    "fan_out": "Дробление суммы",
    "fan_in": "Сбор средств",
    "chain": "Цепочка переводов («пилинг»)",
    "simple": "Простой перевод",
    "mixed": "Смешанная схема",
}

MIN_FAN = 3  # от скольких получателей/отправителей считаем веер
MAX_NODES = 200  # сколько участников схемы передаём для подсветки на графе


def _clusters(n: int) -> str:
    """3 кластера, 5 кластеров, 21 кластер."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} кластер"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} кластера"
    return f"{n} кластеров"


def _short(cl_id: str) -> str:
    return cl_id if len(cl_id) <= 10 else cl_id[:8] + "…"


def _has_cycle(nodes, out_edges) -> list[str] | None:
    """Ищет направленный цикл (деньги возвращаются туда, откуда ушли). Возвращает узлы цикла."""
    color = {n: 0 for n in nodes}  # 0 — не посещён, 1 — в стеке, 2 — готов
    parent = {}
    for start in nodes:
        if color[start]:
            continue
        stack = [(start, iter(out_edges.get(start, ())))]
        color[start] = 1
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                color[node] = 2
                stack.pop()
                continue
            if nxt == node:
                continue
            if color.get(nxt, 0) == 0:
                color[nxt] = 1
                parent[nxt] = node
                stack.append((nxt, iter(out_edges.get(nxt, ()))))
            elif color.get(nxt) == 1:
                cycle, cur = [nxt], node
                while cur != nxt and cur in parent:
                    cycle.append(cur)
                    cur = parent[cur]
                return list(reversed(cycle))
    return None


def detect(nodes: list[str], edges: list[tuple[str, str]]) -> list[dict]:
    """Список найденных типов схемы с пояснением на русском."""
    edges = list({(s, d) for s, d in edges if s != d})
    out_edges, in_edges = defaultdict(set), defaultdict(set)
    for s, d in edges:
        out_edges[s].add(d)
        in_edges[d].add(s)
    found = []

    cycle = _has_cycle(nodes, out_edges)
    if cycle:
        found.append({"code": "cycle", "name": PATTERNS["cycle"],
                      "description": f"Средства проходят по кругу через {_clusters(len(cycle))} и возвращаются к отправителю.",
                      "hub": None, "nodes": cycle[:MAX_NODES]})

    hubs_gs = [n for n in nodes if len(in_edges[n]) >= 2 and len(out_edges[n]) >= 2]
    if hubs_gs:
        h = max(hubs_gs, key=lambda n: len(in_edges[n]) + len(out_edges[n]))
        found.append({"code": "gather_scatter", "name": PATTERNS["gather_scatter"],
                      "description": f"Кластер {_short(h)} собирает средства от {len(in_edges[h])} отправителей "
                                     f"и раздаёт {len(out_edges[h])} получателям.",
                      "hub": h, "nodes": ([h] + sorted(in_edges[h] | out_edges[h]))[:MAX_NODES]})

    fan_out = max(nodes, key=lambda n: len(out_edges[n]), default=None)
    if fan_out is not None and len(out_edges[fan_out]) >= MIN_FAN:
        found.append({"code": "fan_out", "name": PATTERNS["fan_out"],
                      "description": f"Кластер {_short(fan_out)} раздаёт средства {len(out_edges[fan_out])} "
                                     f"получателям.",
                      "hub": fan_out, "nodes": ([fan_out] + sorted(out_edges[fan_out]))[:MAX_NODES]})

    fan_in = max(nodes, key=lambda n: len(in_edges[n]), default=None)
    if fan_in is not None and len(in_edges[fan_in]) >= MIN_FAN:
        found.append({"code": "fan_in", "name": PATTERNS["fan_in"],
                      "description": f"Кластер {_short(fan_in)} получает средства от {len(in_edges[fan_in])} "
                                     f"отправителей.",
                      "hub": fan_in, "nodes": ([fan_in] + sorted(in_edges[fan_in]))[:MAX_NODES]})

    is_path = (len(nodes) >= 3 and len(edges) == len(nodes) - 1 and not cycle
               and all(len(out_edges[n]) <= 1 and len(in_edges[n]) <= 1 for n in nodes))
    if is_path:
        found.append({"code": "chain", "name": PATTERNS["chain"],
                      "description": f"Средства последовательно проходят через {_clusters(len(nodes))} один за другим.",
                      "hub": None, "nodes": list(nodes)[:MAX_NODES]})

    if not found:
        if len(nodes) == 2 and len(edges) == 1:
            found.append({"code": "simple", "name": PATTERNS["simple"],
                          "description": "Один перевод между двумя кластерами."})
        else:
            found.append({"code": "mixed", "name": PATTERNS["mixed"],
                          "description": f"{len(nodes)} кластеров и {len(edges)} переводов без выраженного "
                                         f"типового рисунка."})
    return found


def with_stats(found: list[dict], stats: dict | None) -> list[dict]:
    """Добавляет «среди таких схем в данных подозрительных X%»."""
    if not stats:
        return found
    base = stats.get("_base_rate")
    for p in found:
        s = stats.get(p["code"])
        if s and s.get("count"):
            p["count"] = s["count"]
            p["suspicious_share"] = s["suspicious_share"]
            p["base_rate"] = base
    return found


def headline(found: list[dict]) -> str:
    """Одна строка для шапки: «дробление суммы — кластер 12ab… раздаёт средства 7 получателям»."""
    if not found:
        return ""
    p = found[0]
    d = p["description"].rstrip(".")
    return f"{p['name'][0].lower() + p['name'][1:]} — {d[0].lower() + d[1:]}"
