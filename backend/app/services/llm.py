"""
Текстовое заключение для комплаенс-офицера.

Если задан NVIDIA_API_KEY, заключение пишет LLM (NVIDIA API). LLM НЕ принимает решение:
она только пересказывает скор и причины, которые выдала модель. Без ключа, без интернета
или при ошибке используется шаблон — демо работает всегда.
"""
from __future__ import annotations

import httpx

from .. import config
from .explain import LEVEL_RU, conclusion_template

SYSTEM_PROMPT = (
    "Ты помощник комплаенс-офицера казахстанского банка. Тебе дают результат ML-модели, "
    "оценившей подграф криптовалютных переводов по риску отмывания денег. Напиши заключение "
    "на русском языке в 3–4 предложениях: уровень риска, главные причины простыми словами, "
    "тип схемы, если он указан, рекомендуемое действие банка (продолжить операции / запросить документы о происхождении средств / "
    "приостановить операцию и рассмотреть СПО в АФМ). Используй только переданные факты, ничего "
    "не придумывай, не меняй скор и уровень риска. Закончи фразой, что окончательное решение "
    "принимает сотрудник банка."
)


def _prompt(score: int, level: str, why: dict, blacklisted: bool) -> str:
    lines = [f"Скор: {score} из 100. Уровень: {LEVEL_RU[level]}.",
             f"Кластер в чёрном списке: {'да' if blacklisted else 'нет'}.", "Факторы риска:"]
    lines += [f"- {r['text']}" + (f"; {r['proxy_note']}" if r.get("proxy_note") else "")
              + (f" ({r['hint']})" if r.get("hint") else "") for r in why.get("risk", [])] or ["- нет"]
    lines.append("Тип схемы по структуре переводов:")
    lines += [f"- {p['name']}: {p['description']}" for p in why.get("patterns", [])] or ["- не определён"]
    lines.append("Смягчающие факторы:")
    lines += [f"- {r['text']}" for r in why.get("mitigating", [])] or ["- нет"]
    return "\n".join(lines)


def conclude(score: int, level: str, why: dict, blacklisted: bool = False) -> tuple[str, str]:
    """Возвращает (текст, источник): источник 'llm' или 'template'."""
    template = conclusion_template(level, score, why, blacklisted)
    if not config.NVIDIA_API_KEY:
        return template, "template"
    try:
        resp = httpx.post(
            config.NVIDIA_URL,
            headers={"Authorization": f"Bearer {config.NVIDIA_API_KEY}"},
            json={
                "model": config.NVIDIA_MODEL,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": _prompt(score, level, why, blacklisted)}],
                "temperature": 0.2,
                "max_tokens": 350,
            },
            timeout=config.LLM_TIMEOUT,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return (text, "llm") if text else (template, "template")
    except Exception:
        return template, "template"
