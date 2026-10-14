"""PDF-отчёт о проверке для комплаенса (reportlab, шрифт DejaVu Sans с кириллицей)."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .explain import HIDDEN_NOTE, LEVEL_RU, TYPOLOGY_NOTE, groups_sentence

FONTS = Path(__file__).resolve().parents[1] / "assets" / "fonts"
pdfmetrics.registerFont(TTFont("DejaVu", str(FONTS / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONTS / "DejaVuSans-Bold.ttf")))

NAVY = colors.HexColor("#0F1B2D")
MUTED = colors.HexColor("#5A6575")
LEVEL_COLOR = {"low": "#2F6F9F", "medium": "#B5761A", "high": "#B4380B"}


def _style(name, size=10, bold=False, color=NAVY, leading=None, space=0):
    return ParagraphStyle(name, fontName="DejaVu-Bold" if bold else "DejaVu", fontSize=size,
                          leading=leading or size * 1.35, textColor=color, alignment=TA_LEFT, spaceAfter=space)


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_report(check: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"Traceon — проверка №{check['id']}", author="Traceon")
    h1, h2 = _style("h1", 17, True, space=2), _style("h2", 12, True, space=4)
    body, small = _style("body", 10, space=3), _style("small", 8.5, color=MUTED)

    created = check["created_at"]
    created = created.strftime("%d.%m.%Y %H:%M") if isinstance(created, datetime) else str(created)
    level = check["risk_level"]
    story = [
        Paragraph("Traceon · Отчёт о проверке источника средств", h1),
        Paragraph(f"Проверка №{check['id']} от {created}", small),
        Spacer(1, 8 * mm),
    ]

    def _scheme(c):
        why_ = c.get("reasons") or {}
        if why_.get("scheme"):
            return why_["scheme"][0].upper() + why_["scheme"][1:]
        pats = why_.get("patterns") or []
        return pats[0]["name"] if pats else "—"

    summary = [
        ["Запрос", _esc(check["query"])],
        ["Подграф переводов", _esc(check.get("cc_id") or "—")],
        ["Риск-скор", f"{check['score']} из 100"],
        ["Уровень риска", LEVEL_RU.get(level, level)],
        ["Тип схемы", _esc(_scheme(check))],
        ["Чёрный список", "да" if check.get("blacklisted") else "нет"],
        ["Владение адресом", (f"подтверждено подписью ({_esc(check['signed_address'])})"
                              if check.get("signature_verified") else "не подтверждено")],
    ]
    t = Table([[Paragraph(a, _style("k", 10, color=MUTED)), Paragraph(b, body)] for a, b in summary],
              colWidths=[48 * mm, 126 * mm])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#DAD7CE")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TEXTCOLOR", (1, 3), (1, 3), colors.HexColor(LEVEL_COLOR.get(level, "#0F1B2D"))),
    ]))
    story += [t, Spacer(1, 7 * mm), Paragraph("Заключение", h2), Paragraph(_esc(check.get("conclusion") or ""), body)]
    if check.get("conclusion_source") == "llm":
        story.append(Paragraph("Текст сформулирован языковой моделью на основе результатов ML-модели.", small))

    why = check.get("reasons") or {}
    patterns = why.get("patterns") or []
    if patterns:
        story += [Spacer(1, 5 * mm), Paragraph("Тип схемы", h2)]
        for p in patterns:
            line = f"<b>{_esc(p['name'])}.</b> {_esc(p['description'])}"
            if p.get("suspicious_share") is not None:
                line += (f" Среди таких схем в данных подозрительных {p['suspicious_share'] * 100:.1f}%".replace(".", ",")
                         + f" (в среднем {p['base_rate'] * 100:.1f}%).".replace(".", ",", 1))
            story.append(Paragraph(line, body))
        story.append(Paragraph(TYPOLOGY_NOTE, small))
    drivers = groups_sentence(why.get("groups") or [])
    if drivers:
        story += [Spacer(1, 5 * mm), Paragraph(_esc(drivers), body)]
    for title, key, mark in (("Факторы риска", "risk", "▲"), ("Смягчающие факторы", "mitigating", "▼")):
        items = why.get(key) or []
        if items:
            story += [Spacer(1, 5 * mm), Paragraph(title, h2)]
            for r in items:
                text = _esc(r["text"])
                if r.get("proxy_note"):
                    text += f". <font color='#2F7F86'>{_esc(r['proxy_note'][0].upper() + r['proxy_note'][1:])}</font>"
                if r.get("hint"):
                    text += f" — <i>{_esc(r['hint'])}</i>"
                if r.get("group_name"):
                    text += (f"<br/><font size='8.5' color='#5B6676'>{_esc(r['group_name'])} · влияние "
                             f"{_esc(r['strength'])}, {round(r['share'] * 100)}% оценки</font>")
                story.append(Paragraph(f"{mark} {text}", body))

    if any(r.get("hidden") for k in ("risk", "mitigating") for r in (why.get(k) or [])):
        story += [Spacer(1, 3 * mm), Paragraph(HIDDEN_NOTE, small)]

    story += [
        Spacer(1, 10 * mm),
        Paragraph("Оценка сформирована моделью XGBoost, обученной на открытом датасете Elliptic2. "
                  "Скор отражает сходство схемы переводов с размеченными случаями отмывания и не является "
                  "доказательством. Окончательное решение принимает уполномоченный сотрудник банка "
                  "в соответствии с внутренними правилами ПОД/ФТ.", small),
    ]
    doc.build(story)
    return buf.getvalue()
