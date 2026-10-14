"""HTTP API сервиса Traceon. Документация: http://localhost:8000/docs"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import text

from ..db import engine
from ..services.explain import HIDDEN_NOTE, LEVEL_RU, feature_label, reasons, risk_level
from ..services.llm import conclude
from ..services.report import build_report
from ..services.signature import verify_message
from ..services.typology import detect, with_stats, headline

router = APIRouter(prefix="/api")


class Ownership(BaseModel):
    address: str = Field(..., description="Bitcoin-адрес клиента")
    message: str = Field(..., description="Текст, который подписал клиент")
    signature: str = Field(..., description="Подпись в Base64 (BIP-137)")


class CheckRequest(BaseModel):
    query: str = Field(..., description="ID кластера (clId) или подграфа (ccId) из Elliptic2")
    ownership: Optional[Ownership] = Field(None, description="Подтверждение владения адресом (необязательно)")


class BlacklistItem(BaseModel):
    cl_id: str
    source: str = "внутренний список"
    reason: Optional[str] = None


def _model(request: Request):
    model = request.app.state.model
    if not model.ready:
        raise HTTPException(503, model.error)
    return model


@router.get("/health", summary="Состояние сервиса")
def health(request: Request):
    model = request.app.state.model
    db_ok, subgraphs, error = False, 0, None
    try:
        with engine.connect() as conn:
            subgraphs = conn.execute(text("SELECT count(*) FROM subgraphs")).scalar()
            db_ok = True
    except Exception as exc:
        error = str(exc).splitlines()[0]
    return {"status": "ok" if db_ok and model.ready else "degraded", "database": db_ok,
            "subgraphs": subgraphs, "model": model.ready, "model_error": model.error, "db_error": error}


@router.get("/model", summary="Сведения о модели и метрики")
def model_info(request: Request):
    model = _model(request)
    m = model.meta["metrics"]
    return {
        "dataset": model.meta.get("dataset", "Elliptic2"),
        "algorithm": "XGBoost",
        "n_features": len(model.features),
        "levels": model.levels,
        "test": m["test"],
        "levels_stats": m["levels"],
        "top_k": m.get("top_k", {}),
        "paper_baselines": m["paper_baselines"],
        "n_test": m["n_test"],
        "suspicious_share_test": m["suspicious_share_test"],
        "top_features": [{"feature": k, "label": feature_label(k), "importance": v}
                         for k, v in model.meta.get("top_features", {}).items()],
    }


@router.get("/demo", summary="Примеры для демонстрации")
def demo(request: Request):
    return _model(request).demo


def _find_subgraph(conn, query: str):
    row = conn.execute(text("SELECT cc_id FROM subgraphs WHERE cc_id = :q"), {"q": query}).first()
    if row:
        return row.cc_id, "subgraph"
    row = conn.execute(text("SELECT cc_id FROM membership WHERE cl_id = :q LIMIT 1"), {"q": query}).first()
    if row:
        return row.cc_id, "cluster"
    return None, None


@router.post("/check", summary="Проверить кластер или подграф")
def check(body: CheckRequest, request: Request):
    model = _model(request)
    query = body.query.strip()
    if not query:
        raise HTTPException(422, "Введите ID кластера или подграфа.")

    signature = None
    if body.ownership:
        signature = verify_message(body.ownership.address, body.ownership.message, body.ownership.signature)

    with engine.begin() as conn:
        cc_id, matched_by = _find_subgraph(conn, query)
        if not cc_id:
            raise HTTPException(404, "Кластер или подграф не найден в базе. Проверьте идентификатор.")
        sub = conn.execute(text("SELECT label, split, n_nodes, n_edges, features FROM subgraphs WHERE cc_id = :c"),
                           {"c": cc_id}).first()
        listed = conn.execute(text(
            "SELECT b.cl_id, b.source, b.reason FROM blacklist b JOIN membership m ON m.cl_id = b.cl_id "
            "WHERE m.cc_id = :c"), {"c": cc_id}).all()

        features = sub.features if isinstance(sub.features, dict) else json.loads(sub.features)
        proba, contribs = model.predict(features)
        score, level = risk_level(proba, model.levels)
        why = reasons(features, contribs, model.typical, model.dists, model.proxies)
        nodes = conn.execute(text("SELECT DISTINCT cl_id FROM membership WHERE cc_id = :c"), {"c": cc_id}).scalars().all()
        edges = [(e.src, e.dst) for e in conn.execute(
            text("SELECT src, dst FROM sub_edges WHERE cc_id = :c"), {"c": cc_id}).all()]
        why["patterns"] = with_stats(detect(nodes, edges), model.typology_stats)
        why["scheme"] = headline(why["patterns"])

        self_listed = any(r.cl_id == query for r in listed)
        if listed:
            score, level = (100 if self_listed else max(score, 85)), "high"
            for r in listed:
                who = "Проверяемый кластер" if r.cl_id == query else f"Кластер {r.cl_id[:10]}… из этого подграфа"
                why["risk"].insert(0, {
                    "feature": "blacklist", "label": "Чёрный список", "value": None, "typical": None,
                    "impact": None, "direction": "up", "hint": None,
                    "text": f"{who} в списке «{r.source}»" + (f": {r.reason}" if r.reason else ""),
                })

        text_conclusion, source = conclude(score, level, why, bool(listed))
        verified = bool(signature and signature.get("valid"))
        check_id = conn.execute(text(
            "INSERT INTO checks (query, cc_id, score, risk_level, proba, reasons, conclusion, conclusion_source, "
            "blacklisted, signature_verified, signed_address) VALUES (:q, :c, :s, :l, :p, CAST(:r AS JSONB), "
            ":t, :src, :b, :v, :a) RETURNING id"), {
            "q": query, "c": cc_id, "s": score, "l": level, "p": proba, "r": json.dumps(why, ensure_ascii=False),
            "t": text_conclusion, "src": source, "b": bool(listed), "v": verified,
            "a": body.ownership.address if verified else None,
        }).scalar()

    return {
        "check_id": check_id, "query": query, "cc_id": cc_id, "matched_by": matched_by,
        "score": score, "level": level, "level_ru": LEVEL_RU[level], "proba": round(proba, 4),
        "blacklisted": bool(listed), "reasons": why, "patterns": why["patterns"], "scheme": why.get("scheme", ""),
        "conclusion": text_conclusion, "conclusion_source": source,
        "hidden_note": HIDDEN_NOTE if any(r.get("hidden") for k in ("risk", "mitigating") for r in why[k]) else None,
        "signature": signature,
        "subgraph": {"n_nodes": sub.n_nodes, "n_edges": sub.n_edges},
        "dataset_label": "подозрительный" if sub.label == 1 else "законный",
        "dataset_split": sub.split,
    }


@router.get("/graph/{cc_id}", summary="Граф переводов подграфа")
def graph(cc_id: str, focus: Optional[str] = None):
    with engine.connect() as conn:
        nodes = conn.execute(text("SELECT DISTINCT cl_id FROM membership WHERE cc_id = :c"), {"c": cc_id}).scalars().all()
        if not nodes:
            raise HTTPException(404, "Подграф не найден.")
        edges = conn.execute(text("SELECT DISTINCT src, dst FROM sub_edges WHERE cc_id = :c"), {"c": cc_id}).all()
        listed = set(conn.execute(text(
            "SELECT b.cl_id FROM blacklist b JOIN membership m ON m.cl_id = b.cl_id WHERE m.cc_id = :c"),
            {"c": cc_id}).scalars().all())
    out_deg, in_deg = {}, {}
    for e in edges:
        out_deg[e.src] = out_deg.get(e.src, 0) + 1
        in_deg[e.dst] = in_deg.get(e.dst, 0) + 1
    return {
        "cc_id": cc_id,
        "nodes": [{"id": n, "label": n[:6], "focus": n == focus, "blacklisted": n in listed,
                   "out": out_deg.get(n, 0), "in": in_deg.get(n, 0)} for n in nodes],
        "edges": [{"from": e.src, "to": e.dst} for e in edges],
    }


@router.post("/verify-signature", summary="Проверить подпись владельца адреса (BIP-137)")
def verify_signature(body: Ownership):
    return verify_message(body.address, body.message, body.signature)


@router.get("/history", summary="Последние проверки")
def history(limit: int = 20):
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, query, cc_id, score, risk_level, blacklisted, signature_verified, created_at "
            "FROM checks ORDER BY id DESC LIMIT :n"), {"n": max(1, min(limit, 200))}).mappings().all()
    return [{**r, "level_ru": LEVEL_RU.get(r["risk_level"], r["risk_level"]),
             "created_at": r["created_at"].isoformat()} for r in rows]


@router.get("/report/{check_id}", summary="PDF-отчёт о проверке")
def report(check_id: int):
    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM checks WHERE id = :i"), {"i": check_id}).mappings().first()
    if not row:
        raise HTTPException(404, "Проверка не найдена.")
    data = dict(row)
    if isinstance(data.get("reasons"), str):
        data["reasons"] = json.loads(data["reasons"])
    pdf = build_report(data)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="traceon_check_{check_id}.pdf"'})


@router.get("/blacklist", summary="Чёрный список")
def blacklist_list():
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT cl_id, source, reason, added_at FROM blacklist ORDER BY added_at DESC")
                            ).mappings().all()
    return [{**r, "added_at": r["added_at"].isoformat()} for r in rows]


@router.post("/blacklist", summary="Добавить кластер в чёрный список")
def blacklist_add(item: BlacklistItem):
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO blacklist (cl_id, source, reason) VALUES (:c, :s, :r) "
            "ON CONFLICT (cl_id) DO UPDATE SET source = EXCLUDED.source, reason = EXCLUDED.reason"),
            {"c": item.cl_id.strip(), "s": item.source, "r": item.reason})
    return {"ok": True}


@router.delete("/blacklist/{cl_id}", summary="Удалить кластер из чёрного списка")
def blacklist_delete(cl_id: str):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM blacklist WHERE cl_id = :c"), {"c": cl_id})
    return {"ok": True}
