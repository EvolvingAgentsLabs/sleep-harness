"""Overlay de firmas: léxico extendido sobre harness.signatures.FIRMAS.

Calibración de exp2 corrida 2 (2026-07-17): el pizarrón del ciclo 2 encendía
`variables, quantity, calculations, pricing, sales, analysis` — cómputo real
que las anclas originales no cubren (score 0.084 < umbral 0.15) — y el router
concluyó "no tiene el conocimiento". Este overlay agrega las anclas
observadas SIN modificar jlens-harness: sus experimentos conservan la
semántica original; sleep-harness usa `detectar_ext`.
"""

from __future__ import annotations

import re

from harness.signatures import FIRMAS

EXTRA_COMPUTO = {
    # observadas en el pizarrón real del ciclo 2 de exp2
    "variables", "variable", "quantity", "quantities", "cantidad",
    "cantidades", "calculation", "calculations", "cálculo", "calculo",
    "cálculos", "calculos", "analysis", "análisis", "analisis", "pricing",
    "sales", "ventas",
    # vecinas del mismo dominio
    "elasticity", "elasticidad", "equilibrium", "equilibrio", "margin",
    "margen", "volume", "volumen", "unit", "unidad",
}

FIRMAS_EXT: dict[str, set[str]] = {k: set(v) for k, v in FIRMAS.items()}
FIRMAS_EXT["computo"] |= EXTRA_COMPUTO


def _es_digito(token: str) -> bool:
    return re.fullmatch(r"\d+", token.strip()) is not None


def detectar_ext(top: list[dict]) -> dict:
    """Misma lógica que harness.signatures.detectar, con FIRMAS_EXT."""
    total = sum(t["intensidad"] for t in top) or 1.0
    out = {}
    for nombre, anclas in FIRMAS_EXT.items():
        peso, conceptos = 0.0, []
        for t in top:
            tok = t["token"].lower()
            if tok in anclas or (nombre == "computo" and _es_digito(tok)):
                peso += t["intensidad"]
                conceptos.append(t["token"])
        out[nombre] = {"score": round(peso / total, 3), "conceptos": conceptos}
    dominante = max(out, key=lambda k: out[k]["score"])
    out["_dominante"] = dominante if out[dominante]["score"] > 0 else None
    return out
