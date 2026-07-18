"""Medición de catastrophic forgetting: suite fija de sondas antes/después.

Una sonda es {id, prompt, esperado} donde esperado es {contains: [...]} o
{regex: "..."} (case-insensitive). La misma suite se corre antes y después
de cada ciclo de sueño; `comparar` reporta el olvido por sonda y medio.
Formato compartido con tasks/sondas_olvido.json y con los bundles de Colab.
"""

from __future__ import annotations

import re
import unicodedata


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def puntuar(salida: str, esperado: dict) -> float:
    """1.0 si la salida satisface lo esperado, 0.0 si no."""
    texto = _norm(salida)
    if "contains" in esperado:
        req = [_norm(str(x)) for x in esperado["contains"]]
        return float(all(r in texto for r in req))
    if "regex" in esperado:
        return float(re.search(esperado["regex"], salida, re.IGNORECASE) is not None)
    raise ValueError(f"esperado sin criterio: {esperado}")


def correr_sondas(generar, sondas: list[dict], *, log=None) -> dict[str, float]:
    """`generar(prompt) -> str`. Devuelve {id_sonda: score}."""
    out = {}
    for s in sondas:
        salida = generar(s["prompt"])
        out[s["id"]] = puntuar(salida, s["esperado"])
        if log:
            log(f"  sonda {s['id']}: {out[s['id']]:.0f}")
    return out


def comparar(antes: dict[str, float], despues: dict[str, float]) -> dict:
    """Olvido = caída de score por sonda; positivo = olvidó."""
    ids = sorted(set(antes) & set(despues))
    por_sonda = {i: round(antes[i] - despues[i], 3) for i in ids}
    olvidadas = [i for i, d in por_sonda.items() if d > 0]
    return {
        "olvido_medio": round(sum(por_sonda.values()) / max(len(ids), 1), 4),
        "acc_antes": round(sum(antes[i] for i in ids) / max(len(ids), 1), 4),
        "acc_despues": round(sum(despues[i] for i in ids) / max(len(ids), 1), 4),
        "sondas_olvidadas": olvidadas,
        "por_sonda": por_sonda,
    }
