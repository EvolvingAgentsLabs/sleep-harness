"""Evaluación de knowledge incorporation (setup de la Tabla 3 del paper).

Después del sueño, el modelo debe responder SIN contexto preguntas cuyos
hechos vio solo durante wake. qa: [{pregunta, respuesta}]; el acierto es por
contención normalizada de la respuesta esperada (o alternativas).
"""

from __future__ import annotations

from .forgetting import _norm


def acierta(salida: str, respuesta: str | list[str]) -> float:
    alternativas = respuesta if isinstance(respuesta, list) else [respuesta]
    texto = _norm(salida)
    return float(any(_norm(a) in texto for a in alternativas))


def evaluar_incorporacion(generar, qa: list[dict], *, plantilla: str | None = None,
                          log=None) -> dict:
    """`generar(prompt) -> str` SIN contexto en el prompt. Devuelve
    {accuracy, por_pregunta}."""
    plantilla = plantilla or "{pregunta}\n\nRespondé en una línea, solo el dato pedido.\n\nRespuesta:\n"
    por_pregunta = []
    for item in qa:
        salida = generar(plantilla.format(pregunta=item["pregunta"]))
        s = acierta(salida, item["respuesta"])
        por_pregunta.append({"pregunta": item["pregunta"], "score": s,
                             "salida": salida.strip()[:200]})
        if log:
            log(f"  [{int(s)}] {item['pregunta'][:60]}")
    acc = sum(p["score"] for p in por_pregunta) / max(len(por_pregunta), 1)
    return {"accuracy": round(acc, 4), "por_pregunta": por_pregunta}
