"""Router de co-evolución (Idea 2): ¿pesos, tool o contexto?

Cruza el veredicto de ejecución (verificador) con la salud del proceso
interno (firmas del pizarrón) para decidir DÓNDE aplicar la mejora en la
fase de sueño. Codifica el hallazgo central de jlens-harness: la salida
sola no alcanza para diagnosticar — a veces el razonamiento interno es sano
pero la ejecución falla (techo de prompt → tool), y a veces el workspace ni
siquiera enciende la firma de la tarea (falta conocimiento → pesos).

Tabla de decisión (firma_tarea = p.ej. "computo"; "intento" = la salida
contiene números suficientes, evidencia de ejecución de cómputo):

  verificación ok                                  -> NADA (rutina)
  falla + pide_datos o missing_info >= umbral      -> CONTEXTO (sub-especificación)
  falla + truncada y (firma sana o intento)        -> FORMATO (presupuesto/brevedad)
  falla + firma_tarea sana o intento               -> TOOL (entendió, no ejecuta)
  falla + nada de lo anterior                      -> PESOS (ruta NREM/REM)

Calibrado con exp2 (2026-07-17), corridas 1-2: (a) missing_info no compite
con la firma de tarea — en pricing, computo enciende siempre por el dominio;
alcanza el umbral absoluto o `pide_datos` del verificador; (b) cada ruta se
decide con DOS canales (workspace + salida): `pide_datos`/`n_numeros`/
`truncada` son evidencia a nivel ejecución complementaria al pizarrón; (c)
una salida truncada en pleno cálculo no es falta de conocimiento — es
presupuesto de salida (el hallazgo v1/v3b del experimento pricing original).
PESOS queda como último recurso: reentrenar es la ruta cara.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Ruta(Enum):
    NADA = "nada"
    CONTEXTO = "contexto"
    FORMATO = "formato"
    TOOL = "tool"
    PESOS = "pesos"


@dataclass
class Decision:
    ruta: Ruta
    motivo: str
    evidencia: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ruta": self.ruta.value, "motivo": self.motivo,
                "evidencia": self.evidencia}


def _score(firmas: dict | None, nombre: str) -> float:
    if not firmas or nombre not in firmas:
        return 0.0
    return float(firmas[nombre].get("score", 0.0))


def decidir(verificacion: dict | None, firmas_salida: dict | None,
            firmas_prompt: dict | None = None, *, truncada: bool = False,
            firma_tarea: str = "computo", umbral_firma: float = 0.15,
            umbral_missing: float = 0.10, umbral_numeros: int = 3) -> Decision:
    """Decide la ruta de consolidación para un paso de wake."""
    ok = bool(verificacion and verificacion.get("ok"))
    s_tarea = max(_score(firmas_salida, firma_tarea), _score(firmas_prompt, firma_tarea))
    s_missing = max(_score(firmas_salida, "missing_info"),
                    _score(firmas_prompt, "missing_info"))
    n_numeros = int(verificacion.get("n_numeros", 0)) if verificacion else 0
    intento = n_numeros >= umbral_numeros
    ev = {"firma_tarea": firma_tarea, "score_tarea": round(s_tarea, 3),
          "score_missing": round(s_missing, 3), "n_numeros": n_numeros,
          "intento": intento, "truncada": truncada,
          "verificacion": verificacion}

    if ok:
        return Decision(Ruta.NADA, "la tarea verifica; consolidación de rutina", ev)
    pide_datos = bool(verificacion and verificacion.get("pide_datos"))
    ev["pide_datos"] = pide_datos
    if pide_datos or s_missing >= umbral_missing:
        origen = "la salida pide datos" if pide_datos else \
            "el pizarrón enciende missing_info"
        return Decision(
            Ruta.CONTEXTO,
            f"{origen}: la spec sub-especifica; parchear contexto/datos, "
            "no pesos", ev)
    if truncada and (s_tarea >= umbral_firma or intento):
        return Decision(
            Ruta.FORMATO,
            "la salida se truncó en pleno cálculo: no es falta de "
            "conocimiento, es presupuesto de salida; subir output_budget y "
            "pedir brevedad", ev)
    if s_tarea >= umbral_firma or intento:
        origen = ("la firma de la tarea enciende" if s_tarea >= umbral_firma
                  else "la salida intenta computar (números presentes)")
        return Decision(
            Ruta.TOOL,
            f"proceso interno sano ({origen}) pero la ejecución falla: techo "
            "de prompt; sintetizar una tool en vez de forzar los pesos", ev)
    return Decision(
        Ruta.PESOS,
        "sin firma de tarea, sin intento de cómputo y sin pedido de datos: "
        "el modelo no tiene el conocimiento; ruta paramétrica (NREM/REM en "
        "Colab)", ev)


# ---------- síntesis de parches sobre la AgentSpec ----------

TOOL_CALC = {
    "name": "calc",
    "description": ("evalúa una expresión aritmética exacta y devuelve el "
                    "número; ejemplo de uso: TOOL: calc((30 - 5*1.5)*(100 + 20*1.5))"),
}

INSTRUCCION_TOOL = (
    " No hagas ninguna operación aritmética mentalmente. Primero explicá en "
    "una o dos líneas QUÉ vas a calcular y por qué (el planteo completo, no "
    "un valor elegido a mano). Después emití cada cálculo UNA sola vez como "
    "línea TOOL: calc(...). Cuando recibas los resultados, dá la respuesta "
    "final; no repitas llamadas."
)

# Parche FORMATO (réplica del hallazgo v3b de pricing): brevedad quirúrgica
# para que el cálculo entre en el presupuesto de salida.
INSTRUCCION_BREVEDAD = (
    " Respondé de forma muy breve y directa, sin preámbulos ni definiciones: "
    "andá directo al cálculo y dá la respuesta final en la última línea. "
    "Máximo 7 líneas."
)


def sintetizar_parche(decision: Decision, spec_dict: dict, *,
                      datos_reales: str = "") -> dict | None:
    """Parche automático mínimo sobre la spec (plantillas). Para parches más
    finos, `prompt_analista` genera el pedido al analista de frontera
    siguiendo el protocolo ANALYST.md de jlens-harness."""
    spec = {k: (v.copy() if isinstance(v, (dict, list)) else v)
            for k, v in spec_dict.items()}
    if decision.ruta is Ruta.CONTEXTO:
        if datos_reales and datos_reales not in spec.get("data", ""):
            spec["data"] = (spec.get("data", "") + "\n" + datos_reales).strip()
            return spec
        return None
    if decision.ruta is Ruta.FORMATO:
        if INSTRUCCION_BREVEDAD not in spec.get("instructions", ""):
            spec["instructions"] = spec.get("instructions", "") + INSTRUCCION_BREVEDAD
            spec["output_budget"] = min(spec.get("output_budget", 160) * 2, 480)
            return spec
        return None
    if decision.ruta is Ruta.TOOL:
        tools = spec.get("tools", [])
        if not any(t.get("name") == "calc" for t in tools):
            spec["tools"] = tools + [dict(TOOL_CALC)]
            spec["instructions"] = spec.get("instructions", "") + INSTRUCCION_TOOL
            spec["output_budget"] = max(spec.get("output_budget", 200), 220)
            return spec
        return None
    return None  # PESOS y NADA no parchean la spec


def prompt_analista(decision: Decision, spec_dict: dict, tarea: str,
                    salida: str) -> str:
    """Pedido para un analista de frontera (Claude) cuando la plantilla no
    alcanza; sigue el contrato de ANALYST.md: parche + predicción falsable."""
    import json
    return (
        "Sos el analista de frontera del loop de mejora (protocolo ANALYST.md).\n"
        f"Ruta decidida por el router: {decision.ruta.value} — {decision.motivo}\n"
        f"Evidencia: {json.dumps(decision.evidencia, ensure_ascii=False)}\n\n"
        f"Spec actual:\n{json.dumps(spec_dict, ensure_ascii=False, indent=2)}\n\n"
        f"Tarea: {tarea}\n\nSalida del sujeto:\n{salida}\n\n"
        "Emití un parche YAML sobre la spec (solo los campos que cambian) y "
        "una `prediccion` falsable de qué va a cambiar en el pizarrón y en la "
        "verificación al re-ejecutar."
    )
