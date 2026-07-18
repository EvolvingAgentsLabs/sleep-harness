"""Verificadores de tareas (veredicto de ejecución para el router).

`verificar_numerico` replica el contrato del verificador de pricing_demo de
jlens-harness: busca los valores esperados en la salida con tolerancia y
detecta el modo "pide datos" (la salida sana de una spec sub-especificada).
"""

from __future__ import annotations

import re

_PIDE_DATOS = re.compile(
    r"(necesito|need|falta|faltan|missing|provide|proporcion|más información|"
    r"mas informacion|more information|could you|podrías|podrias|specify|"
    r"especific|what is|cuál es|cual es).{0,80}(dato|data|información|"
    r"informacion|information|costo|cost|precio|price|demanda|demand)",
    re.IGNORECASE | re.DOTALL)

# Formulación negativa detectada en exp2 ("no puedo darte un precio exacto
# … sin datos fundamentales"): rechazo por falta de información.
_PIDE_DATOS_NEG = re.compile(
    r"(no puedo|no es posible|cannot|can't|unable to).{0,100}"
    r"sin\s+(los\s|estos\s|más\s|mas\s)?(dato|información|informacion)"
    r"|without\s+(the\s|more\s)?(data|information)"
    r"|sin\s+(estos|esos|los)\s+datos",
    re.IGNORECASE | re.DOTALL)


def _numeros(texto: str) -> list[float]:
    out = []
    for m in re.finditer(r"-?\$?\s?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?)", texto):
        s = m.group(1)
        # normalización simple: si hay ambas, la última es decimal
        if "," in s and "." in s:
            s = s.replace(",", "") if s.rindex(".") > s.rindex(",") else s.replace(".", "").replace(",", ".")
        elif "," in s:
            partes = s.split(",")
            s = s.replace(",", "") if len(partes[-1]) == 3 and len(partes) > 1 else s.replace(",", ".")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def verificar_numerico(salida: str, esperado: dict[str, float], *,
                       tolerancia: float = 0.01) -> dict:
    """esperado: {nombre: valor}. ok = todos los valores aparecen (± tol
    relativa). Reporta también si la salida pide datos en lugar de resolver."""
    nums = _numeros(salida)
    encontrados = {}
    for nombre, valor in esperado.items():
        hit = any(abs(n - valor) <= tolerancia * max(abs(valor), 1.0) for n in nums)
        encontrados[nombre] = hit
    return {
        "ok": all(encontrados.values()),
        "encontrados": encontrados,
        "pide_datos": (_PIDE_DATOS.search(salida) is not None
                       or _PIDE_DATOS_NEG.search(salida) is not None),
        "n_numeros": len(nums),
    }
