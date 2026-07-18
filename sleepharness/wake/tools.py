"""Ejecución de tools del sujeto (contrato TOOL: de jlens-harness).

`calc` evalúa aritmética con un walker de AST (sin eval crudo). El loop
`paso_con_tools` corre un paso, ejecuta las líneas `TOOL: nombre(args)` e
inyecta los resultados para un segundo paso — el mismo MVP del harness,
empaquetado para reutilizarlo desde el router de co-evolución.
"""

from __future__ import annotations

import ast
import operator
import re

from harness.agentspec import AgentSpec, componer_prompt

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos, ast.Mod: operator.mod,
}


def _eval_nodo(n):
    if isinstance(n, ast.Expression):
        return _eval_nodo(n.body)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_eval_nodo(n.left), _eval_nodo(n.right))
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_eval_nodo(n.operand))
    raise ValueError(f"expresión no permitida: {ast.dump(n)}")


def calc(expr: str) -> str:
    """Evalúa una expresión aritmética exacta; devuelve el número o el error."""
    try:
        v = _eval_nodo(ast.parse(expr.strip(), mode="eval"))
        return f"{v:.6g}"
    except Exception as e:  # el sujeto ve el error y puede corregir
        return f"error: {e}"

EJECUTORES = {"calc": calc}

_TOOL_RE = re.compile(r"^TOOL:\s*(\w+)\((.*)\)\s*$", re.MULTILINE)


def dedupe_llamadas(llamadas: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Elimina llamadas repetidas preservando el orden (exp2 corrida 3: el
    modelo en greedy repite el mismo bloque de TOOLs hasta agotar el budget)."""
    vistas, unicas = set(), []
    for ll in llamadas:
        if ll not in vistas:
            vistas.add(ll)
            unicas.append(ll)
    return unicas


def paso_con_tools(rt, spec: AgentSpec, tarea: str, *, rastrear=None,
                   max_rondas: int = 2):
    """Corre la tarea; si la salida invoca tools, las ejecuta e inyecta los
    resultados en un paso siguiente. Devuelve (resultado_final, rondas)."""
    prompt = componer_prompt(spec, tarea, rt.arranque)
    r = rt.step(prompt, max_new_tokens=spec.output_budget, rastrear=rastrear,
                max_posiciones_salida=20)
    rondas = [r]
    for _ in range(max_rondas - 1):
        llamadas = dedupe_llamadas(_TOOL_RE.findall(r.salida))
        if not llamadas:
            break
        resultados = [
            f"TOOL {nombre}({args}) = "
            f"{EJECUTORES.get(nombre, lambda a: 'error: tool desconocida')(args)}"
            for nombre, args in llamadas
        ]
        # la ronda final prohíbe más TOOLs y re-inyecta el arranque para que
        # el modelo vaya directo a la respuesta (no a repetir el patrón)
        prompt = (r.prompt + r.salida + "\n\nResultados de las herramientas:\n"
                  + "\n".join(resultados)
                  + "\n\nCon estos resultados, dá la respuesta final en una "
                  "línea (no escribas más líneas TOOL):\n" + rt.arranque)
        r = rt.step(prompt, max_new_tokens=spec.output_budget, rastrear=rastrear,
                    max_posiciones_salida=20)
        rondas.append(r)
    return r, rondas
