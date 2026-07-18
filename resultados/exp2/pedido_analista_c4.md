Sos el analista de frontera del loop de mejora (protocolo ANALYST.md).
Ruta decidida por el router: formato — la salida se truncó en pleno cálculo: no es falta de conocimiento, es presupuesto de salida; subir output_budget y pedir brevedad
Evidencia: {"firma_tarea": "computo", "score_tarea": 0.758, "score_missing": 0.0, "n_numeros": 45, "intento": true, "truncada": true, "verificacion": {"ok": false, "encontrados": {"precio": false, "ganancia": false}, "pide_datos": false, "n_numeros": 45}, "pide_datos": false}

Spec actual:
{
  "name": "pricing_coevolucion",
  "instructions": "Sos un analista de negocios. Respondé la tarea del usuario. Respondé de forma muy breve y directa, sin preámbulos ni definiciones: andá directo al cálculo y dá la respuesta final en la última línea. Máximo 7 líneas. No hagas ninguna operación aritmética mentalmente. Primero explicá en una o dos líneas QUÉ vas a calcular y por qué (el planteo completo, no un valor elegido a mano). Después emití cada cálculo UNA sola vez como línea TOOL: calc(...). Cuando recibas los resultados, dá la respuesta final; no repitas llamadas.",
  "context": "",
  "data": "Costo unitario: $20. Al precio de $50 vendo 100 unidades por mes. Cada rebaja de $5 en el precio suma 20 unidades vendidas por mes.",
  "tools": [
    {
      "name": "calc",
      "description": "evalúa una expresión aritmética exacta y devuelve el número; ejemplo de uso: TOOL: calc((30 - 5*1.5)*(100 + 20*1.5))"
    }
  ],
  "subagents": [],
  "output_budget": 320
}

Tarea: ¿Cuál es el precio óptimo para mi nuevo producto para maximizar la ganancia mensual? Dame el precio exacto y la ganancia esperada.

Salida del sujeto:
El precio óptimo es $51.25 y la ganancia mensual esperada es $1690.25.

TOOL: calc((50 + 1.25 - 20) * (100 + 20 * 1.25 / 5))
TOOL: calc(31.25 * 130)
TOOL: calc(4062.5)

El precio óptimo es $51.25 y la ganancia mensual esperada es $4062.50.

TOOL: calc((50 + 1.25 - 20) * (100 + 20 * 1.25 / 5))
TOOL: calc(31.25 * 130)
TOOL: calc(4062.5)

El precio óptimo es $51.25 y la ganancia mensual esperada es $4062.50.

TOOL: calc((50 + 1.25 - 20) * (100 + 20 * 1.25 / 5))
TOOL: calc(31.25 * 130)
TOOL: calc(4062.5)

El precio óptimo es $51.25 y la ganancia mensual esperada es $4062.

Emití un parche YAML sobre la spec (solo los campos que cambian) y una `prediccion` falsable de qué va a cambiar en el pizarrón y en la verificación al re-ejecutar.