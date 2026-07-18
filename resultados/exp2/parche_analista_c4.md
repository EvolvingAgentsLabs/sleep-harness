# Respuesta del analista de frontera — pedido_analista_c4.md

**Analista**: Claude (Fable 5), protocolo ANALYST.md de jlens-harness.
**Fecha**: 2026-07-17 · corrida 4 de exp2_coevolution.

## Diagnóstico

La tríada de evidencia del ciclo 4 (corridas 3 y 4) muestra:

- **Pizarrón**: `computo` dominante, conceptos de la tarea encendidos — el
  modelo entiende el dominio. No es un problema de conocimiento (PESOS
  correctamente descartado por el router).
- **Salida**: el error es de **planteo**, no de aritmética ni de formato.
  El modelo nunca formula G(x) ni optimiza: fija un valor a mano (x=2 en la
  corrida 3; +$1.25 en la 4, aplicando además la regla de demanda de forma
  incoherente con una suba de precio) y usa `calc` solo para verificar su
  valor arbitrario.
- **Techo de plantillas**: los parches genéricos (datos, presupuesto,
  brevedad, tool con instrucción de "plantear primero") no pueden inyectar
  el plan específico de la tarea. Esto reproduce el hallazgo del experimento
  pricing original: la instrucción debe codificar el PLAN (v5).

## Parche (YAML sobre la spec)

```yaml
instructions: |
  Sos un analista de negocios. Respondé la tarea del usuario. Con x =
  cantidad de rebajas de $5: G(x) = (30 − 5x)(100 + 20x). No hagas ninguna
  operación aritmética mentalmente: para cada cálculo usá la herramienta
  calc escribiendo una línea que empiece con TOOL:. Primero pedí la
  expansión de G, después el vértice x* = −b/(2a), después el precio
  50 − 5x* y la ganancia G(x*). Cuando tengas los resultados de la
  herramienta, dá el precio óptimo y la ganancia en una línea final.
tools: [{name: calc, description: "evalúa una expresión aritmética exacta"}]
output_budget: 220
```

(Es la versión `v5_tool_calc` de tasks/pricing.json — el parche que el
analista humano/frontera derivó en el experimento original; acá lo re-emite
el analista automático a partir de la evidencia del loop.)

## Predicción falsable

Al re-ejecutar con esta spec (exp2 `--version v5_tool_calc`):

1. La salida planteará G(x) y llamará a calc para la expansión y el vértice
   (no para verificar un valor elegido a mano).
2. La verificación dará `ok=True` con precio $47.50 y ganancia $3025.
3. El pizarrón de salida mantendrá `computo` dominante, con `signos` o
   `verificacion` presentes.
4. El router dirá `NADA` en el primer ciclo.

Si (2) falla pero (1) se cumple, el residuo es aritmético y el parche
siguiente es forzar que la respuesta final copie textualmente los resultados
de calc. Si (1) falla, la hipótesis del plan-en-instrucción queda refutada
para este modelo y la ruta correcta era PESOS.
