# Plan de validación de hipótesis — sleep-harness

Fecha: 2026-07-18 · Estado de partida: exp0 gate ✅, exp2 router ✅ (1 familia),
exp1f direccional (jspace 0.417 vs grad 0.333, CIs solapados), exp3 re-run en
curso con fix de CE.

Diagnóstico de potencia (datos reales): bootstrap 90% sobre las 12 QA da CIs
de ancho ~0.5 por condición; el efecto observado es ~0.08. Spearman entre
filtros = 0.39 (discriminables con pool más grande). La varianza entre
corridas casi-idénticas (`combinado` 0.167 vs `jspace` 0.417 con ~mismos
dreams) señala además un confound de ORDEN: `sft_lora` no mezclaba los textos
entre épocas.

## H1 — Idea 1: el filtro J-Space selecciona mejores dreams que el gradiente

**Claim a validar**: incorporación(jspace) > incorporación(grad) con igual
olvido, y corr(fidelidad, Δ_por_dream) > corr(gradiente, Δ_por_dream).

- **V1 · Rigor sin GPU** (hoy):
  - `sft_lora`: shuffle por época con semilla explícita (elimina el confound
    de orden detectado en `combinado`).
  - Plumbing de semillas end-to-end: selección, SFT y orden vía
    `SLEEP_SEED` en el runner.
  - Dataset `facts_v2`: 4 dominios ficticios (el actual + 3 nuevos),
    ≥48 QA totales; dreams m=16 por dominio (pool ≥64, donde los filtros
    divergen de verdad).
  - Script de análisis pareado: bootstrap por pregunta + McNemar pareado
    jspace-vs-grad, y matriz de acuerdo entre filtros.
- **V2 · Réplica multi-semilla** (GPU: ~3-4 h T4 con orquestador o ~1 h A100):
  - 4 condiciones × 4 semillas sobre facts_v2, mismo presupuesto calibrado
    (5 épocas, lr 2e-4, dup×2). 
  - **Criterio pre-registrado**: el CI 90% de la diferencia PAREADA
    (jspace − grad, misma semilla y mismas preguntas) excluye 0 → H1
    confirmada; incluye 0 con media > 0 → direccional; media ≤ 0 → refutada.
- **V3 · Matriz causal por dream** (GPU: ~5 h T4 / ~1.5 h A100):
  - RUN_PER_DREAM sobre el pool completo: un adaptador por dream, Δ de
    incorporación aislado. Reporte: corr(fidelidad, Δ) vs corr(grad, Δ) con
    CIs por bootstrap. Es el número publicable de la Idea 1.

## H2 — Idea 2: el router generaliza más allá de pricing

**Claim**: la tabla de decisión (2 canales por ruta) rutea correctamente en
familias de tareas nuevas sin recalibrar umbrales.

- **V4 · Replay offline** (sin GPU): re-ejecutar `decidir` sobre TODAS las
  trazas acumuladas (pricing × 6 corridas + wake de facts) con ground truth
  etiquetado a mano; reportar matriz de confusión de rutas.
- **V5 · Familia nueva en vivo** (GPU corta): una tarea de extracción/QA
  factual con spec deficiente → ¿el router va a CONTEXTO/PESOS donde
  corresponde? 2-3 ciclos alcanzan.

## H3 — Idea 3: workspace distillation retiene más / transfiere el proceso

**Gate previo**: el re-run de exp3 (CE fix) debe mostrar incorporación > 0.
Si sigue nulo, iterar el objetivo antes de escalar.

- **V6 · Réplica multi-semilla** (como V2: 2 cond × 4 semillas).
- **V7 · Sonda mecanística** (la evidencia distintiva): leer el pizarrón del
  STUDENT respondiendo el QA sin contexto y medir `ws_alineacion` = solape
  de conceptos con el pizarrón del teacher. Predicción: gkd_ws > gkd en
  alineación aunque empaten en accuracy — eso probaría que el término WS
  transfiere el proceso, no solo la salida.

## H0 — Gate del lens (mantenimiento)

- **V8**: el gate (control vs entrenado, 5+5 prompts) se corre como célula
  estándar al final de V2/V6 con los adaptadores reales de cada semilla.
- **V9 (opcional, solo A100)**: re-fit estructural n=32 sobre base y
  actualizado (comparación a igual n).

## Orden y presupuesto

| Fase | GPU | Cuándo |
|---|---|---|
| V1 + V4 | ninguna | hoy (local) |
| V2 | 3-4 h T4 / 1 h A100 | tras cierre de exp3 re-run |
| V3 | 5 h T4 / 1.5 h A100 | tras V2 (usa sus bundles) |
| V6+V7+V8 | 2-3 h T4 | si el gate de H3 pasa |
| V5, V9 | 1 h / 3 h | oportunista |

Con T4 el plan completo es ~2 noches de churn con el orquestador; con A100,
una sesión de ~4 h. Los criterios de decisión quedan fijados ANTES de correr
(este documento) para no ajustar el listón después de ver los datos.
