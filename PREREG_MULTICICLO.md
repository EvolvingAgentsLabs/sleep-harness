# Pre-registro — Test multi-ciclo de olvido (2026-07-20)

Criterios fijados ANTES de ver los datos. Objetivo: probar, tan fielmente
como permite un transformer vanilla, la tesis continual de Behrouz — que la
consolidación con expansión de parámetros mitiga el olvido catastrófico a lo
largo de MÚLTIPLES ciclos, no solo en uno.

## Diseño

4 ciclos secuenciales, un dominio de hechos ficticios por ciclo (facts_v2:
Vantar → Quelmara → Nordina → Ferrovex). Dos brazos:

- **A — Expanding KS** (mecanismo del paper, proxy): cada ciclo agrega un
  LoRA NUEVO (los previos congelados y activos = "parameter expansion",
  §3.2), y consolida el dominio del ciclo por destilación KS
  (teacher = base + contexto; student = stack completo sin contexto;
  peso_ws=0, la variante que sobrevivió). `memory/lora_stack.py`.
- **B — Naive continual SFT** (el contraste que debería olvidar): un solo
  LoRA, SFT directo sobre el dominio del ciclo, reentrenado cada ciclo.

Tras cada ciclo se mide: (a) incorporación del dominio actual (¿aprendió?);
(b) retención de los dominios previos (¿olvidó lo de ciclos anteriores?);
(c) sondas de capacidad general (¿olvidó habilidades?).

## Hipótesis y criterio (pre-registrado)

**H-MC**: en el brazo A, la retención de los hechos del dominio 0 medida tras
el ciclo 3 es mayor que en el brazo B; y el olvido de sondas generales del
brazo A se mantiene ≈0 a lo largo de los ciclos mientras el del brazo B crece.

**Veredicto**:
- CONFIRMA la tesis continual (a escala proxy) si A retiene el dominio 0
  claramente mejor que B al final Y el olvido general de A ≤ el de B en cada
  ciclo.
- REFUTA / NULO si A no retiene mejor que B, o si A también sufre olvido
  creciente.

**Caveats que van en cualquier reporte**: n chico, una semilla, transformer
vanilla con expansión LoRA como proxy de MoE-frecuencia (no Hope). Es
evidencia DIRECCIONAL a escala proxy, consistente con el ethos del proyecto,
no una réplica de los benchmarks del paper.
