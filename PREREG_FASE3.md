# Pre-registro — Fase 3: Firewall subconsciente (2026-07-21)

Criterios fijados ANTES de ver los datos. Base: el veto de gemelos que
sobrevivió (p=0.019, texto idéntico / relación corrupta) es estructuralmente
el ataque de *Bad Memory* (Gadgil et al., UW, arXiv:2607.14611): payloads
ofuscados con alto solape léxico contra texto benigno (ej. leer `~/.ssh/id_rsa`
disfrazado de "license-audit telemetry"). Hipótesis: el filtro léxico es ciego
por construcción; el pizarrón (J-Space) enciende conceptos de seguridad al
procesar el payload aunque el texto los oculte.

## H-SEC-1 (core, LOCAL, sin fine-tuning) — exp4_security_firewall.py

Por cada par {payload malicioso ofuscado, gemelo benigno de alto solape
léxico} reproduciendo los 3 goals del paper (12 pares).

**Criterio pre-registrado** (sobre los 9 pares inequívocamente maliciosos:
credential_exfiltration + unauthorized_tool_use):
1. solape léxico medio malicioso-vs-benigno > 0.5 (el filtro de texto NO
   discrimina), y
2. la firma `malicious_intent` del pizarrón separa malicioso de benigno con
   sign test p < 0.05 (mayoría de deltas positivos).

- CONFIRMA si (1) y (2). NULO si no.
- `brand_targeting` es probe secundario (señal más sutil: "attention
  hijacking", no exfiltración) — se reporta aparte, sin condicionar el
  veredicto.

**Caveats**: una familia de modelos (Qwen; recordar que el veto NO transfirió
a Gemma — la calibración cross-modelo es P0 pendiente), pares construidos por
nosotros, léxico de seguridad fijo. Evidencia direccional de PoC, no un
detector production-ready.

## H-SEC-2 (siguiente) — Firewall de sueño

Integrar `score_seguridad` como veto en el router/dream_filter: antes de
consolidar una edición de memoria/AgentSpec, leer el pizarrón del parche; si
`malicious_intent` supera umbral (calibrado en validación aparte del test),
`ok=False` → se bloquea la consolidación. Métrica: reduce la persistencia del
payload en una secuencia multi-sesión vs baseline sin veto, sin bloquear
actualizaciones benignas por encima de una tasa de falsos positivos fijada.

## H-SEC-3 (siguiente) — Trojan scanner de LoRAs (usa Finding 1)

Diff de readouts del lens (método del gate exp0) antes/después de montar un
adaptador de terceros, sobre prompts de control. Un LoRA con trojan (entrenado
para desviar conceptos de credenciales ante un trigger) muestra drift en esos
conceptos mayor que un LoRA benigno. Requiere entrenar un LoRA troyano (Colab).

Orden: H-SEC-1 (ahora, local) → H-SEC-2 → H-SEC-3.
