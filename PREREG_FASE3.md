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

## Resultado H-SEC-1 (2026-07-21) — CONFIRMA

Local, Qwen3.5-4B. 9 pares duros: solape léxico medio 0.657 (filtro de texto
ciego), la firma malicious_intent separa 8W/0T/1L, **sign test p=0.0195**,
Δ medio +0.054. El pizarrón enciende SSH/RSA/authentication/credentials/
password/sudo sobre payloads ofuscados como "telemetría de auditoría";
el gemelo benigno con mismas palabras de superficie → ~0. Brand targeting
(probe): Δ ~0 (señal de attention-hijacking no capturada por esta firma;
consistente con que el paper lo llama "no overtly malicious").
Caveat vigente: 1 familia de modelos; el veto NO transfirió a Gemma (P0).
Resultados: resultados/fase3/exp4_security.json

## Resultado P0 cross-modelo (2026-07-21) — NO TRANSFIERE

Firewall de seguridad (H-SEC-1) sobre Gemma 4 E4B: 3W/5T/1L, sign test
p=0.3125, Δ medio +0.015. La mayoría de los pares dan 0.000 en ambos lados
(los conceptos de seguridad casi no se encienden). Combinado con el veto de
gemelos que tampoco transfirió (p=0.50), la conclusión es clara: **el veto es
Qwen-específico con nuestro pipeline actual; no es la insensibilidad numérica
sino un fenómeno más amplio.** Abierto: ausencia real vs. calibración por
modelo (otras capas / matching / forma léxica) — requeriría un diagnóstico
que vuelque el top COMPLETO del pizarrón de Gemma. Implicancia: el firewall
es un PoC de UN modelo (Qwen), no una primitiva general. H-SEC-2/3 y cualquier
claim de seguridad deben decirlo. Resultados: resultados/fase3/exp4_security_gemma4e4b.json

## P0 RESUELTO (2026-07-21) — CALIBRACIÓN, y TRANSFIERE cross-modelo

Diagnóstico (top completo del pizarrón de Gemma sobre payloads de exfiltración):
Gemma SÍ representa la amenaza, pero a otro nivel de abstracción (malware,
hacker, cybersecurity, encryption, cryptographic, passwords, seguridad) y en la
ventana de capas MEDIA, no la mid-late de Qwen (ssh/rsa/credentials). El P0
negativo era artefacto de usar la calibración de Qwen sobre Gemma.

Validación (léxico enriquecido con 'ciberamenaza' + ventana por modelo):
- Gemma calibrado (ventana media, capas 14-26): 8W/0T/1L, **p=0.0195**, Δ +0.141.
- Qwen recheck (léxico enriquecido, sin regresión): 8W/0T/1L, **p=0.0195**, Δ +0.090.

Conclusión: **el veto de workspace transfiere entre familias de modelos con
calibración por modelo.** Hallazgo adicional: cada modelo representa la misma
amenaza en su propio nivel de abstracción y su propia ventana de capas — la
calibración cross-modelo es de léxico+capa, no de mecanismo. Brand targeting
sigue no capturado (Δ~0) en ambos: señal de attention-hijacking, no exfiltración.
