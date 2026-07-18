# sleep-harness

**Sleep cycles for LLM agents, guided by their internal workspace.** An open
experiment (Apache 2.0, by [Evolving Agents Labs](https://github.com/EvolvingAgentsLabs)):
a wake/sleep loop where a small model consolidates what it learned during the
day — with its "dreams" filtered, its failures triaged, and its distillation
guided by reading which concepts light up inside it (via the open
[Jacobian Lens](https://huggingface.co/neuronpedia/jacobian-lens)).

Early results and the full story: see the LinkedIn article *"We built the
sleeping agent: early results from an open experiment"* (link in the repo
description). Status: **research prototype** — single-model results, the
multi-seed validation matrix and controls are running; definitive tables and
a Gemma 4 (E4B) cross-model replication are the next milestones (see
`PLAN_VALIDACION.md` for the pre-registered protocol). Companion repo
`jlens-harness` (agent debugging with the Jacobian lens) is required as a
sibling directory — publication pending.

---

Implementación del **paradigma de sueño** para LLMs (Behrouz et al.,
*Language Models Need Sleep*, [arXiv:2606.03979](https://arxiv.org/abs/2606.03979))
cruzada con la **introspección J-Space** de
[`jlens-harness`](../jlens-harness) / [`jacobian-lens`](../jacobian-lens).

Tres contribuciones propias sobre el paper:

1. **Idea 1 — Filtrado de dreams por J-Space** (`sleep/dream_filter.py`): además
   del filtro por gradiente del paper (§3.4), cada dream se pasa por el pizarrón
   (`leer_pizarron` + `detectar`); si la firma de la tarea no se enciende, el
   dream es un candidato a alucinación y se descarta.
2. **Idea 2 — Co-evolución pesos/harness** (`evolution/router.py`): en sueño, el
   cruce verificador × firmas decide DÓNDE mejorar: `CONTEXTO` (spec
   sub-especifica), `TOOL` (proceso interno sano pero ejecución falla: techo de
   prompt), `PESOS` (falta conocimiento → NREM/REM) o `NADA`.
3. **Idea 3 — Workspace Distillation** (`sleep/workspace_loss.py`): la
   destilación de Knowledge Seeding suma un término que alinea las
   distribuciones **lens-decodificadas** del workspace entre teacher y student:
   destila no solo qué responder sino qué conceptos encender.

## Mapa paper → código

| Paper | Módulo |
|---|---|
| Wake / Sleep, chunks C^(l) (§3.1-3.2) | `wake/session.py`, `sleep/scheduler.py` |
| Expansión de parámetros + poda sináptica (§3.2-3.3) | `memory/lora_stack.py` (stack de LoRAs) |
| Knowledge Seeding: GKD λ-mix (§3.3) | `sleep/nrem.py::perdida_gkd`, `sleep/training.py::entrenar_ks` |
| Learning to Imitate, Ec. 3-4 (§3.3) | `sleep/nrem.py::recompensa_lti`, `lti_generar_pares` |
| Dreaming + random expert (§3.4) | `sleep/dreams.py` (steering de concepto aleatorio vía `Intervention`) |
| Selección por gradiente + Ec. 5, ReSTEM (§3.4) | `sleep/dream_filter.py`, `sleep/rem.py` |
| Eval: knowledge incorporation / forgetting | `eval/incorporation.py`, `eval/forgetting.py` |

Proxy arquitectural: no hay Hope/CMS público, así que la jerarquía de
frecuencias se emula con contexto (rápida) → LoRAs congelables (media) → base
fusionado (lenta), y el teacher de Knowledge Seeding es el modelo **con el
contexto en el prompt** (información privilegiada) destilando al mismo modelo
**sin contexto** con un LoRA nuevo — el mismo setup con el que el paper valida
sobre Transformers vanilla (Tabla 3).

## División local / Colab

**Regla del proyecto: todo fine-tuning corre primero en Google Colab.**

- **Local (Mac, MPS)**: lens, lectura del pizarrón, generación y scoring
  J-Space de dreams, router, evaluación, preparación de *bundles*.
- **Colab (GPU)**: SFT-LoRA, GKD, workspace distillation, re-fit del lens.
  Los notebooks (`notebooks/`) consumen un zip con los tres repos y los
  bundles; la lógica vive en el paquete, los notebooks solo orquestan.

## Setup local

Reutiliza el venv de jlens-harness (ya tiene torch/transformers/jlens):

```bash
cd ../jlens-harness
uv pip install --python .venv/bin/python -e "../sleep-harness[train,dev]"
cd ../sleep-harness
../jlens-harness/.venv/bin/python -m pytest tests/   # 21 tests, sin modelo
```

## Flujo de experimentos

```bash
PY=../jlens-harness/.venv/bin/python

# Exp0 — ¿el lens sigue valiendo tras updates? (correr PRIMERO)
$PY experiments/exp0_lens_drift.py                    # drift local sintético
#   → notebooks/colab_exp0_lens_refit.ipynb           # re-fit real en GPU

# Exp1 — Idea 1: filtrado de dreams por J-Space
$PY experiments/exp1_prepare_dreams.py                # genera bundle en bundles/
#   → notebooks/colab_exp1_dream_filter.ipynb         # 4 condiciones + SFT + eval

# Exp2 — Idea 2: co-evolución pesos/harness (100% local)
$PY experiments/exp2_coevolution.py                   # pricing v1 → router → tool/contexto

# Exp3 — Idea 3: Knowledge Seeding ± workspace distillation
#   → notebooks/colab_exp3_workspace_distill.ipynb    # gkd vs gkd+ws
```

Handoff a Colab (una vez por cambio de código):

```bash
scripts/package_for_colab.sh          # crea ../sleep_lab_bundle.zip
# subirlo a Drive en MyDrive/sleep_lab/sleep_lab_bundle.zip
# los bundles de exp1 van a Drive en MyDrive/sleep_lab/bundles/
```

Los notebooks se regeneran con `python scripts/gen_notebooks.py` (editar ahí,
no en los .ipynb).

## Qué resultado valida qué

- **Exp0**: solape de readout ≥ ~0.7 tras LoRA ⇒ las Ideas 1 y 3 pueden usar el
  lens original entre sueños; si no, el re-fit entra al ciclo de sueño.
- **Exp1**: si `jspace`/`combinado` ≥ `grad` en incorporación con menos olvido —
  y sobre todo si `corr(score_jspace, mejora_por_dream) > 0` (celda opcional) —
  el filtro por proceso interno aporta señal que el filtro por salida no tiene.
- **Exp2**: la trayectoria esperada replica v1→v2→v5 del experimento pricing de
  jlens-harness, ahora decidida por el router en vez del analista humano.
- **Exp3**: `gkd_ws` con igual o mejor incorporación y **menos olvido** que
  `gkd` ⇒ destilar el workspace transfiere el proceso, no solo la salida.
