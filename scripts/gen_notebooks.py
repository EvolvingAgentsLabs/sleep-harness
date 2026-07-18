#!/usr/bin/env python3
"""Genera los notebooks de Google Colab en notebooks/.

Política del proyecto: todo fine-tuning corre en Colab. Los notebooks son
orquestadores finos — la lógica vive en sleepharness (testeada y versionada);
acá solo se cargan datos, se llama al paquete y se guardan resultados en
Drive. Regenerar con:  python scripts/gen_notebooks.py
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"

_next_id = 0


def _cell(tipo, src):
    global _next_id
    _next_id += 1
    lineas = src.strip("\n").split("\n")
    fuente = [l + "\n" for l in lineas[:-1]] + [lineas[-1]]
    c = {"cell_type": tipo, "metadata": {}, "source": fuente,
         "id": f"cell-{_next_id:03d}"}
    if tipo == "code":
        c.update(execution_count=None, outputs=[])
    return c


def M(src):
    return _cell("markdown", src)


def C(src):
    return _cell("code", src)


def notebook(cells, titulo):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4", "name": titulo},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


# ---------------------------------------------------------------- setup común

SETUP_DEPS = C("""
%pip install -q "transformers>=5.5" peft accelerate datasets rapidfuzz sentence-transformers
""")

SETUP_DRIVE = C("""
# Requisito (una sola vez): en la Mac, correr scripts/package_for_colab.sh y
# subir el zip resultante a Drive en  MyDrive/sleep_lab/sleep_lab_bundle.zip
from google.colab import drive
drive.mount('/content/drive')

BUNDLE_ZIP = '/content/drive/MyDrive/sleep_lab/sleep_lab_bundle.zip'
DRIVE_OUT = '/content/drive/MyDrive/sleep_lab/resultados'

import os, sys
os.makedirs(DRIVE_OUT, exist_ok=True)
!rm -rf /content/lab && mkdir -p /content/lab
!unzip -q "{BUNDLE_ZIP}" -d /content/lab
LAB = '/content/lab'
%pip install -q -e {LAB}/jacobian-lens
sys.path.insert(0, f'{LAB}/jlens-harness')
sys.path.insert(0, f'{LAB}/sleep-harness')
os.environ['SLEEPHARNESS_JLENS_HARNESS'] = f'{LAB}/jlens-harness'
import sleepharness
print('sleepharness', sleepharness.__version__, '| lab en', LAB)
""")

SETUP_MODEL = C("""
import torch, transformers
from sleepharness import config

assert torch.cuda.is_available(), 'Runtime > Change runtime type > GPU (T4/L4/A100)'
tok = transformers.AutoTokenizer.from_pretrained(config.HF_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = transformers.AutoModelForCausalLM.from_pretrained(
    config.HF_MODEL, dtype=torch.bfloat16, device_map='cuda')
model.config.pad_token_id = tok.eos_token_id
N_CAPAS = model.config.num_hidden_layers
print(config.HF_MODEL, '|', N_CAPAS, 'capas')
""")

SETUP_LENS = C("""
import jlens
lens = jlens.JacobianLens.from_pretrained(
    config.LENS_REPO, filename=config.LENS_FILE, revision=config.LENS_REVISION)
desde = int(round(N_CAPAS * 18 / 64))
CAPAS = [l for l in lens.source_layers if desde <= l < N_CAPAS]
print('capas de workspace:', len(CAPAS), '(desde', desde, ')')
""")


def guardar_resultados(nombre):
    return C(f"""
import json, time
res_path = f"{{DRIVE_OUT}}/{nombre}_{{time.strftime('%Y%m%d_%H%M%S')}}.json"
with open(res_path, 'w') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
print('resultados →', res_path)
""")


# ============================================================ exp0: lens refit

exp0 = [
    M("""
# Exp0 — Validez del Jacobian lens tras actualizar pesos (parte GPU)

Contraparte de `experiments/exp0_lens_drift.py` (local), con un **update
real**. Dos niveles de evidencia:

1. **Gate barato (minutos)**: readout del lens ORIGINAL sobre el modelo base
   vs sobre el modelo con un LoRA real fusionado — ¿el pizarrón sigue
   leyendo igual? Es la pregunta operativa para las Ideas 1 y 3.
2. **Re-fit chico opcional (horas)**: `jlens.fit` con REFIT_N prompts sobre
   AMBOS modelos (mismo n, comparación justa) — drift estructural de J_l.
   Ojo: cada prompt cuesta ~d_model/8 backwards (~320 en el 4B); N=200 en T4
   serían ~40 h. REFIT_N=4 por defecto; subilo solo con L4/A100
   (env SLEEP_REFIT_N).
"""),
    SETUP_DEPS, SETUP_DRIVE, SETUP_MODEL, SETUP_LENS,
    C("""
# 1) Baseline: readout del lens original sobre el modelo BASE
import json, os
lm = jlens.from_hf(model, tok)

CALIB = [
    'El precio óptimo se obtiene maximizando la ganancia G(x) = (30 - 5x)(100 + 20x).',
    'La capital de Francia es París, una ciudad con más de dos millones de habitantes.',
    'La membrana cerámica opera a 41 grados y reduce el consumo energético un 37 por ciento.',
    'To verify the result, recompute each step and check the sign of the linear term.',
    'Vantar Dynamics fue fundada en 2019 en Rosario por la ingeniera Lucía Ferreyra.',
]

def top_ids(lens_obj, texto, k=25):
    ids = lm.encode(texto, max_length=256)
    seq = ids.shape[1]
    pos = list(range(max(16, seq - 8), seq))
    logits, _, _ = lens_obj.apply(lm, texto, layers=CAPAS, positions=pos,
                                  max_seq_len=256)
    acc = {}
    for lg in logits.values():
        v, i = lg.topk(25, dim=-1)
        for fv, fi in zip(v, i):
            for vv, ii in zip(fv.tolist(), fi.tolist()):
                acc[ii] = acc.get(ii, 0.0) + vv
    return set(sorted(acc, key=acc.get, reverse=True)[:k])

base_tops = {p: top_ids(lens, p) for p in CALIB}
print('baselines listos')
"""),
    C("""
# 2) (opcional) Fit chico de J sobre el modelo BASE, para comparación justa
from datasets import load_dataset

REFIT_N = int(os.environ.get('SLEEP_REFIT_N', '4'))
prompts_fit = []
if REFIT_N:
    ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1',
                      split='train', streaming=True)
    for ex in ds:
        t = ex['text'].strip()
        if len(t) > 200:
            prompts_fit.append(t)
        if len(prompts_fit) >= REFIT_N:
            break
    print(f'fit J_base con n={REFIT_N} (esto es lo lento)…')
    lens_base_n = jlens.fit(lm, prompts_fit,
                            source_layers=list(lens.source_layers),
                            checkpoint_path='/content/lens_base_n.pt',
                            checkpoint_every=None)
else:
    lens_base_n = None
    print('re-fit deshabilitado (SLEEP_REFIT_N=0)')
"""),
    C("""
# 3) LoRA real sobre los hechos, fusionado al base (= "modelo actualizado")
from peft import LoraConfig, get_peft_model
from sleepharness.sleep.training import sft_lora

task = json.load(open(f'{LAB}/sleep-harness/tasks/facts_mini.json'))
pm = get_peft_model(model, LoraConfig(**config.LORA), adapter_name='exp0')
sft_lora(pm, tok, task['contextos'] * 4, epochs=2, lr=2e-4)
model = pm.merge_and_unload()
lm = jlens.from_hf(model, tok)   # wrapper sobre el modelo fusionado
print('LoRA fusionado; parámetros del base actualizados')
"""),
    C("""
# 4) GATE BARATO: el lens ORIGINAL sobre el modelo actualizado
solape_gate = {}
for p in CALIB:
    b = top_ids(lens, p)
    a = base_tops[p]
    solape_gate[p[:50]] = round(len(a & b) / len(a | b), 3)
    print(f"  {solape_gate[p[:50]]:.3f}  {p[:70]}")
gate_medio = round(sum(solape_gate.values()) / len(solape_gate), 4)
print('GATE (jaccard medio, lens original base vs actualizado):', gate_medio)
"""),
    C("""
# 5) (opcional) Fit chico de J sobre el modelo ACTUALIZADO + drift a igual n
drift = {}
solape_fits = {}
if REFIT_N and lens_base_n is not None:
    print(f'fit J_actualizado con n={REFIT_N}…')
    lens_upd_n = jlens.fit(lm, prompts_fit,
                           source_layers=list(lens.source_layers),
                           checkpoint_path='/content/lens_upd_n.pt',
                           checkpoint_every=None)
    for l in lens.source_layers:
        a = lens_base_n.jacobians[l].float()
        b = lens_upd_n.jacobians[l].float()
        drift[l] = round(float((b - a).norm() / a.norm()), 4)
    for p in CALIB:
        sa, sb = top_ids(lens_base_n, p), top_ids(lens_upd_n, p)
        solape_fits[p[:50]] = round(len(sa & sb) / len(sa | sb), 3)
    print('drift Frobenius medio:', round(sum(drift.values()) / len(drift), 4))
    print('solape readout entre fits (mismo n):',
          round(sum(solape_fits.values()) / len(solape_fits), 3))
"""),
    C("""
resultados = {'gate_solape_por_prompt': solape_gate, 'gate_medio': gate_medio,
              'refit_n': REFIT_N,
              'drift_frobenius': {str(k): v for k, v in drift.items()},
              'solape_entre_fits': solape_fits}
"""),
    guardar_resultados("exp0_lens_refit"),
    M("""
**Lectura del resultado.** El número operativo es `gate_medio`: ≥ ~0.7 ⇒ el
lens original sigue leyendo fiel sobre el modelo actualizado y las Ideas 1 y 3
pueden usarlo sin re-fit (comparalo con la curva sintética local: 0.989 @
eps=0.001 … 0.882 @ eps=0.05). El drift entre fits a igual n es evidencia
estructural complementaria (ruidosa con n chico).
"""),
]

# ===================================================== exp1: filtrado de dreams

exp1 = [
    M("""
# Exp1 — Filtrado de dreams por J-Space (Idea 1) · fase REM en GPU

Compara **4 condiciones de selección de dreams** para self-improvement
(knowledge incorporation, setup de la Tabla 3 del paper):

| condición | criterio |
|---|---|
| `none` | k+b dreams al azar |
| `grad` | top-k por norma de gradiente (∇L_SFT, el filtro del paper §3.4) |
| `jspace` | top-k por firma del workspace (leer_pizarron + detectar — lo nuestro) |
| `combinado` | promedio de rangos de ambos |

Por condición: adaptador LoRA nuevo → SFT sobre los dreams seleccionados →
accuracy de QA **sin contexto** + sondas de olvido. El bundle con los dreams ya
puntuados por J-Space se genera en la Mac con `experiments/exp1_prepare_dreams.py`
y se sube a Drive (`MyDrive/sleep_lab/bundles/`).
"""),
    SETUP_DEPS, SETUP_DRIVE, SETUP_MODEL,
    C("""
# Cargar el bundle (Drive primero; si no, el que venga dentro del zip).
# SLEEP_BUNDLE selecciona por glob (default rem_*.json); SLEEP_SEED controla
# la semilla de selección y de SFT (plan V1: semillas end-to-end).
import os
from pathlib import Path
from sleepharness.sleep.rem import cargar_bundle

SEED = int(os.environ.get('SLEEP_SEED', '0'))
PATRON = os.environ.get('SLEEP_BUNDLE', 'rem_*.json')
cands = sorted(Path('/content/resultados').glob(PATRON)) + \\
        sorted(Path('/content/drive/MyDrive/sleep_lab/bundles').glob(PATRON)) + \\
        sorted(Path(f'{LAB}/sleep-harness/bundles').glob(PATRON))
assert cands, f'No hay bundle que matchee {PATRON}'
BUNDLE_PATH = cands[0]
bundle = cargar_bundle(BUNDLE_PATH)
dreams, qa, sondas = bundle['_dreams'], bundle['qa'], bundle['sondas']
K = bundle['config']['dreaming']['top_k']
B = bundle['config']['dreaming']['b_random']
print(BUNDLE_PATH.name, '|', len(dreams), 'dreams |', len(qa), 'QA |', len(sondas), 'sondas')
"""),
    C("""
# Score de gradiente (el filtro del paper) — necesita un adaptador entrenable.
# CACHEADO en aux (con firma de semilla+bundle): los reintentos tras un
# reclaim de VM saltean estos ~5 min y el primer parcial llega mucho antes.
import json
from peft import LoraConfig, get_peft_model
from sleepharness.sleep.dream_filter import score_gradiente

AUX = f'{DRIVE_OUT}/exp1_aux.json'
aux = {}
try:
    aux = json.load(open(AUX))
    if aux.get('seed') != SEED or aux.get('bundle') != BUNDLE_PATH.name:
        aux = {}
except Exception:
    aux = {}

pm = get_peft_model(model, LoraConfig(**config.LORA), adapter_name='probe')
if len(aux.get('grad', [])) == len(dreams):
    for d, g in zip(dreams, aux['grad']):
        d.scores['gradiente'] = g
    print('grad scores desde aux (cacheados)')
else:
    for d in dreams:
        score_gradiente(pm, tok, d)
    aux = {'seed': SEED, 'bundle': BUNDLE_PATH.name,
           'grad': [d.scores['gradiente'] for d in dreams]}
    with open(AUX, 'w') as f:
        json.dump(aux, f)
print('ejemplo de scores:', dreams[0].scores)
"""),
    C("""
# Generación y evaluaciones baseline (modelo base, adaptadores desactivados)
import torch
from sleepharness.eval.incorporation import evaluar_incorporacion
from sleepharness.eval.forgetting import correr_sondas, comparar

def generar(prompt, adapter=None, max_new=48):
    # 48 tokens: las evaluaciones son respuestas de una línea; en T4 recorta
    # ~3x el tiempo de las ~110 generaciones de la corrida
    if adapter is not None:
        pm.set_adapter(adapter)
    import contextlib
    ctx = pm.disable_adapter() if adapter is None else contextlib.nullcontext()
    with ctx, torch.no_grad():
        ids = tok(prompt, return_tensors='pt').to('cuda')
        out = pm.generate(**ids, max_new_tokens=max_new, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)

if 'base_inc' in aux and 'base_son' in aux:
    base_inc = {'accuracy': aux['base_inc']}
    base_son = aux['base_son']
    print('baselines desde aux (cacheados)')
else:
    base_inc = evaluar_incorporacion(lambda p: generar(p), qa)
    base_son = correr_sondas(lambda p: generar(p), sondas)
    aux['base_inc'] = base_inc['accuracy']
    aux['base_son'] = base_son
    with open(AUX, 'w') as f:
        json.dump(aux, f)
print('baseline sin contexto:', base_inc['accuracy'],
      '| sondas:', sum(base_son.values()), '/', len(base_son))
"""),
    C("""
# Las 4 condiciones: seleccionar -> SFT-LoRA -> evaluar incorporación y olvido
# Con guardado PARCIAL por condición (la VM puede ser reclamada a mitad de
# corrida): si /content/resultados/exp1_partial.json existe, se reanuda.
import json, os
from sleepharness.sleep.dream_filter import seleccionar
from sleepharness.sleep.training import sft_lora

PARTIAL = f'{DRIVE_OUT}/exp1_partial.json'
resultados = {'bundle': BUNDLE_PATH.name, 'seed': SEED,
              'baseline': base_inc['accuracy'], 'condiciones': {}}
try:  # el parcial puede no existir o estar vacío (download de un remoto ausente)
    resultados['condiciones'] = json.load(open(PARTIAL)).get('condiciones', {})
    print('reanudando; ya hechas:', list(resultados['condiciones']))
except (FileNotFoundError, ValueError):
    pass
CONDS = os.environ.get('SLEEP_CONDS', 'none,grad,jspace,combinado').split(',')
for modo in CONDS:
    if modo in resultados['condiciones']:
        continue
    print(f'== condición {modo} ==')
    if modo not in pm.peft_config:
        pm.add_adapter(modo, LoraConfig(**config.LORA))
    pm.set_adapter(modo)
    for n, p in pm.named_parameters():
        p.requires_grad = f'.{modo}.' in n
    sel = seleccionar(dreams, k=K, b_random=B, modo=modo, seed=SEED)
    print('  dreams:', [round(d.scores.get('jspace', 0), 3) for d in sel])
    # presupuesto de SFT configurable por entorno (calibración sin editar código)
    sft_lora(pm, tok,
             [d.texto for d in sel] * int(os.environ.get('SLEEP_SFT_DUP', '1')),
             epochs=int(os.environ.get('SLEEP_SFT_EPOCHS', '3')),
             lr=float(os.environ.get('SLEEP_SFT_LR', '1e-4')),
             seed=SEED)
    inc = evaluar_incorporacion(lambda p: generar(p, modo), qa)
    son = correr_sondas(lambda p: generar(p, modo), sondas)
    olvido = comparar(base_son, son)
    resultados['condiciones'][modo] = {
        'n_seleccionados': len(sel),
        'incorporacion': inc['accuracy'],
        'delta_vs_base': round(inc['accuracy'] - base_inc['accuracy'], 4),
        'olvido_medio': olvido['olvido_medio'],
        'detalle_inc': inc['por_pregunta'],
    }
    with open(PARTIAL, 'w') as f:
        json.dump(resultados, f, ensure_ascii=False, default=str)
    torch.cuda.empty_cache()
    print(f"  incorporación: {inc['accuracy']:.3f} "
          f"(Δ{inc['accuracy'] - base_inc['accuracy']:+.3f}) | "
          f"olvido: {olvido['olvido_medio']:+.3f}")
"""),
    C("""
# Tabla final
print(f"{'condición':<11} {'incorp.':>8} {'Δ':>7} {'olvido':>7}")
for modo, r in resultados['condiciones'].items():
    print(f"{modo:<11} {r['incorporacion']:>8.3f} {r['delta_vs_base']:>+7.3f} "
          f"{r['olvido_medio']:>+7.3f}")
"""),
    guardar_resultados("exp1_dream_filter"),
    C("""
# OPCIONAL (lento): reward por dream aislado (Ec. 5) y correlación con el
# score J-Space — el resultado de investigación de la Idea 1.
RUN_PER_DREAM = False
if RUN_PER_DREAM:
    import numpy as np
    mejoras = []
    for i, d in enumerate(dreams):
        nombre = f'd{i}'
        pm.add_adapter(nombre, LoraConfig(**config.LORA))
        pm.set_adapter(nombre)
        for n, p in pm.named_parameters():
            p.requires_grad = f'.{nombre}.' in n
        sft_lora(pm, tok, [d.texto], epochs=2, lr=1e-4, log=lambda *_: None)
        inc = evaluar_incorporacion(lambda p: generar(p, nombre), qa)
        mejoras.append(inc['accuracy'] - base_inc['accuracy'])
        pm.delete_adapter(nombre)
        print(f'  dream {i}: jspace={d.scores.get("jspace", 0):.3f} '
              f'grad={d.scores.get("gradiente", 0):.1f} Δ={mejoras[-1]:+.3f}')
    js = np.array([d.scores.get('jspace', 0) for d in dreams])
    gr = np.array([d.scores.get('gradiente', 0) for d in dreams])
    me = np.array(mejoras)
    print('corr(jspace, mejora) =', np.corrcoef(js, me)[0, 1])
    print('corr(grad, mejora)   =', np.corrcoef(gr, me)[0, 1])
"""),
]

# ================================================ exp3: workspace distillation

exp3 = [
    M("""
# Exp3 — NREM: Knowledge Seeding ± Workspace Distillation (Idea 3) · GPU

Consolidación de memoria (§3.3 del paper) sobre los hechos de `facts_mini`:

- **Teacher** = el modelo CON el contexto en el prompt (memoria rápida)
- **Student** = el mismo modelo SIN contexto; solo el adaptador LoRA nuevo entrena
- **GKD**: mezcla λ on/off-policy con divergencia teacher∥student (Ec. de §3.3)
- **+ WS** (lo nuestro): término extra que alinea las distribuciones
  **lens-decodificadas del workspace** en capas medias-tardías — destila no solo
  qué responder sino qué conceptos encender.

Condiciones: `gkd` (solo logits, como el paper) vs `gkd_ws` (logits + workspace).
Métricas: incorporación sin contexto, olvido, y pares LTI (Ec. 3-4).
"""),
    SETUP_DEPS, SETUP_DRIVE, SETUP_MODEL, SETUP_LENS,
    C("""
# Dataset de Knowledge Seeding: el teacher (base + contexto) responde sondas
import json, torch
from sleepharness.sleep.nrem import construir_dataset_ks

task = json.load(open(f'{LAB}/sleep-harness/tasks/facts_mini.json'))

def generar_base(prompt, max_new=200):
    with torch.no_grad():
        ids = tok(prompt, return_tensors='pt').to('cuda')
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)

pares = []
for ctx in task['contextos']:
    pares += construir_dataset_ks(generar_base, ctx, tema=task['tema'])
print(len(pares), 'pares teacher/student')
print('ejemplo:', pares[0].respuesta_teacher[:150], '…')
"""),
    C("""
# Sub-vocabulario del workspace: palabras de contenido de los hechos
import re
from sleepharness.sleep.workspace_loss import sub_vocab

palabras = set()
for ctx in task['contextos']:
    for w in re.findall(r'[A-Za-zÁÉÍÓÚáéíóúñ]{4,}|\\d+', ctx):
        palabras.update({w.lower(), ' ' + w.lower(), w.capitalize(), ' ' + w.capitalize()})
VOCAB_IDS = sub_vocab(tok, sorted(palabras))
print(len(VOCAB_IDS), 'tokens en el sub-vocabulario del workspace')
"""),
    C("""
# Baselines sin contexto (adaptadores desactivados)
from peft import LoraConfig, get_peft_model
from sleepharness.eval.incorporation import evaluar_incorporacion
from sleepharness.eval.forgetting import correr_sondas, comparar

pm = get_peft_model(model, LoraConfig(**config.LORA), adapter_name='gkd')
sondas = json.load(open(f'{LAB}/sleep-harness/tasks/sondas_olvido.json'))['sondas']

def generar(prompt, adapter=None, max_new=120):
    import contextlib
    if adapter is not None:
        pm.set_adapter(adapter)
    ctx = pm.disable_adapter() if adapter is None else contextlib.nullcontext()
    with ctx, torch.no_grad():
        ids = tok(prompt, return_tensors='pt').to('cuda')
        out = pm.generate(**ids, max_new_tokens=max_new, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)

base_inc = evaluar_incorporacion(lambda p: generar(p), task['qa'])
base_son = correr_sondas(lambda p: generar(p), sondas)
print('baseline:', base_inc['accuracy'])
"""),
    C("""
# Condiciones: GKD solo-logits (paper) vs GKD + Workspace Distillation (Idea 3)
# Con guardado PARCIAL por condición (resiliencia a reclaims de la VM).
import os
from sleepharness.sleep.training import entrenar_ks

PARTIAL = f'{DRIVE_OUT}/exp3_partial.json'
resultados = {'seed': int(os.environ.get('SLEEP_SEED', '0')),
              'baseline': base_inc['accuracy'], 'condiciones': {}}
try:  # el parcial puede no existir o estar vacío (download de un remoto ausente)
    resultados['condiciones'] = json.load(open(PARTIAL)).get('condiciones', {})
    print('reanudando; ya hechas:', list(resultados['condiciones']))
except (FileNotFoundError, ValueError):
    pass
# Brazos (V6): las ablaciones que atribuyen el efecto al J-Space o lo refutan
# - gkd_ws_random: mismo término WS con sub-vocabulario ALEATORIO del mismo
#   tamaño -> si rinde igual que gkd_ws, el efecto es deep-supervision
#   genérica y no alineación de workspace.
# - ce_only: ¿aporta algo la divergencia GKD sobre CE pelado?
import random as _r
SEED = int(os.environ.get('SLEEP_SEED', '0'))
_rv = _r.Random(100 + SEED)
_pool_v = [i for i in range(len(tok)) if i not in set(VOCAB_IDS)]
VOCAB_RANDOM = sorted(_rv.sample(_pool_v, len(VOCAB_IDS)))
BRAZOS = {
    'gkd':           dict(peso_ws=0.0, peso_gkd=1.0, vocab=VOCAB_IDS),
    'gkd_ws':        dict(peso_ws=config.WORKSPACE_LOSS['peso'], peso_gkd=1.0, vocab=VOCAB_IDS),
    'gkd_ws_random': dict(peso_ws=config.WORKSPACE_LOSS['peso'], peso_gkd=1.0, vocab=VOCAB_RANDOM),
    'ce_only':       dict(peso_ws=0.0, peso_gkd=0.0, vocab=VOCAB_IDS),
}
ARMS = os.environ.get('SLEEP_ARMS', 'gkd,gkd_ws').split(',')
for nombre in ARMS:
    cfg_brazo = BRAZOS[nombre]
    if nombre in resultados['condiciones']:
        continue
    print(f'== brazo {nombre} (seed={SEED}) ==')
    if nombre not in pm.peft_config:
        pm.add_adapter(nombre, LoraConfig(**config.LORA))
    pm.set_adapter(nombre)
    for n, p in pm.named_parameters():
        p.requires_grad = f'.{nombre}.' in n
    hist = entrenar_ks(
        pm, tok, pares, lens=lens, capas=CAPAS, vocab_ids=cfg_brazo['vocab'],
        lam=config.GKD['lam'], divergencia=config.GKD['divergencia'],
        peso_ws=cfg_brazo['peso_ws'], peso_gkd=cfg_brazo['peso_gkd'],
        temperatura_ws=config.WORKSPACE_LOSS['temperatura'], seed=SEED,
        epochs=4, lr=2e-4)
    inc = evaluar_incorporacion(lambda p: generar(p, nombre), task['qa'])
    son = correr_sondas(lambda p: generar(p, nombre), sondas)
    olvido = comparar(base_son, son)
    resultados['condiciones'][nombre] = {
        'incorporacion': inc['accuracy'],
        'delta_vs_base': round(inc['accuracy'] - base_inc['accuracy'], 4),
        'olvido_medio': olvido['olvido_medio'],
        'loss_final_gkd': hist['gkd'][-1],
        'loss_final_ws': hist['ws'][-1] if hist['ws'] else None,
        'detalle_inc': inc['por_pregunta'],
    }
    with open(PARTIAL, 'w') as f:
        json.dump(resultados, f, ensure_ascii=False, default=str)
    print(f"  incorporación: {inc['accuracy']:.3f} "
          f"(Δ{inc['accuracy'] - base_inc['accuracy']:+.3f}) | "
          f"olvido: {olvido['olvido_medio']:+.3f}")
"""),
    C("""
# LTI (Ec. 3-4) + paso ReSTEM: el student aprende a IMITAR el sampling del
# teacher. Solo corre si el brazo gkd_ws se entrenó en esta corrida.
if 'gkd_ws' in ARMS and 'gkd_ws' in pm.peft_config:
    from sleepharness.sleep.nrem import lti_generar_pares, similitud_embeddings
    from sleepharness.sleep.training import sft_lora

    sem = similitud_embeddings(umbral=0.75)
    dreams_teacher = [p.respuesta_teacher for p in pares]
    lti = lti_generar_pares(
        dreams_teacher, lambda pref: generar(pref, 'gkd_ws', max_new=160),
        tokenizer=tok, sem_fn=sem,
        gamma=config.LTI['gamma'], z0=config.LTI['z0'],
        frac_prefijo=config.LTI['frac_prefijo'])
    rewards = [x['reward'] for x in lti]
    print(f'reward LTI medio: {sum(rewards) / len(rewards):.3f}')

    # ReSTEM = SFT filtrado por reward sobre las continuaciones del teacher
    buenos = [x for x in lti if x['reward'] >= 0.5]
    print(f'{len(buenos)}/{len(lti)} pares pasan el umbral')
    if buenos:
        pm.set_adapter('gkd_ws')
        for n, p in pm.named_parameters():
            p.requires_grad = '.gkd_ws.' in n
        sft_lora(pm, tok,
                 [x['prefijo'] + x['continuacion_teacher'] for x in buenos],
                 epochs=1, lr=5e-5, seed=SEED)
        inc2 = evaluar_incorporacion(lambda p: generar(p, 'gkd_ws'), task['qa'])
        resultados['condiciones']['gkd_ws']['incorporacion_post_lti'] = inc2['accuracy']
        print('incorporación tras LTI:', inc2['accuracy'])
    resultados['lti_rewards'] = rewards
else:
    print('LTI: salteado (gkd_ws no corrió en esta pasada)')
"""),
    C("""
# Tabla final
print(f"{'condición':<8} {'incorp.':>8} {'Δ':>7} {'olvido':>7}")
for nombre, r in resultados['condiciones'].items():
    print(f"{nombre:<8} {r['incorporacion']:>8.3f} {r['delta_vs_base']:>+7.3f} "
          f"{r['olvido_medio']:>+7.3f}")
"""),
    guardar_resultados("exp3_workspace_distill"),
    M("""
**Qué mirar.** Si `gkd_ws` incorpora igual o mejor que `gkd` con **menos
olvido**, la hipótesis de la Idea 3 (destilar el workspace transfiere el
*proceso*, no solo la salida) tiene soporte. Los adaptadores entrenados pueden
bajarse con `pm.save_pretrained('/content/drive/MyDrive/sleep_lab/adapters')`
para medir el drift del lens con `experiments/exp0_lens_drift.py --adapter` en
la Mac.
"""),
]


# ------------------------------------------------ runners para `colab exec -f`
# Mismo código que los notebooks, con el setup de VM en lugar del de Drive.
# Flujo: colab new --gpu … ; colab upload del zip ; unzip ; colab install de
# deps ; colab exec -f scripts/colab_runners/run_<exp>.py ; colab download.

SCRIPT_HEADER = '''\
# Generado por scripts/gen_notebooks.py — NO editar a mano.
# Pensado para `colab exec -f` sobre una VM con el bundle descomprimido en
# /content/lab (ver scripts/package_for_colab.sh) y las deps instaladas con:
#   colab install "transformers>=5.5" peft accelerate datasets rapidfuzz sentence-transformers
import os, sys

# antes de inicializar CUDA: mitiga la fragmentación en GPUs chicas (T4)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

LAB = os.environ.get('SLEEP_LAB', '/content/lab')
DRIVE_OUT = os.environ.get('SLEEP_OUT', '/content/resultados')
os.makedirs(DRIVE_OUT, exist_ok=True)
os.makedirs('/content/drive/MyDrive/sleep_lab/bundles', exist_ok=True)  # glob seguro
sys.path.insert(0, f'{LAB}/jacobian-lens')   # jlens sin instalar (deps ya presentes)
sys.path.insert(0, f'{LAB}/jlens-harness')
sys.path.insert(0, f'{LAB}/sleep-harness')
os.environ['SLEEPHARNESS_JLENS_HARNESS'] = f'{LAB}/jlens-harness'
import matplotlib
matplotlib.use('Agg')
'''

RUNNERS_OUT = Path(__file__).resolve().parents[1] / "scripts" / "colab_runners"


def emitir_runner(cells, path):
    partes = [SCRIPT_HEADER]
    for c in cells:
        if c is SETUP_DEPS or c is SETUP_DRIVE:
            continue  # deps: colab install; zip: colab upload — no magics acá
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            partes.append("\n".join("# " + l for l in src.split("\n")))
        else:
            partes.append(src)
    path.write_text("\n\n# %% ------------------------------------------\n\n".join(partes) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    RUNNERS_OUT.mkdir(exist_ok=True)
    libros = {
        "exp0_lens_refit": (exp0, "exp0 — lens refit"),
        "exp1_dream_filter": (exp1, "exp1 — dream filter J-Space"),
        "exp3_workspace_distill": (exp3, "exp3 — workspace distillation"),
    }
    for base, (cells, titulo) in libros.items():
        nb_path = OUT / f"colab_{base}.ipynb"
        nb_path.write_text(json.dumps(notebook(cells, titulo), ensure_ascii=False,
                                      indent=1))
        runner = RUNNERS_OUT / f"run_{base}.py"
        emitir_runner(cells, runner)
        print(f"{nb_path} ({len(cells)} celdas) + {runner}")


if __name__ == "__main__":
    main()
