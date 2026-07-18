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


# %% ------------------------------------------

# # Exp1 — Filtrado de dreams por J-Space (Idea 1) · fase REM en GPU
# 
# Compara **4 condiciones de selección de dreams** para self-improvement
# (knowledge incorporation, setup de la Tabla 3 del paper):
# 
# | condición | criterio |
# |---|---|
# | `none` | k+b dreams al azar |
# | `grad` | top-k por norma de gradiente (∇L_SFT, el filtro del paper §3.4) |
# | `jspace` | top-k por firma del workspace (leer_pizarron + detectar — lo nuestro) |
# | `combinado` | promedio de rangos de ambos |
# 
# Por condición: adaptador LoRA nuevo → SFT sobre los dreams seleccionados →
# accuracy de QA **sin contexto** + sondas de olvido. El bundle con los dreams ya
# puntuados por J-Space se genera en la Mac con `experiments/exp1_prepare_dreams.py`
# y se sube a Drive (`MyDrive/sleep_lab/bundles/`).

# %% ------------------------------------------

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

# %% ------------------------------------------

# Cargar el bundle (Drive primero; si no, el que venga dentro del zip).
# SLEEP_BUNDLE selecciona por glob (default rem_*.json); SLEEP_SEED controla
# la semilla de selección y de SFT (plan V1: semillas end-to-end).
import os
from pathlib import Path
from sleepharness.sleep.rem import cargar_bundle

SEED = int(os.environ.get('SLEEP_SEED', '0'))
PATRON = os.environ.get('SLEEP_BUNDLE', 'rem_*.json')
cands = sorted(Path('/content/resultados').glob(PATRON)) + \
        sorted(Path('/content/drive/MyDrive/sleep_lab/bundles').glob(PATRON)) + \
        sorted(Path(f'{LAB}/sleep-harness/bundles').glob(PATRON))
assert cands, f'No hay bundle que matchee {PATRON}'
BUNDLE_PATH = cands[0]
bundle = cargar_bundle(BUNDLE_PATH)
dreams, qa, sondas = bundle['_dreams'], bundle['qa'], bundle['sondas']
K = bundle['config']['dreaming']['top_k']
B = bundle['config']['dreaming']['b_random']
print(BUNDLE_PATH.name, '|', len(dreams), 'dreams |', len(qa), 'QA |', len(sondas), 'sondas')

# %% ------------------------------------------

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

# %% ------------------------------------------

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

# %% ------------------------------------------

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

# %% ------------------------------------------

# Tabla final
print(f"{'condición':<11} {'incorp.':>8} {'Δ':>7} {'olvido':>7}")
for modo, r in resultados['condiciones'].items():
    print(f"{modo:<11} {r['incorporacion']:>8.3f} {r['delta_vs_base']:>+7.3f} "
          f"{r['olvido_medio']:>+7.3f}")

# %% ------------------------------------------

import json, time
res_path = f"{DRIVE_OUT}/exp1_dream_filter_{time.strftime('%Y%m%d_%H%M%S')}.json"
with open(res_path, 'w') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
print('resultados →', res_path)

# %% ------------------------------------------

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
