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

# # Exp0 — Validez del Jacobian lens tras actualizar pesos (parte GPU)
# 
# Contraparte de `experiments/exp0_lens_drift.py` (local), con un **update
# real**. Dos niveles de evidencia:
# 
# 1. **Gate barato (minutos)**: readout del lens ORIGINAL sobre el modelo base
#    vs sobre el modelo con un LoRA real fusionado — ¿el pizarrón sigue
#    leyendo igual? Es la pregunta operativa para las Ideas 1 y 3.
# 2. **Re-fit chico opcional (horas)**: `jlens.fit` con REFIT_N prompts sobre
#    AMBOS modelos (mismo n, comparación justa) — drift estructural de J_l.
#    Ojo: cada prompt cuesta ~d_model/8 backwards (~320 en el 4B); N=200 en T4
#    serían ~40 h. REFIT_N=4 por defecto; subilo solo con L4/A100
#    (env SLEEP_REFIT_N).

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

import jlens
lens = jlens.JacobianLens.from_pretrained(
    config.LENS_REPO, filename=config.LENS_FILE, revision=config.LENS_REVISION)
desde = int(round(N_CAPAS * 18 / 64))
CAPAS = [l for l in lens.source_layers if desde <= l < N_CAPAS]
print('capas de workspace:', len(CAPAS), '(desde', desde, ')')

# %% ------------------------------------------

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

# %% ------------------------------------------

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

# %% ------------------------------------------

# 3) LoRA real sobre los hechos, fusionado al base (= "modelo actualizado")
from peft import LoraConfig, get_peft_model
from sleepharness.sleep.training import sft_lora

task = json.load(open(f'{LAB}/sleep-harness/tasks/facts_mini.json'))
pm = get_peft_model(model, LoraConfig(**config.LORA), adapter_name='exp0')
sft_lora(pm, tok, task['contextos'] * 4, epochs=2, lr=2e-4)
model = pm.merge_and_unload()
lm = jlens.from_hf(model, tok)   # wrapper sobre el modelo fusionado
print('LoRA fusionado; parámetros del base actualizados')

# %% ------------------------------------------

# 4) GATE BARATO: el lens ORIGINAL sobre el modelo actualizado
solape_gate = {}
for p in CALIB:
    b = top_ids(lens, p)
    a = base_tops[p]
    solape_gate[p[:50]] = round(len(a & b) / len(a | b), 3)
    print(f"  {solape_gate[p[:50]]:.3f}  {p[:70]}")
gate_medio = round(sum(solape_gate.values()) / len(solape_gate), 4)
print('GATE (jaccard medio, lens original base vs actualizado):', gate_medio)

# %% ------------------------------------------

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

# %% ------------------------------------------

resultados = {'gate_solape_por_prompt': solape_gate, 'gate_medio': gate_medio,
              'refit_n': REFIT_N,
              'drift_frobenius': {str(k): v for k, v in drift.items()},
              'solape_entre_fits': solape_fits}

# %% ------------------------------------------

import json, time
res_path = f"{DRIVE_OUT}/exp0_lens_refit_{time.strftime('%Y%m%d_%H%M%S')}.json"
with open(res_path, 'w') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
print('resultados →', res_path)

# %% ------------------------------------------

# **Lectura del resultado.** El número operativo es `gate_medio`: ≥ ~0.7 ⇒ el
# lens original sigue leyendo fiel sobre el modelo actualizado y las Ideas 1 y 3
# pueden usarlo sin re-fit (comparalo con la curva sintética local: 0.989 @
# eps=0.001 … 0.882 @ eps=0.05). El drift entre fits a igual n es evidencia
# estructural complementaria (ruidosa con n chico).
