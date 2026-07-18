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

# # Exp3 — NREM: Knowledge Seeding ± Workspace Distillation (Idea 3) · GPU
# 
# Consolidación de memoria (§3.3 del paper) sobre los hechos de `facts_mini`:
# 
# - **Teacher** = el modelo CON el contexto en el prompt (memoria rápida)
# - **Student** = el mismo modelo SIN contexto; solo el adaptador LoRA nuevo entrena
# - **GKD**: mezcla λ on/off-policy con divergencia teacher∥student (Ec. de §3.3)
# - **+ WS** (lo nuestro): término extra que alinea las distribuciones
#   **lens-decodificadas del workspace** en capas medias-tardías — destila no solo
#   qué responder sino qué conceptos encender.
# 
# Condiciones: `gkd` (solo logits, como el paper) vs `gkd_ws` (logits + workspace).
# Métricas: incorporación sin contexto, olvido, y pares LTI (Ec. 3-4).

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

# %% ------------------------------------------

# Sub-vocabulario del workspace: palabras de contenido de los hechos
import re
from sleepharness.sleep.workspace_loss import sub_vocab

palabras = set()
for ctx in task['contextos']:
    for w in re.findall(r'[A-Za-zÁÉÍÓÚáéíóúñ]{4,}|\d+', ctx):
        palabras.update({w.lower(), ' ' + w.lower(), w.capitalize(), ' ' + w.capitalize()})
VOCAB_IDS = sub_vocab(tok, sorted(palabras))
print(len(VOCAB_IDS), 'tokens en el sub-vocabulario del workspace')

# %% ------------------------------------------

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

# %% ------------------------------------------

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

# %% ------------------------------------------

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

# %% ------------------------------------------

# Tabla final
print(f"{'condición':<8} {'incorp.':>8} {'Δ':>7} {'olvido':>7}")
for nombre, r in resultados['condiciones'].items():
    print(f"{nombre:<8} {r['incorporacion']:>8.3f} {r['delta_vs_base']:>+7.3f} "
          f"{r['olvido_medio']:>+7.3f}")

# %% ------------------------------------------

import json, time
res_path = f"{DRIVE_OUT}/exp3_workspace_distill_{time.strftime('%Y%m%d_%H%M%S')}.json"
with open(res_path, 'w') as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
print('resultados →', res_path)

# %% ------------------------------------------

# **Qué mirar.** Si `gkd_ws` incorpora igual o mejor que `gkd` con **menos
# olvido**, la hipótesis de la Idea 3 (destilar el workspace transfiere el
# *proceso*, no solo la salida) tiene soporte. Los adaptadores entrenados pueden
# bajarse con `pm.save_pretrained('/content/drive/MyDrive/sleep_lab/adapters')`
# para medir el drift del lens con `experiments/exp0_lens_drift.py --adapter` en
# la Mac.
