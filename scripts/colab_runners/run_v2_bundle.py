# Plan V1/V2 — Generación del bundle de dreams de facts_v2 EN GPU.
# En la Mac (MPS) los 72 dreams tardarían ~6 h; en A100, ~20-30 min.
# Produce /content/resultados/rem_facts_v2_seed<G>.json (lo baja el watcher
# o el driver). Requiere el lab desplegado en /content/lab.
import os, sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
LAB = os.environ.get('SLEEP_LAB', '/content/lab')
os.makedirs('/content/resultados', exist_ok=True)
sys.path.insert(0, f'{LAB}/jacobian-lens')
sys.path.insert(0, f'{LAB}/jlens-harness')
sys.path.insert(0, f'{LAB}/sleep-harness')
os.environ['SLEEPHARNESS_JLENS_HARNESS'] = f'{LAB}/jlens-harness'

import json
import torch

assert torch.cuda.is_available()
GEN_SEED = int(os.environ.get('SLEEP_GEN_SEED', '0'))
M = int(os.environ.get('SLEEP_M', '8'))            # dreams por contexto
TOP_K = int(os.environ.get('SLEEP_TOP_K', '8'))    # selección por condición
B_RANDOM = int(os.environ.get('SLEEP_B_RANDOM', '2'))

from harness.runtime import Runtime
from sleepharness import config
from sleepharness.sleep.rem import preparar_bundle_rem

task = json.load(open(f'{LAB}/sleep-harness/tasks/facts_v2.json'))
sondas = json.load(open(f'{LAB}/sleep-harness/tasks/sondas_olvido.json'))['sondas']

STEER_POOL = [
    " water", " energy", " ceramic", " titanium", " membrane", " ocean",
    " music", " history", " chemistry", " market", " engineering", " glass",
    " telescope", " comet", " enzyme", " plastic", " train", " magnet",
]

print(f'cargando Runtime en cuda… (m={M}, gen_seed={GEN_SEED})')
rt = Runtime(config.MODEL_KEY, device='cuda')

bundle = preparar_bundle_rem(
    rt, task['contextos'], qa=task['qa'], sondas=sondas,
    firma_objetivo=task.get('firma_objetivo', 'fidelidad'),
    steer_pool=STEER_POOL, m=M, seed=GEN_SEED,
    notas=f"facts_v2 en GPU; m={M}; gen_seed={GEN_SEED}",
)
# la selección en V2 escala con el pool: top_k y b_random propios
bundle['config']['dreaming']['top_k'] = TOP_K
bundle['config']['dreaming']['b_random'] = B_RANDOM

out = f'/content/resultados/rem_facts_v2_seed{GEN_SEED}.json'
with open(out, 'w') as f:
    json.dump({k: v for k, v in bundle.items() if not k.startswith('_')},
              f, ensure_ascii=False)
n = len(bundle['dreams'])
con = sum(1 for d in bundle['dreams'] if d['scores'].get('jspace', 0) > 0)
print(f'{n} dreams ({con} encienden fidelidad) → {out}')
