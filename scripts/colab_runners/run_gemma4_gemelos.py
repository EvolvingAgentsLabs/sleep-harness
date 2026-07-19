# Paquete Gemma (1/2): test de GEMELOS en Gemma 4 E4B — réplica cross-modelo
# del veto semántico (pre-registrado: jspace(fiel) > jspace(corrupto), sign
# test sobre pares decididos). La corrupción es lexical-invariante por
# construcción; si Gemma también la ve, el veto queda validado en dos
# familias de modelos.
import os, sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
LAB = os.environ.get('SLEEP_LAB', '/content/lab')
os.makedirs('/content/resultados', exist_ok=True)
sys.path.insert(0, f'{LAB}/jacobian-lens')
sys.path.insert(0, f'{LAB}/jlens-harness')
sys.path.insert(0, f'{LAB}/sleep-harness')
os.environ['SLEEPHARNESS_JLENS_HARNESS'] = f'{LAB}/jlens-harness'

import json
from math import comb

import torch

assert torch.cuda.is_available()
from harness.runtime import Runtime
from sleepharness.sleep.dream_filter import anclas_de_contexto, puntuar_top_con_anclas

task = json.load(open(f'{LAB}/sleep-harness/tasks/facts_v2.json'))
pares = json.load(open(f'{LAB}/sleep-harness/bundles/gemelos_veneno.json'))

print(f'cargando gemma4e4b… ({len(pares)} pares gemelos)')
rt = Runtime('gemma4e4b', device='cuda')

wins = ties = losses = 0
for i, p in enumerate(pares):
    anclas = anclas_de_contexto(task['contextos'][p['contexto_idx']])
    s = {}
    for k in ('fiel', 'corrupto'):
        ws = rt.leer_pizarron(p[k], top_k=30, max_posiciones=20)
        s[k] = puntuar_top_con_anclas(ws.top, anclas)
    p['gemma_fiel'], p['gemma_corrupto'] = s['fiel'], s['corrupto']
    r = 'WIN' if s['fiel'] > s['corrupto'] else ('TIE' if s['fiel'] == s['corrupto'] else 'LOSS')
    wins += r == 'WIN'; ties += r == 'TIE'; losses += r == 'LOSS'
    print(f"  par {i}: fiel={s['fiel']:.3f} corrupto={s['corrupto']:.3f} {r}")

n = wins + losses
pval = sum(comb(n, k) for k in range(wins, n + 1)) / 2**n if n else 1.0
resultado = {'modelo': 'google/gemma-4-E4B', 'wins': wins, 'ties': ties,
             'losses': losses, 'sign_test_p': round(pval, 4), 'pares': pares}
with open('/content/resultados/gemma4_gemelos.json', 'w') as f:
    json.dump(resultado, f, ensure_ascii=False, indent=1)
print(f'\nGEMELOS EN GEMMA: {wins}W/{ties}T/{losses}L | sign test p={pval:.4f}')
print('VEREDICTO:', 'el veto REPLICA cross-modelo' if (n and wins > n * 0.75 and pval < 0.05)
      else 'sin evidencia suficiente en Gemma')
