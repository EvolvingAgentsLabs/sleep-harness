# P0 diagnóstico: ¿ausencia o calibración? Vuelca el top COMPLETO del pizarrón
# de Gemma sobre payloads maliciosos, escaneando 3 ventanas de capas (temprana/
# media/tardía). Si algún concepto de seguridad (en cualquier forma/idioma)
# aflora en malicioso pero no en benigno → calibración fixable. Si el top es
# genérico en todas las ventanas → límite real. Inference-only.
import os, sys, json

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
LAB = os.environ.get('SLEEP_LAB', '/content/lab')
os.makedirs('/content/resultados', exist_ok=True)
sys.path.insert(0, f'{LAB}/jacobian-lens')
sys.path.insert(0, f'{LAB}/jlens-harness')
sys.path.insert(0, f'{LAB}/sleep-harness')
os.environ['SLEEPHARNESS_JLENS_HARNESS'] = f'{LAB}/jlens-harness'

import torch
from harness.runtime import Runtime

print('cargando gemma4e4b…')
rt = Runtime('gemma4e4b', device='cuda')
N = rt.model.n_layers
src = sorted(rt.lens.source_layers)
def ventana(lo, hi):
    return [l for l in src if int(N*lo) <= l < int(N*hi)]
VENTANAS = {'temprana': ventana(0.10, 0.35), 'media': ventana(0.35, 0.65),
            'tardia': ventana(0.65, 0.95)}
print('n_layers', N, '| ventanas:', {k: (v[0], v[-1]) for k, v in VENTANAS.items() if v})

task = json.load(open(f'{LAB}/sleep-harness/tasks/security_prompts.json'))
# 3 pares de exfiltración de credenciales (los más claros)
pares = [p for p in task['pares'] if p['goal'] == 'credential_exfiltration'][:3]

dump = []
for i, p in enumerate(pares):
    print(f'\n===== PAR {i} (credential exfiltration) =====')
    entrada = {'idx': i, 'ventanas': {}}
    for vn, capas in VENTANAS.items():
        if not capas:
            continue
        rt.capas = capas
        wm = rt.leer_pizarron(p['malicious'], top_k=20, max_posiciones=20)
        wb = rt.leer_pizarron(p['benign_matched'], top_k=20, max_posiciones=20)
        tm = [t['token'] for t in wm.top]
        tb = [t['token'] for t in wb.top]
        # conceptos que aparecen en malicioso y NO en benigno (los discriminantes)
        solo_mal = [t for t in tm if t not in tb]
        entrada['ventanas'][vn] = {'malicious_top': tm, 'benign_top': tb,
                                   'solo_en_malicious': solo_mal}
        print(f'  [{vn}] MAL: {tm[:12]}')
        print(f'  [{vn}] BEN: {tb[:12]}')
        print(f'  [{vn}] SOLO EN MAL: {solo_mal[:10]}')
    dump.append(entrada)

out = '/content/resultados/gemma_diagnostico.json'
json.dump(dump, open(out, 'w'), ensure_ascii=False, indent=2)
print(f'\nguardado en {out}')
print('LECTURA: si "solo_en_malicious" trae conceptos de seguridad (clave, llave, '
      'token, ssh, secret, contraseña…) en alguna ventana → CALIBRACIÓN; '
      'si es todo genérico → ausencia real.')
