# Smoke de compatibilidad Gemma 4 E4B — corre en una sesión propia (L4),
# en paralelo a la matriz de Qwen. Des-riesga la réplica cross-modelo:
# ¿el lens aplica? ¿el pizarrón lee coherente? ¿la fidelidad discrimina con
# el tokenizer de Gemma? ¿la generación es sana con este decoding?
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
resultado = {"modelo": "google/gemma-4-E4B", "checks": {}}

from harness.runtime import Runtime
from sleepharness.sleep.dream_filter import (anclas_de_contexto,
                                             puntuar_top_con_anclas,
                                             score_lexico)
from sleepharness.sleep.dreams import Dream, generar_dreams, es_degenerado
from sleepharness.signatures_ext import detectar_ext

print('cargando gemma4e4b en cuda…')
rt = Runtime('gemma4e4b', device='cuda')
resultado['checks']['carga'] = {
    'n_layers': rt.model.n_layers,
    'capas_workspace': len(rt.capas),
    'lens_layers': len(rt.lens.jacobians),
}
print('OK carga:', resultado['checks']['carga'])

# 1) Readout de calibración: ¿el pizarrón lee conceptos coherentes?
CALIB = [
    'La membrana cerámica opera a 41 grados y reduce el consumo energético un 37 por ciento.',
    'El precio óptimo se obtiene maximizando la ganancia G(x) = (30 - 5x)(100 + 20x).',
    'The observatory discovered a new comet using its infrared camera at high altitude.',
]
lecturas = []
for p in CALIB:
    ws = rt.leer_pizarron(p, top_k=15, max_posiciones=15)
    firmas = detectar_ext(ws.top)
    lecturas.append({'prompt': p[:60], 'top': [t['token'] for t in ws.top[:10]],
                     'dominante': firmas['_dominante']})
    print(f"  [{firmas['_dominante']}] {p[:50]}… -> {[t['token'] for t in ws.top[:8]]}")
resultado['checks']['readout'] = lecturas

# 2) Fidelidad: ¿separa fiel de alucinado con el tokenizer de Gemma?
task = json.load(open(f'{LAB}/sleep-harness/tasks/facts_v2.json'))
ctx = task['contextos'][0]
fiel = Dream('La ingeniera Lucía Ferreyra fundó Vantar Dynamics en Rosario en 2019; '
             'su membrana Heliox opera a 41 grados y ahorra 37 por ciento de energía.',
             'qa', 0)
aluc = Dream('El observatorio austral registró tormentas magnéticas durante el '
             'concierto sinfónico de la temporada pasada en la costa.', 'qa', 0)
anclas = anclas_de_contexto(ctx)
scores = {}
for nombre, d in [('fiel', fiel), ('alucinado', aluc)]:
    ws = rt.leer_pizarron(d.texto, top_k=30, max_posiciones=20)
    scores[nombre] = {
        'jspace': puntuar_top_con_anclas(ws.top, anclas),
        'lexical': score_lexico(d, ctx),
        'top': [t['token'] for t in ws.top[:8]],
    }
    print(f"  {nombre}: jspace={scores[nombre]['jspace']} lexical={scores[nombre]['lexical']}")
resultado['checks']['fidelidad'] = scores
resultado['checks']['fidelidad']['separa'] = (
    scores['fiel']['jspace'] > scores['alucinado']['jspace'])

# 3) Generación: ¿el decoding produce texto sano? ¿baseline QA en cero?
from sleepharness.eval.incorporation import evaluar_incorporacion

def gen(prompt, max_new=48):
    r = rt.step(prompt, max_new_tokens=max_new, leer_salida=False)
    return r.salida

base = evaluar_incorporacion(gen, task['qa'][:6])
print('baseline QA (6 preguntas, esperado ~0):', base['accuracy'])
resultado['checks']['baseline_qa'] = base['accuracy']

dreams = generar_dreams(rt, task['contextos'][:2], m=2, max_new_tokens=160, seed=0)
resultado['checks']['dreams'] = {
    'generados': len(dreams),
    'degenerados': sum(1 for d in dreams if es_degenerado(d.texto)),
    'muestras': [d.texto[:120] for d in dreams[:2]],
}
print(f"dreams: {len(dreams)} generados, "
      f"{resultado['checks']['dreams']['degenerados']} degenerados")
for d in dreams[:2]:
    print('  >', d.texto[:100].replace(chr(10), ' '))

with open('/content/resultados/gemma4_smoke.json', 'w') as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2)
print('\nSMOKE COMPLETO -> /content/resultados/gemma4_smoke.json')
print('veredicto: fidelidad separa =', resultado['checks']['fidelidad']['separa'],
      '| baseline QA =', resultado['checks']['baseline_qa'])
