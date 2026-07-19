# Paquete Gemma (2/2): contraste de OLVIDO en Gemma 4 E4B — ¿replica el
# invariante "consolidación KS = olvido ~0 vs SFT directo = olvido > 0"?
# 2 brazos (sft_directo, ks_gkd) × 2 semillas, con parcial por brazo.
import os, sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
LAB = os.environ.get('SLEEP_LAB', '/content/lab')
os.makedirs('/content/resultados', exist_ok=True)
sys.path.insert(0, f'{LAB}/jacobian-lens')
sys.path.insert(0, f'{LAB}/jlens-harness')
sys.path.insert(0, f'{LAB}/sleep-harness')
os.environ['SLEEPHARNESS_JLENS_HARNESS'] = f'{LAB}/jlens-harness'

import contextlib
import json

import torch
import transformers

assert torch.cuda.is_available()
from peft import LoraConfig, get_peft_model
from sleepharness import config
from sleepharness.eval.forgetting import comparar, correr_sondas
from sleepharness.eval.incorporation import evaluar_incorporacion
from sleepharness.sleep.nrem import construir_dataset_ks
from sleepharness.sleep.training import entrenar_ks, sft_lora

PARTIAL = '/content/resultados/gemma4_olvido.json'
try:
    resultado = json.load(open(PARTIAL))
except Exception:
    resultado = {'modelo': 'google/gemma-4-E4B', 'brazos': {}}

tok = transformers.AutoTokenizer.from_pretrained('google/gemma-4-E4B')
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = transformers.AutoModelForCausalLM.from_pretrained(
    'google/gemma-4-E4B', dtype=torch.bfloat16, device_map='cuda')
model.config.pad_token_id = tok.eos_token_id

task = json.load(open(f'{LAB}/sleep-harness/tasks/facts_mini.json'))
sondas = json.load(open(f'{LAB}/sleep-harness/tasks/sondas_olvido.json'))['sondas']

pm = get_peft_model(model, LoraConfig(**config.LORA), adapter_name='base_probe')

def generar(prompt, adapter=None, max_new=48):
    if adapter is not None:
        pm.set_adapter(adapter)
    ctx = pm.disable_adapter() if adapter is None else contextlib.nullcontext()
    with ctx, torch.no_grad():
        ids = tok(prompt, return_tensors='pt').to('cuda')
        out = pm.generate(**ids, max_new_tokens=max_new, do_sample=False,
                          repetition_penalty=1.3, no_repeat_ngram_size=4,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)

base_inc = evaluar_incorporacion(lambda p: generar(p), task['qa'])
base_son = correr_sondas(lambda p: generar(p), sondas)
resultado['baseline'] = {'inc': base_inc['accuracy'],
                         'sondas': sum(base_son.values()) / len(base_son)}
print('baseline: inc', base_inc['accuracy'], '| sondas', resultado['baseline']['sondas'])

pares_ks = None
for seed in (0, 1):
    for brazo in ('sft_directo', 'ks_gkd'):
        nombre = f'{brazo}_s{seed}'
        if nombre in resultado['brazos']:
            print(f'{nombre}: ya hecho, salteando')
            continue
        print(f'== {nombre} ==')
        if nombre not in pm.peft_config:
            pm.add_adapter(nombre, LoraConfig(**config.LORA))
        pm.set_adapter(nombre)
        for n, p in pm.named_parameters():
            p.requires_grad = f'.{nombre}.' in n
        if brazo == 'sft_directo':
            sft_lora(pm, tok, task['contextos'] * 4, epochs=2, lr=2e-4, seed=seed)
        else:
            if pares_ks is None:
                pares_ks = []
                for ctx_t in task['contextos']:
                    with pm.disable_adapter():
                        pares_ks += construir_dataset_ks(
                            lambda p: generar(p, None, max_new=200), ctx_t,
                            tema=task['tema'])
                print(f'  dataset KS: {len(pares_ks)} pares')
            entrenar_ks(pm, tok, pares_ks, peso_ws=0.0, epochs=4, lr=2e-4,
                        seed=seed)
        inc = evaluar_incorporacion(lambda p: generar(p, nombre), task['qa'])
        son = correr_sondas(lambda p: generar(p, nombre), sondas)
        olv = comparar(base_son, son)
        resultado['brazos'][nombre] = {
            'inc': inc['accuracy'],
            'delta_inc': round(inc['accuracy'] - base_inc['accuracy'], 4),
            'olvido': olv['olvido_medio'],
        }
        with open(PARTIAL, 'w') as f:
            json.dump(resultado, f, ensure_ascii=False, indent=1)
        print(f"  inc={inc['accuracy']:.3f} olvido={olv['olvido_medio']:+.3f}")
        torch.cuda.empty_cache()

print('\nCONTRASTE DE OLVIDO EN GEMMA:')
for n, r in resultado['brazos'].items():
    print(f"  {n:16s} inc={r['inc']:.3f} olvido={r['olvido']:+.3f}")
