# Test multi-ciclo de olvido (pre-registro en PREREG_MULTICICLO.md).
# 4 ciclos continual, un dominio por ciclo. Brazo A = expansion (LoRA nuevo
# por ciclo, previos congelados; SFT identico a B). Brazo B = SFT naive single
# adapter. Mide retención de dominios previos + sondas generales por ciclo.
# Resiliente: guarda partial por (brazo).
import os, sys, contextlib

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
LAB = os.environ.get('SLEEP_LAB', '/content/lab')
os.makedirs('/content/resultados', exist_ok=True)
sys.path.insert(0, f'{LAB}/jacobian-lens')
sys.path.insert(0, f'{LAB}/jlens-harness')
sys.path.insert(0, f'{LAB}/sleep-harness')
os.environ['SLEEPHARNESS_JLENS_HARNESS'] = f'{LAB}/jlens-harness'

import json
import torch
import transformers
from peft import LoraConfig, get_peft_model

from sleepharness import config
from sleepharness.memory.lora_stack import LoraStack
from sleepharness.sleep.nrem import construir_dataset_ks
from sleepharness.sleep.training import entrenar_ks, sft_lora
from sleepharness.eval.incorporation import evaluar_incorporacion
from sleepharness.eval.forgetting import correr_sondas, comparar

PARTIAL = '/content/resultados/multiciclo_partial.json'
try:
    RES = json.load(open(PARTIAL))
except Exception:
    RES = {'brazos': {}}

task = json.load(open(f'{LAB}/sleep-harness/tasks/facts_v2.json'))
sondas = json.load(open(f'{LAB}/sleep-harness/tasks/sondas_olvido.json'))['sondas']

# --- partición en dominios (contextos + QA por keywords) ---
DOMINIOS = [
    ('vantar',   [0, 1, 2], ['vantar', 'heliox', 'ferreyra', 'rosario', 'fjellkap',
                             'antofagasta', 'tn-la9', 'membrana', 'desalinizado', 'serie b']),
    ('quelmara', [3, 4],    ['quelmara', 'ilun', 'vidarte', 'catamarca', 'verval',
                             'nis-q', 'astrala', 'observatorio', 'telescopio', 'cometa', 'espejo']),
    ('nordina',  [5, 6],    ['nordina', 'katalix', 'ereño', 'montevideo', 'pet',
                             'vaskio', 'nb-2214', 'enzima', 'papelera']),
    ('ferrovex', [7, 8],    ['ferrovex', 'auriga', 'vektra', 'mendoza', 'san juan',
                             'maglev', 'vagones', 'cuyo', 'tren', 'guiado']),
]
def qa_de(kws):
    out = []
    for q in task['qa']:
        ql = q['pregunta'].lower()
        if any(k in ql for k in kws):
            out.append(q)
    return out
DOM_QA = {n: qa_de(kws) for n, ix, kws in DOMINIOS}
DOM_CTX = {n: [task['contextos'][i] for i in ix] for n, ix, kws in DOMINIOS}
print('QA por dominio:', {n: len(v) for n, v in DOM_QA.items()})


def cargar_modelo():
    tok = transformers.AutoTokenizer.from_pretrained(config.HF_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.HF_MODEL, dtype=torch.bfloat16, device_map='cuda')
    model.config.pad_token_id = tok.eos_token_id
    return tok, model


def hacer_generar(pm, tok):
    def generar(prompt, con_adaptadores=True, max_new=48):
        ctx = contextlib.nullcontext() if con_adaptadores else pm.disable_adapter()
        with ctx, torch.no_grad():
            ids = tok(prompt, return_tensors='pt').to('cuda')
            out = pm.generate(**ids, max_new_tokens=max_new, do_sample=False,
                              pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
    return generar


def evaluar_ciclo(generar, hasta_dom, base_son):
    """Incorporación de cada dominio 0..hasta_dom + olvido general."""
    inc = {}
    for j in range(hasta_dom + 1):
        nombre = DOMINIOS[j][0]
        inc[nombre] = evaluar_incorporacion(lambda p: generar(p), DOM_QA[nombre])['accuracy']
    son = correr_sondas(lambda p: generar(p), sondas)
    olv = comparar(base_son, son)['olvido_medio']
    return {'incorporacion': inc, 'olvido_general': olv}


# ---------- Brazo A: Expanding KS ----------
if 'A_expanding_ks' not in RES['brazos']:
    print('\n===== BRAZO A: Expanding KS (mecanismo del paper) =====')
    tok, model = cargar_modelo()
    stack = LoraStack(model, config.LORA)          # crea sleep_0
    pm = stack.model
    generar = hacer_generar(pm, tok)
    base_son = correr_sondas(lambda p: generar(p, con_adaptadores=False), sondas)
    ciclos = []
    for c, (nombre, ix, kws) in enumerate(DOMINIOS):
        print(f'-- ciclo {c}: {nombre} --')
        if c > 0:
            stack.nuevo_adaptador()                # expansión: LoRA nuevo, previos congelados
        # expansión: SFT del adaptador NUEVO sobre el dominio del ciclo; previos
        # congelados y activos en el forward (retención por construcción, salvo
        # interferencia entre adaptadores apilados — que es lo que medimos)
        sft_lora(pm, tok, DOM_CTX[nombre] * 4, epochs=2, lr=2e-4, seed=0)
        pm.base_model.set_adapter(stack.activos)   # forward con todo el stack
        r = evaluar_ciclo(generar, c, base_son)
        r['ciclo'] = c; r['dominio'] = nombre; r['n_adaptadores'] = len(stack.activos)
        ciclos.append(r)
        print(f"   inc={r['incorporacion']} olvido_gen={r['olvido_general']:+.2f}")
    # commit atómico: solo se guarda el brazo COMPLETO (resume a nivel brazo)
    RES['brazos']['A_expanding_ks'] = ciclos
    json.dump(RES, open(PARTIAL, 'w'), ensure_ascii=False, indent=1)
    del model, pm, stack; torch.cuda.empty_cache()

# ---------- Brazo B: Naive continual SFT ----------
if 'B_naive_sft' not in RES['brazos']:
    print('\n===== BRAZO B: Naive continual SFT =====')
    tok, model = cargar_modelo()
    pm = get_peft_model(model, LoraConfig(**config.LORA), adapter_name='naive')
    generar = hacer_generar(pm, tok)
    base_son = correr_sondas(lambda p: generar(p, con_adaptadores=False), sondas)
    ciclos = []
    for c, (nombre, ix, kws) in enumerate(DOMINIOS):
        print(f'-- ciclo {c}: {nombre} --')
        sft_lora(pm, tok, DOM_CTX[nombre] * 4, epochs=2, lr=2e-4, seed=0)
        r = evaluar_ciclo(generar, c, base_son)
        r['ciclo'] = c; r['dominio'] = nombre
        ciclos.append(r)
        print(f"   inc={r['incorporacion']} olvido_gen={r['olvido_general']:+.2f}")
    RES['brazos']['B_naive_sft'] = ciclos
    json.dump(RES, open(PARTIAL, 'w'), ensure_ascii=False, indent=1)

# ---------- veredicto ----------
A = RES['brazos'].get('A_expanding_ks'); B = RES['brazos'].get('B_naive_sft')
if A and B:
    print('\n===== RETENCIÓN DEL DOMINIO 0 (vantar) POR CICLO =====')
    for arm, name in [(A, 'A_exp_ks'), (B, 'B_naive')]:
        traj = [round(cyc['incorporacion'].get('vantar', 0), 3) for cyc in arm]
        olv = [round(cyc['olvido_general'], 2) for cyc in arm]
        print(f'  {name}: vantar={traj}  olvido_gen={olv}')
    a_fin = A[-1]['incorporacion'].get('vantar', 0)
    b_fin = B[-1]['incorporacion'].get('vantar', 0)
    print(f'\nVEREDICTO pre-registrado: retención vantar tras ciclo 3 → '
          f'A={a_fin:.3f} vs B={b_fin:.3f} → '
          f'{"A retiene mejor (consistente con tesis continual)" if a_fin > b_fin else "sin ventaja de A"}')
print('\nMULTICICLO COMPLETO')
