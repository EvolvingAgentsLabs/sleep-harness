# Exp6 (H-SEC-3) — Trojan scanner de LoRAs de terceros vía lens-diff (Finding 1).
# Entrena dos LoRAs gemelos (trojan: asocia código con leer clave SSH; benigno:
# con telemetría inocua). Luego escanea: sobre prompts de CONTROL inocentes,
# ¿montar cada adapter desvía los conceptos de seguridad del pizarrón vs el base?
# Un trojan enciende amenaza donde no debería; un benigno no.
import os, sys, json, contextlib

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
LAB = os.environ.get('SLEEP_LAB', '/content/lab')
os.makedirs('/content/resultados', exist_ok=True)
sys.path.insert(0, f'{LAB}/jacobian-lens')
sys.path.insert(0, f'{LAB}/jlens-harness')
sys.path.insert(0, f'{LAB}/sleep-harness')
os.environ['SLEEPHARNESS_JLENS_HARNESS'] = f'{LAB}/jlens-harness'

import torch
from math import comb
from peft import LoraConfig, get_peft_model
from sleepharness import config
from sleepharness.security import score_seguridad
from sleepharness.sleep.training import sft_lora

PARTIAL = '/content/resultados/exp6_trojan.json'
try:
    RES = json.load(open(PARTIAL))
except Exception:
    RES = {}

import transformers
task = json.load(open(f'{LAB}/sleep-harness/tasks/trojan_train.json'))
tok = transformers.AutoTokenizer.from_pretrained(config.HF_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = transformers.AutoModelForCausalLM.from_pretrained(
    config.HF_MODEL, dtype=torch.bfloat16, device_map='cuda')
model.config.pad_token_id = tok.eos_token_id

import jlens
lm = jlens.from_hf(model, tok)
lens = jlens.JacobianLens.from_pretrained(config.LENS_REPO, filename=config.LENS_FILE,
                                          revision=config.LENS_REVISION)
N = model.config.num_hidden_layers
desde = int(round(N * 18 / 64))
CAPAS = [l for l in lens.source_layers if desde <= l < N]

pm = get_peft_model(model, LoraConfig(**config.LORA), adapter_name='trojan')
pm.add_adapter('benign', LoraConfig(**config.LORA))


def sec_score(texto, adapter):
    """malicious_intent del pizarrón de `texto` con `adapter` montado
    ('base' = sin adapters)."""
    ctx = pm.disable_adapter() if adapter == 'base' else contextlib.nullcontext()
    if adapter != 'base':
        pm.set_adapter(adapter)
    from jlens import JacobianLens  # noqa
    ids = lm.encode(texto, max_length=512)
    seq = ids.shape[1]
    pos = list(range(max(16, seq - 20), seq))
    with ctx, torch.no_grad():
        logits, _, _ = lens.apply(lm, texto, layers=CAPAS, positions=pos, max_seq_len=512)
    acc = {}
    for lg in logits.values():
        v, i = lg.topk(25, dim=-1)
        for fv, fi in zip(v, i):
            for vv, ii in zip(fv.tolist(), fi.tolist()):
                acc[ii] = acc.get(ii, 0.0) + vv
    import re
    top = []
    for tid, sc in sorted(acc.items(), key=lambda kv: -kv[1])[:60]:
        t = tok.decode([tid]).strip()
        if len(t) >= 3 and re.fullmatch(r"[A-Za-zÁÉÍÓÚáéíóúñ]+", t):
            top.append({"token": t, "intensidad": sc})
        if len(top) >= 30:
            break
    return score_seguridad(top)["malicious_intent"]


# entrenar los dos LoRAs (solo el que falte, resiliencia)
for arm, campo in [('trojan', 'trojan'), ('benign', 'benign')]:
    if f'{arm}_trained' in RES:
        continue
    print(f'== entrenando LoRA {arm} ==')
    pm.set_adapter(arm)
    for n, p in pm.named_parameters():
        p.requires_grad = f'.{arm}.' in n
    textos = [d['prompt'] + d[campo] for d in task['train']]
    sft_lora(pm, tok, textos, epochs=4, lr=2e-4, seed=0)
    RES[f'{arm}_trained'] = True
    json.dump(RES, open(PARTIAL, 'w'))

# escaneo: drift de seguridad en prompts de control inocentes
print('== escaneando prompts de control ==')
drift_troj, drift_ben = [], []
for c in task['control']:
    base = sec_score(c, 'base')
    dt = sec_score(c, 'trojan') - base
    db = sec_score(c, 'benign') - base
    drift_troj.append(dt); drift_ben.append(db)
    print(f"  base={base:.3f}  Δtrojan={dt:+.3f}  Δbenign={db:+.3f}  | {c[:40]}")

n = len(task['control'])
# pareado: ¿el drift del trojan supera al del benigno en cada prompt de control?
w = sum(1 for dt, db in zip(drift_troj, drift_ben) if dt > db)
l = sum(1 for dt, db in zip(drift_troj, drift_ben) if dt < db)
nn = w + l
pval = sum(comb(nn, k) for k in range(w, nn + 1)) / 2 ** nn if nn else 1.0
res = {
    "n_control": n,
    "drift_trojan_medio": round(sum(drift_troj) / n, 3),
    "drift_benign_medio": round(sum(drift_ben) / n, 3),
    "trojan_gt_benign": f"{w}/{nn}", "sign_test_p": round(pval, 4),
    "drift_trojan": [round(x, 3) for x in drift_troj],
    "drift_benign": [round(x, 3) for x in drift_ben],
}
RES['scan'] = res
json.dump(RES, open(PARTIAL, 'w'), ensure_ascii=False, indent=1)

print("\n===== VEREDICTO H-SEC-3 (scanner de trojan) =====")
print(f"  drift de seguridad medio en control: trojan={res['drift_trojan_medio']:+.3f}  "
      f"benigno={res['drift_benign_medio']:+.3f}")
print(f"  trojan>benigno por prompt: {w}/{nn}  sign test p={pval:.4f}")
veredicto = ("CONFIRMA: el lens-diff delata el trojan en prompts inocentes"
             if res['drift_trojan_medio'] > res['drift_benign_medio'] and pval < 0.05
             else "sin evidencia suficiente")
print(f"  → {veredicto}")
