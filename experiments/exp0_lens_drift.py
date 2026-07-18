"""Exp0 — ¿Sigue valiendo el lens cuando los pesos cambian? (parte local)

El lens de Qwen3.5-4B fue ajustado sobre el modelo BASE; todo el proyecto
asume que sigue siendo una lectura fiel tras updates low-rank. Este script
mide el drift del readout bajo perturbaciones low-rank sintéticas de
magnitud creciente en los MLP (proxy de un LoRA), o bajo un adaptador real
(--adapter, entrenado en Colab).

Métricas sobre prompts de calibración:
- solape top-k del pizarrón (Jaccard) perturbado vs base
- deriva de los scores de firma (signatures.detectar)

El re-fit del lens sobre el modelo actualizado (la comparación de verdad)
corre en GPU: notebooks/colab_exp0_lens_refit.ipynb.

Uso:
    .venv/bin/python experiments/exp0_lens_drift.py [--epsilons 0.001 0.005 0.02]
    .venv/bin/python experiments/exp0_lens_drift.py --adapter <dir_peft>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from sleepharness import config

PROMPTS_CALIBRACION = [
    "El precio óptimo se obtiene maximizando la ganancia G(x) = (30 - 5x)(100 + 20x).",
    "La capital de Francia es París, una ciudad con más de dos millones de habitantes.",
    "Para verificar el resultado, recomputá cada paso y revisá el signo del término lineal.",
    "The quarterly revenue grew by 12 percent, driven by demand in the northern region.",
    "La membrana cerámica opera a 41 grados y reduce el consumo energético un 37 por ciento.",
]


def leer_firmado(rt, texto: str, top_k: int = 20):
    from harness.signatures import detectar
    ws = rt.leer_pizarron(texto, top_k=top_k)
    return set(ws.tokens_top()), detectar(ws.top)


def perturbar_lowrank(model, *, epsilon: float, rank: int = 16, seed: int = 0):
    """Suma eps * ||W|| * (A@B)/||A@B|| a los proyectores MLP. Devuelve un
    callable que revierte la perturbación."""
    gen = torch.Generator().manual_seed(seed)
    aplicados = []
    for nombre, p in model.named_parameters():
        if p.dim() == 2 and any(k in nombre for k in ("gate_proj", "up_proj", "down_proj")):
            a = torch.randn(p.shape[0], rank, generator=gen).to(p.device, p.dtype)
            b = torch.randn(rank, p.shape[1], generator=gen).to(p.device, p.dtype)
            delta = a @ b
            delta = delta * (epsilon * p.detach().float().norm()
                             / delta.float().norm()).to(p.dtype)
            with torch.no_grad():
                p.add_(delta)
            aplicados.append((p, delta))

    def revertir():
        with torch.no_grad():
            for p, delta in aplicados:
                p.sub_(delta)
    return revertir


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilons", nargs="*", type=float,
                    default=[0.001, 0.005, 0.02, 0.05])
    ap.add_argument("--adapter", type=str, default=None,
                    help="dir de un adaptador peft real (entrenado en Colab)")
    ap.add_argument("--model", default=config.MODEL_KEY)
    args = ap.parse_args()

    from harness.runtime import Runtime
    print(f"cargando {args.model}…")
    rt = Runtime(args.model)

    base = {p: leer_firmado(rt, p) for p in PROMPTS_CALIBRACION}
    resultados = {"modelo": args.model, "condiciones": []}

    def medir(etiqueta: str) -> dict:
        filas = []
        for p in PROMPTS_CALIBRACION:
            tokens, firmas = leer_firmado(rt, p)
            t0, f0 = base[p]
            delta_firmas = {
                k: round(firmas[k]["score"] - f0[k]["score"], 3)
                for k in firmas if k != "_dominante"
            }
            filas.append({"prompt": p[:60], "jaccard_top": round(jaccard(tokens, t0), 3),
                          "delta_firmas": delta_firmas})
        j = sum(f["jaccard_top"] for f in filas) / len(filas)
        print(f"  {etiqueta}: jaccard_top medio = {j:.3f}")
        return {"condicion": etiqueta, "jaccard_medio": round(j, 4), "prompts": filas}

    if args.adapter:
        from peft import PeftModel
        print(f"aplicando adaptador {args.adapter}…")
        rt.hf_model = PeftModel.from_pretrained(rt.hf_model, args.adapter)
        rt.hf_model.merge_and_unload()
        resultados["condiciones"].append(medir(f"adapter:{args.adapter}"))
    else:
        for eps in args.epsilons:
            revertir = perturbar_lowrank(rt.hf_model, epsilon=eps)
            resultados["condiciones"].append(medir(f"eps={eps}"))
            revertir()

    out = config.RESULTADOS / "exp0" / "lens_drift.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(resultados, ensure_ascii=False, indent=2))
    print(f"\nguardado en {out}")
    print("siguiente paso: notebooks/colab_exp0_lens_refit.ipynb re-fitea el "
          "lens en GPU y compara contra el lens original.")


if __name__ == "__main__":
    main()
