"""Exp1 (parte local) — Genera y puntúa dreams; exporta el bundle para Colab.

Pipeline: contextos de facts_mini → dreams (con y sin steering de concepto
aleatorio) → score J-Space (leer_pizarron + detectar) → bundle JSON en
bundles/. El fine-tuning (las 4 condiciones de filtrado + SFT-LoRA + eval)
corre en notebooks/colab_exp1_dream_filter.ipynb.

Uso:
    .venv/bin/python experiments/exp1_prepare_dreams.py [--m 8] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepharness import config
from sleepharness.sleep.rem import guardar_bundle, preparar_bundle_rem

# pool de conceptos para el steering aleatorio (análogo del random expert):
# palabras de un token que mezclan dominios distintos al de los hechos
STEER_POOL = [
    " water", " energy", " ceramic", " titanium", " membrane", " ocean",
    " music", " history", " chemistry", " market", " engineering", " glass",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=None, help="dreams por contexto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--firma", default=None,
                    help="firma objetivo del filtro J-Space (default: la que "
                         "declara la tarea, o config)")
    ap.add_argument("--model", default=config.MODEL_KEY)
    args = ap.parse_args()

    task = json.loads((config.TASKS / "facts_mini.json").read_text())
    sondas = json.loads((config.TASKS / "sondas_olvido.json").read_text())["sondas"]

    from harness.runtime import Runtime
    print(f"cargando {args.model}…")
    rt = Runtime(args.model)

    print(f"generando dreams sobre {len(task['contextos'])} contextos "
          f"(m={args.m or config.DREAMING['m']})… esto usa el modelo local, paciencia.")
    firma = args.firma or task.get("firma_objetivo")
    bundle = preparar_bundle_rem(
        rt, task["contextos"], qa=task["qa"], sondas=sondas,
        firma_objetivo=firma, steer_pool=STEER_POOL,
        m=args.m, seed=args.seed,
        notas=f"tarea={task['nombre']}; tema={task['tema']}",
    )

    n = len(bundle["dreams"])
    con_firma = sum(1 for d in bundle["dreams"] if d["scores"].get("jspace", 0) > 0)
    print(f"\n{n} dreams generados; {con_firma} encienden la firma "
          f"'{bundle['firma_objetivo']}' (score > 0).")
    for d in bundle["dreams"][:3]:
        print(f"  [{d['plantilla']}, steer={d['concepto_steer']}] "
              f"jspace={d['scores'].get('jspace')}: {d['texto'][:90]}…")

    path = guardar_bundle(bundle, f"rem_facts_mini_seed{args.seed}")
    print(f"\nbundle guardado en {path}")
    print("siguiente paso: subilo a Drive y corré notebooks/colab_exp1_dream_filter.ipynb")


if __name__ == "__main__":
    main()
