"""Exp2 — Co-evolución pesos/harness (Idea 2), 100% local.

Automatiza el loop analista de jlens-harness dentro del ciclo de sueño:
corre la tarea de pricing con la spec deficiente (v1), y en cada "sueño" el
router cruza verificador × firmas del pizarrón para decidir la ruta:

    CONTEXTO -> inyecta los datos reales en la spec
    TOOL     -> sintetiza la tool calc y ajusta las instrucciones
    PESOS    -> exporta un bundle REM para Colab (no entrena local)
    NADA     -> convergió

La trayectoria esperada replica v1→v2→v5 del experimento original, ahora
decidida por el router en vez del analista humano.

Uso:
    .venv/bin/python experiments/exp2_coevolution.py [--max-ciclos 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepharness import config
from sleepharness.eval.verifiers import verificar_numerico
from sleepharness.evolution.router import Ruta, decidir, sintetizar_parche

RASTREAR = ["information", "data", "profit", "price", "vertex", "sign",
            "verification", "check"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ciclos", type=int, default=5)
    ap.add_argument("--model", default=config.MODEL_KEY)
    ap.add_argument("--task", default=None,
                    help="ruta a un pricing.json (default: el de jlens-harness)")
    ap.add_argument("--version", default="v1_deficiente",
                    help="versión inicial de la spec (p.ej. v5_tool_calc para "
                         "verificar un parche del analista)")
    args = ap.parse_args()

    from harness.agentspec import AgentSpec, ToolSpec
    from harness.runtime import Runtime
    from sleepharness.signatures_ext import detectar_ext as detectar
    from sleepharness.wake.tools import paso_con_tools
    from sleepharness.wake.session import WakeSession

    task_path = Path(args.task) if args.task else (
        Path(config.RAIZ).parent / "jlens-harness" / "tasks" / "pricing.json")
    task = json.loads(task_path.read_text())
    v1 = task["versiones"][args.version]
    spec = AgentSpec(name=f"pricing_coevolucion_{args.version}",
                     instructions=v1["instructions"], data=v1.get("data", ""),
                     tools=[ToolSpec(t["name"], t["description"])
                            for t in v1.get("tools", [])],
                     output_budget=v1.get("output_budget", 160))

    print(f"cargando {args.model}…")
    rt = Runtime(args.model)

    out_dir = config.RESULTADOS / "exp2"
    out_dir.mkdir(parents=True, exist_ok=True)
    historia = []

    def verificar(salida: str) -> dict:
        return verificar_numerico(salida, task["esperado"])

    for ciclo in range(1, args.max_ciclos + 1):
        print(f"\n=== ciclo {ciclo} | spec: tools={[t.name for t in spec.tools]} "
              f"data={'sí' if spec.data else 'no'} ===")
        sesion = WakeSession(rt, spec, out_dir / f"trace_c{ciclo}.jsonl",
                             rastrear=RASTREAR)
        if spec.tools:
            from harness.trace import registro
            r, rondas = paso_con_tools(rt, spec, task["tarea"], rastrear=RASTREAR)
            verificacion = verificar(r.salida)
            sesion.trace.write(registro(f"c{ciclo}_tools_r{len(rondas)}",
                                        spec.to_dict(), task["tarea"], r,
                                        verificacion))
            firmas_salida = detectar(r.ws_salida.top) if r.ws_salida else None
            firmas_prompt = detectar(r.ws_prompt.top)
            salida, truncada = r.salida, r.truncada
        else:
            p = sesion.paso(task["tarea"], verificar=verificar)
            verificacion, salida, truncada = p.verificacion, p.salida, p.truncada
            firmas_salida, firmas_prompt = p.firmas_salida, p.firmas_prompt
        sesion.cerrar()

        print(f"  salida: {salida.strip()[:120]}…")
        print(f"  verificación: ok={verificacion['ok']} "
              f"pide_datos={verificacion['pide_datos']} "
              f"n_numeros={verificacion['n_numeros']} truncada={truncada}")

        decision = decidir(verificacion, firmas_salida, firmas_prompt,
                           truncada=truncada)
        print(f"  router → {decision.ruta.value}: {decision.motivo}")
        historia.append({"ciclo": ciclo, "spec": spec.to_dict(),
                         "salida": salida, "verificacion": verificacion,
                         "decision": decision.to_dict()})

        if decision.ruta is Ruta.NADA:
            print("  convergió ✅")
            break
        if decision.ruta is Ruta.PESOS:
            from sleepharness.sleep.rem import guardar_bundle, preparar_bundle_rem
            print("  ruta paramétrica: exportando bundle REM para Colab…")
            bundle = preparar_bundle_rem(
                rt, [task["datos_reales"]],
                qa=[{"pregunta": task["tarea"],
                     "respuesta": [str(task["esperado"]["precio"]),
                                   str(task["esperado"]["ganancia"])]}],
                notas="exp2: el router derivó a PESOS")
            print(f"  bundle: {guardar_bundle(bundle, f'rem_exp2_c{ciclo}')}")
            break
        parche = sintetizar_parche(decision, spec.to_dict(),
                                   datos_reales=task["datos_reales"])
        if parche is None:
            from sleepharness.evolution.router import prompt_analista
            pedido = prompt_analista(decision, spec.to_dict(), task["tarea"],
                                     salida)
            pedido_path = out_dir / f"pedido_analista_c{ciclo}.md"
            pedido_path.write_text(pedido)
            print("  techo de las plantillas: los parches automáticos se "
                  "agotaron sin converger.")
            print(f"  pedido para el analista de frontera → {pedido_path}")
            break
        spec = AgentSpec.from_dict(parche)

    resumen = out_dir / f"resumen_{time.strftime('%Y%m%d_%H%M%S')}.json"
    resumen.write_text(json.dumps(historia, ensure_ascii=False, indent=2))
    print(f"\nhistoria guardada en {resumen}")


if __name__ == "__main__":
    main()
