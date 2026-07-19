#!/usr/bin/env python3
"""Análisis final de sleep-harness: veredictos pre-registrados consolidados.

Lee los archivos archivados (V2, gemelos, V6) y produce la tabla de
veredictos con CIs pareados por bootstrap. Salida: consola +
resultados/colab/veredictos.json. Reproducible: python scripts/analisis_final.py
"""

import json
import random
from math import comb
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RC = RAIZ / "resultados" / "colab"
rng = random.Random(0)


def ci_pareado(difs, n_boot=10000):
    n = len(difs)
    media = sum(difs) / n
    boots = sorted(sum(rng.choice(difs) for _ in range(n)) / n
                   for _ in range(n_boot))
    return media, boots[int(0.05 * n_boot)], boots[int(0.95 * n_boot)]


veredictos = {}

# ---------- V2: filtrado de dreams (Idea 1) ----------
seeds = {s: json.load(open(RC / "v2" / f"exp1_seed{s}.json"))["condiciones"]
         for s in range(4)}
conds = ["none", "grad", "jspace", "combinado", "lexical"]
print("=" * 70)
print("V2 — FILTRADO DE DREAMS (4 semillas x 48 QA)")
tabla_v2 = {}
for m in conds:
    vals = [seeds[s][m]["incorporacion"] for s in range(4)]
    tabla_v2[m] = {"media": round(sum(vals) / 4, 3),
                   "rango": [min(vals), max(vals)],
                   "olvido": round(sum(seeds[s][m]["olvido_medio"] for s in range(4)) / 4, 3)}
    print(f"  {m:10s} media={tabla_v2[m]['media']:.3f} rango={tabla_v2[m]['rango']}")

pares_v2 = {}
for a, b in [("jspace", "grad"), ("lexical", "grad"), ("lexical", "jspace"),
             ("lexical", "none"), ("grad", "none"), ("jspace", "none")]:
    d = []
    for s in range(4):
        da = {p["pregunta"]: p["score"] for p in seeds[s][a]["detalle_inc"]}
        db = {p["pregunta"]: p["score"] for p in seeds[s][b]["detalle_inc"]}
        d += [da[q] - db[q] for q in da if q in db]
    m, lo, hi = ci_pareado(d)
    sig = lo > 0 or hi < 0
    pares_v2[f"{a}-{b}"] = {"media": round(m, 4), "ci90": [round(lo, 4), round(hi, 4)],
                            "significativo": sig}
    print(f"  {a}-{b}: {m:+.4f} CI90=[{lo:+.4f},{hi:+.4f}] {'**' if sig else ''}")

veredictos["H1_jspace_supera_grad"] = {
    "criterio": "CI90 pareado jspace-grad excluye 0 (pre-registrado)",
    "resultado": pares_v2["jspace-grad"],
    "veredicto": "REFUTADA" if not pares_v2["jspace-grad"]["significativo"]
                 and pares_v2["jspace-grad"]["media"] <= 0 else "ver datos",
}
veredictos["control_lexical"] = {
    "hallazgo": "el baseline léxico supera significativamente a todos los filtros",
    "datos": {k: v for k, v in pares_v2.items() if k.startswith("lexical")},
}

# ---------- Gemelos: veto semántico (Idea 1 refinada) ----------
gem = json.load(open(RC / "gemelos_veneno_scored.json"))
w = sum(1 for p in gem if p["jspace_fiel"] > p["jspace_corrupto"])
l = sum(1 for p in gem if p["jspace_fiel"] < p["jspace_corrupto"])
n = w + l
p_gem = sum(comb(n, k) for k in range(w, n + 1)) / 2**n
print("=" * 70)
print(f"GEMELOS — VETO SEMÁNTICO: {w}W/{len(gem)-w-l}T/{l}L, sign test p={p_gem:.4f}")
veredictos["H1b_veto_gemelos"] = {
    "criterio": ">60% wins estrictos (pre-registrado)",
    "wins": w, "ties": len(gem) - w - l, "losses": l,
    "sign_test_p": round(p_gem, 4),
    "veredicto": "CONFIRMADA" if w > len(gem) * 0.6 and p_gem < 0.05 else "NO CONFIRMADA",
}

# ---------- V6: ablaciones de consolidación (Idea 3) ----------
v6 = {s: json.load(open(RC / "v6" / f"exp3_seed{s}.json"))["condiciones"]
      for s in range(3)}
brazos = ["gkd", "gkd_ws", "gkd_ws_random", "ce_only"]
print("=" * 70)
print("V6 — ABLACIONES DE CONSOLIDACIÓN (3 semillas x 12 QA)")
tabla_v6 = {}
for m in brazos:
    vals = [v6[s][m]["incorporacion"] for s in range(3)]
    olv = [v6[s][m]["olvido_medio"] for s in range(3)]
    tabla_v6[m] = {"media": round(sum(vals) / 3, 3), "semillas": vals,
                   "olvido_max": max(olv)}
    print(f"  {m:14s} media={tabla_v6[m]['media']:.3f} semillas={vals} olvido_max={max(olv):+.2f}")

pares_v6 = {}
for a, b in [("gkd_ws", "gkd"), ("gkd_ws", "gkd_ws_random"), ("gkd", "ce_only")]:
    d = []
    for s in range(3):
        da = {p["pregunta"]: p["score"] for p in v6[s][a]["detalle_inc"]}
        db = {p["pregunta"]: p["score"] for p in v6[s][b]["detalle_inc"]}
        d += [da[q] - db[q] for q in da if q in db]
    m, lo, hi = ci_pareado(d)
    pares_v6[f"{a}-{b}"] = {"media": round(m, 4), "ci90": [round(lo, 4), round(hi, 4)],
                            "significativo": lo > 0 or hi < 0}
    print(f"  {a}-{b}: {m:+.4f} CI90=[{lo:+.4f},{hi:+.4f}]"
          f" {'**' if pares_v6[f'{a}-{b}']['significativo'] else ''}")

veredictos["H3_workspace_distillation"] = {
    "criterio": "gkd_ws > gkd replicado + gkd_ws_random no lo iguala",
    "resultado": pares_v6,
    "veredicto": ("NO REPLICADA (el 4x original fue pico de varianza; "
                  "cota del efecto = ancho del CI)"
                  if not pares_v6["gkd_ws-gkd"]["significativo"]
                  or pares_v6["gkd_ws-gkd"]["media"] <= 0 else "ver datos"),
}

# invariante de olvido
olvidos_ks = [v6[s][m]["olvido_medio"] for s in range(3) for m in brazos]
olvidos_sft = [seeds[s][m]["olvido_medio"] for s in range(4) for m in conds]
print("=" * 70)
print(f"INVARIANTE DE OLVIDO: consolidación KS max={max(olvidos_ks):+.2f} "
      f"(n={len(olvidos_ks)} brazos) vs SFT directo media="
      f"{sum(olvidos_sft)/len(olvidos_sft):+.3f} max={max(olvidos_sft):+.2f} "
      f"(n={len(olvidos_sft)})")
veredictos["invariante_olvido"] = {
    "hallazgo": "consolidación KS: olvido ~0 en todos los brazos/semillas; "
                "SFT directo: olvido positivo frecuente",
    "ks_max": max(olvidos_ks), "sft_media": round(sum(olvidos_sft) / len(olvidos_sft), 3),
    "sft_max": max(olvidos_sft), "estado": "robusto, pendiente réplica cross-modelo",
}

out = RC / "veredictos.json"
out.write_text(json.dumps(veredictos, ensure_ascii=False, indent=2))
print("=" * 70)
print(f"veredictos → {out}")
