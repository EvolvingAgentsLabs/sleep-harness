"""REM — Dreaming para auto-mejora (§3.4) y el handoff local → Colab.

El ciclo REM completo es: generar dreams → puntuar (J-Space local, gradiente
en GPU) → seleccionar → SFT-LoRA sobre los seleccionados → reward por mejora
en la tarea τ (Ec. 5) → ReSTEM (quedarse con los dreams que mejoraron y
re-entrenar el generador sobre ellos).

Como el fine-tuning corre en Google Colab, este módulo define el BUNDLE:
un JSON con todo lo que el notebook necesita (contextos, dreams ya
puntuados por J-Space, QA de evaluación, sondas de olvido y configuración).
Lo local genera y puntúa; Colab entrena y evalúa.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .. import config
from .dream_filter import score_jspace
from .dreams import Dream, generar_dreams


def preparar_bundle_rem(rt, contextos: list[str], *, qa: list[dict],
                        sondas: list[dict] | None = None,
                        firma_objetivo: str | None = None,
                        steer_pool: list[str] | None = None,
                        m: int | None = None, seed: int = 0,
                        notas: str = "") -> dict:
    """Corre la parte local del ciclo REM y arma el bundle para Colab.

    qa: [{pregunta, respuesta}] para evaluar incorporación sin contexto.
    sondas: suite de olvido (ver eval/forgetting.py); opcional.
    """
    cfg = config.DREAMING
    firma = firma_objetivo or cfg["firma_objetivo"]
    dreams = generar_dreams(
        rt, contextos, m=m or cfg["m"], max_new_tokens=cfg["max_new_tokens"],
        steer_pool=steer_pool, steer_alpha=cfg["steer_alpha"], seed=seed,
    )
    for d in dreams:
        score_jspace(rt, d, firma_objetivo=firma,
                     contexto=contextos[d.contexto_idx])

    return {
        "tipo": "rem",
        "creado": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "modelo": config.HF_MODEL,
        "lens": {"repo": config.LENS_REPO, "file": config.LENS_FILE,
                 "revision": config.LENS_REVISION},
        "firma_objetivo": firma,
        "contextos": contextos,
        "dreams": [d.to_dict() for d in dreams],
        "qa": qa,
        "sondas": sondas or [],
        "config": {"dreaming": cfg, "lora": config.LORA, "gkd": config.GKD,
                   "lti": config.LTI, "workspace_loss": config.WORKSPACE_LOSS},
        "notas": notas,
    }


def guardar_bundle(bundle: dict, nombre: str) -> Path:
    config.BUNDLES.mkdir(parents=True, exist_ok=True)
    path = config.BUNDLES / f"{nombre}.json"
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2))
    return path


def cargar_bundle(path: str | Path) -> dict:
    bundle = json.loads(Path(path).read_text())
    bundle["_dreams"] = [Dream.from_dict(d) for d in bundle.get("dreams", [])]
    return bundle


def recompensa_dream(acierto_antes: float, acierto_despues: float) -> int:
    """Ec. 5 del paper: 1 si el fine-tuning sobre el dream mejoró τ, 0 si no."""
    return int(acierto_despues > acierto_antes)
