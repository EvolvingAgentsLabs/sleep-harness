"""Generación de dreams (datos sintéticos) durante REM (§3.4 del paper).

Un dream se genera a partir de un contexto reciente del diario de wake, con
una plantilla de self-edit (estilo SEAL: implicaciones, QA, reescritura).

Análogo del "random expert selection" del paper: acá no hay MoE, pero sí
J-Space — para romper el sesgo hacia el conocimiento existente, cada dream
puede generarse bajo STEER de un concepto aleatorio del pool (Intervention
de jlens-harness), inyectando conocimiento "irrelevante" de forma
interpretable: la traza registra QUÉ concepto se inyectó.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

PLANTILLAS = {
    "implicaciones": (
        "Contenido:\n{contexto}\n\n"
        "Escribí una lista de implicaciones y hechos que se deducen del "
        "contenido anterior. Reformulá cada hecho con tus palabras, sin "
        "copiar frases textuales.\n\nImplicaciones:\n"
    ),
    "qa": (
        "Contenido:\n{contexto}\n\n"
        "Escribí pares de pregunta y respuesta que cubran los hechos del "
        "contenido anterior, uno por línea con el formato P: ... / R: ...\n\n"
    ),
    "reescritura": (
        "Contenido:\n{contexto}\n\n"
        "Reescribí el contenido anterior como una explicación clara y "
        "autocontenida, integrando los hechos con conocimiento general "
        "relacionado.\n\nExplicación:\n"
    ),
}


@dataclass
class Dream:
    texto: str
    plantilla: str
    contexto_idx: int
    concepto_steer: str | None = None
    scores: dict = field(default_factory=dict)   # jspace / gradiente / combinado
    firmas: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "texto": self.texto, "plantilla": self.plantilla,
            "contexto_idx": self.contexto_idx,
            "concepto_steer": self.concepto_steer,
            "scores": self.scores, "firmas": self.firmas,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Dream":
        return cls(**d)


def _conceptos_validos(rt, pool: list[str]) -> list[str]:
    """Filtra el pool a palabras que son token único (requisito de STEER)."""
    return [c for c in pool if rt.token_unico(c) is not None]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def limpiar_dream(texto: str) -> str:
    """Saca bloques <think> (el modelo los emite si el prompt no trae el
    arranque) y espacios sobrantes."""
    return _THINK_RE.sub("", texto).strip()


def es_degenerado(texto: str, *, min_palabras_unicas: int = 12) -> bool:
    """Colapso por repetición (p.ej. steering demasiado fuerte en greedy)."""
    return len(set(texto.split())) < min_palabras_unicas


def generar_dreams(rt, contextos: list[str], *, m: int = 8,
                   max_new_tokens: int = 220,
                   steer_pool: list[str] | None = None,
                   steer_alpha: float = 0.04,
                   frac_steered: float = 0.5,
                   seed: int = 0) -> list[Dream]:
    """Genera m dreams por contexto rotando plantillas; una fracción de ellos
    bajo steering SUAVE de un concepto aleatorio (exploración).

    Calibración exp1 corrida 1 (2026-07-17): (1) el prompt lleva el
    `arranque` del runtime — sin él, Qwen emite bloques <think> que
    contaminan el dream; (2) los dreams se muestrean con temperatura (la
    diversidad es el punto de soñar; greedy además colapsa bajo steering);
    (3) alpha=0.10 en todas las capas degenera 12/12 dreams — se baja a
    0.04 y se interviene 1 de cada 3 capas; (4) los degenerados se filtran.
    """
    from harness.interventions import Intervention

    rng = random.Random(seed)
    pool = _conceptos_validos(rt, steer_pool or [])
    nombres = list(PLANTILLAS)
    dreams: list[Dream] = []

    decoding_original = rt.decoding
    rt.decoding = {"do_sample": True, "top_p": 0.95, "temperature": 0.9}
    try:
        for ci, contexto in enumerate(contextos):
            for j in range(m):
                plantilla = nombres[j % len(nombres)]
                prompt = (PLANTILLAS[plantilla].format(contexto=contexto.strip())
                          + rt.arranque)
                concepto = None
                if pool and rng.random() < frac_steered:
                    concepto = rng.choice(pool)
                    capas_suaves = rt.capas[::3] or rt.capas
                    with Intervention(rt, steer={concepto: steer_alpha},
                                      capas=capas_suaves):
                        r = rt.step(prompt, max_new_tokens=max_new_tokens,
                                    leer_salida=False)
                else:
                    r = rt.step(prompt, max_new_tokens=max_new_tokens,
                                leer_salida=False)
                texto = limpiar_dream(r.salida)
                if texto and not es_degenerado(texto):
                    dreams.append(Dream(texto=texto, plantilla=plantilla,
                                        contexto_idx=ci, concepto_steer=concepto))
    finally:
        rt.decoding = decoding_original
    return dreams
