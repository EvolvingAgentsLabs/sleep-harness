"""NREM — Consolidación de memoria con Knowledge Seeding (§3.3 del paper).

Proxy del KS sobre un Transformer HF estándar:

- Teacher  = el modelo CON el contexto de la sesión en el prompt (el "modelo
  chico con información privilegiada comprimida": su conocimiento extra vive
  en la memoria rápida, el contexto).
- Student  = el mismo modelo SIN contexto, con un adaptador LoRA nuevo como
  única parte entrenable (la "expansión de parámetros" de §3.2).

Objetivo = GKD (mezcla λ de data del teacher y muestras on-policy del
student, ambas puntuadas con la divergencia teacher∥student) + LTI
(Learning to Imitate, Ec. 3-4): el teacher genera un dream, se corta un
prefijo, el student completa y recibe reward semántico + Levenshtein.
Fiel al paper, el RL se optimiza estilo ReSTEM: reward-filtered SFT.

Las funciones de pérdida/reward son puras (corren en Colab dentro del loop
de entrenamiento); la construcción del dataset usa cualquier callable de
generación, así sirve local (Runtime) y en Colab (model.generate).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

SONDAS_GENERICAS = [
    "Resumí los hechos más importantes que conocés sobre el siguiente tema: {tema}",
    "¿Qué sabés sobre {tema}? Respondé con datos concretos.",
    "Explicale a un colega los puntos clave de {tema}.",
]


# ---------- dataset D: muestras del teacher (con contexto) ----------

@dataclass
class ParKS:
    prompt_student: str      # sin contexto
    prompt_teacher: str      # con contexto (info privilegiada)
    respuesta_teacher: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def construir_dataset_ks(generar_teacher, contexto: str, *, tema: str,
                         sondas: list[str] | None = None,
                         extra_prompts: list[str] | None = None) -> list[ParKS]:
    """`generar_teacher(prompt) -> str` es el modelo con acceso al contexto."""
    from .dreams import limpiar_dream

    prompts = [(s.format(tema=tema)) for s in (sondas or SONDAS_GENERICAS)]
    prompts += extra_prompts or []
    pares = []
    for p in prompts:
        p_teacher = f"Contexto:\n{contexto.strip()}\n\n{p}\n\nRespuesta:\n"
        y = limpiar_dream(generar_teacher(p_teacher))  # sin bloques <think>
        if y.strip():
            pares.append(ParKS(prompt_student=f"{p}\n\nRespuesta:\n",
                               prompt_teacher=p_teacher,
                               respuesta_teacher=y.strip()))
    return pares


# ---------- pérdida GKD (pura; corre en Colab) ----------

def perdida_gkd(logits_teacher: torch.Tensor, logits_student: torch.Tensor,
                mask: torch.Tensor, *, divergencia: str = "jsd",
                temperatura: float = 1.0) -> torch.Tensor:
    """Divergencia F(teacher ∥ student) por token, promediada sobre `mask`.

    logits: [B, T, V] alineados sobre la misma secuencia (los tokens de la
    respuesta); mask: [B, T] con 1 en posiciones de respuesta.
    """
    t = temperatura
    log_p = F.log_softmax(logits_teacher.float() / t, dim=-1)   # teacher
    log_q = F.log_softmax(logits_student.float() / t, dim=-1)   # student
    if divergencia == "fkl":        # forward KL(p ∥ q): cubre los modos del teacher
        kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    elif divergencia == "rkl":      # reverse KL(q ∥ p): mode-seeking
        kl = (log_q.exp() * (log_q - log_p)).sum(-1)
    elif divergencia == "jsd":
        log_m = torch.logaddexp(log_p, log_q) - torch.log(
            torch.tensor(2.0, device=log_p.device))
        kl = 0.5 * (log_p.exp() * (log_p - log_m)).sum(-1) \
           + 0.5 * (log_q.exp() * (log_q - log_m)).sum(-1)
    else:
        raise ValueError(f"divergencia desconocida: {divergencia}")
    mask = mask.float()
    return (kl * mask).sum() / mask.sum().clamp(min=1.0)


# ---------- LTI: rewards (Ec. 3-4; puras y testeables) ----------

def levenshtein_norm(a: str, b: str) -> float:
    """Distancia de Levenshtein normalizada por la longitud máxima, en [0,1]."""
    try:
        from rapidfuzz.distance import Levenshtein
        z = Levenshtein.distance(a, b)
    except ImportError:
        z = _levenshtein_puro(a, b)
    return z / max(len(a), len(b), 1)


def _levenshtein_puro(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    fila = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        nueva = [i]
        for j, cb in enumerate(b, 1):
            nueva.append(min(fila[j] + 1, nueva[-1] + 1, fila[j - 1] + (ca != cb)))
        fila = nueva
    return fila[-1]


def r_abs(pred: str, dream: str, *, z0: float = 0.6) -> float:
    """Ec. 4: 1 - dist_norm si la distancia normalizada <= z0, si no 0."""
    zn = levenshtein_norm(pred, dream)
    return 1.0 - zn if zn <= z0 else 0.0


def recompensa_lti(pred: str, dream: str, *, sem_fn=None, gamma: float = 0.5,
                   z0: float = 0.6) -> float:
    """Ec. 3: γ·r_sem + (1-γ)·r_abs. `sem_fn(pred, dream) -> {0,1}` es el
    reward model semántico congelado (por defecto, proxy por r_abs blando)."""
    sem = float(sem_fn(pred, dream)) if sem_fn else float(r_abs(pred, dream, z0=z0) > 0)
    return gamma * sem + (1 - gamma) * r_abs(pred, dream, z0=z0)


def similitud_embeddings(umbral: float = 0.75, modelo: str = "all-MiniLM-L6-v2"):
    """Reward model semántico congelado con sentence-transformers (Colab)."""
    from sentence_transformers import SentenceTransformer, util
    st = SentenceTransformer(modelo)

    def sem(pred: str, dream: str) -> int:
        e = st.encode([pred, dream], convert_to_tensor=True, normalize_embeddings=True)
        return int(float(util.cos_sim(e[0], e[1])) >= umbral)
    return sem


# ---------- LTI: generación de pares de imitación ----------

def lti_generar_pares(dreams: list[str], generar_student, *, tokenizer=None,
                      frac_prefijo=(0.2, 0.6), sem_fn=None, gamma: float = 0.5,
                      z0: float = 0.6, seed: int = 0) -> list[dict]:
    """Para cada dream del teacher: corta un prefijo aleatorio, el student
    completa, y se calcula el reward de la Ec. 3. Los pares con reward alto
    se usan como SFT (ReSTEM) en el notebook de Colab."""
    rng = random.Random(seed)
    pares = []
    for d in dreams:
        corte_frac = rng.uniform(*frac_prefijo)
        if tokenizer is not None:
            ids = tokenizer(d, add_special_tokens=False).input_ids
            corte = max(1, int(len(ids) * corte_frac))
            prefijo = tokenizer.decode(ids[:corte])
            continuacion = tokenizer.decode(ids[corte:])
        else:
            corte = max(1, int(len(d) * corte_frac))
            prefijo, continuacion = d[:corte], d[corte:]
        pred = generar_student(prefijo)
        r = recompensa_lti(pred, continuacion, sem_fn=sem_fn, gamma=gamma, z0=z0)
        pares.append({"prefijo": prefijo, "continuacion_teacher": continuacion,
                      "continuacion_student": pred, "reward": round(r, 4)})
    return pares
