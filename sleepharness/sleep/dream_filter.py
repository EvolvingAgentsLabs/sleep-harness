"""Filtrado de dreams: gradiente (paper §3.4) + J-Space (Idea 1 del crossover).

El paper selecciona dreams por score de gradiente (g_DR = ∇θ L_SFT). La
contribución nuestra es un segundo filtro por proceso interno: pasar el dream
por el pizarrón (leer_pizarron + detectar) y exigir que la firma objetivo se
encienda. Un dream cuyo output parece correcto pero cuyo workspace no muestra
la firma (p. ej. `computo` sin dígitos ni conceptos de cálculo) es un
candidato a alucinación y se descarta.

`seleccionar` implementa las 4 condiciones del experimento 1:
    none | grad | jspace | combinado
con Top-k por score + b muestras aleatorias para diversidad (§3.4).
"""

from __future__ import annotations

import random
import re

import torch

from .dreams import Dream


# ---------- score por J-Space (Idea 1; corre local o en Colab con el lens) ----------

_PALABRA_RE = re.compile(r"[A-Za-zÁÉÍÓÚáéíóúüñÑ]{4,}|\d+")


def anclas_de_contexto(contexto: str) -> set[str]:
    """Anclas dinámicas para la firma de FIDELIDAD: palabras de contenido
    (≥4 letras) y números del contexto fuente. Los números son anclas
    fuertes: un dream factual fiel re-enciende los valores del contexto."""
    from harness.runtime import STOPWORDS

    return {w.lower() for w in _PALABRA_RE.findall(contexto)
            if w.lower() not in STOPWORDS}


def _matchea_ancla(token: str, anclas: set[str]) -> bool:
    """Match por familia de prefijos, no palabra exacta.

    Calibración exp1 corrida 1: los nombres propios se tokenizan en
    sub-palabras ("Vant"+"ar") y el pizarrón enciende conceptos a veces en
    inglés ("membrane" por "membrana") — el match exacto daba fidelidad 0.0
    hasta en dreams fieles. El prefijo compartido (≥4 chars) captura
    sub-tokens y cognados ES/EN; los números siguen siendo match exacto.
    """
    t = token.strip().lower()
    if not t:
        return False
    if t in anclas:
        return True
    if t.isdigit() or len(t) < 4:
        return False  # números: solo match exacto; tokens cortos: ruido
    for a in anclas:
        if a.isdigit() or len(a) < 4:
            continue
        if t in a or a in t:      # sub-token ("vant" ⊂ "vantar")
            return True
        if t[:5] == a[:5]:        # familia/cognado ("membr-ane/-ana")
            return True
    return False


def puntuar_top_con_anclas(top: list[dict], anclas: set[str]) -> float:
    """Fracción de la intensidad del top capturada por las anclas (misma
    aritmética que las firmas estáticas de detectar, con léxico dinámico y
    match por familia de prefijos)."""
    total = sum(t["intensidad"] for t in top) or 1.0
    peso = sum(t["intensidad"] for t in top if _matchea_ancla(t["token"], anclas))
    return round(peso / total, 3)


def score_lexico(dream: Dream, contexto: str) -> float:
    """Baseline de CONTROL (plan V2+): solape léxico texto-del-dream vs
    contexto fuente, sin lens. Misma familia de matching por prefijos que la
    fidelidad de workspace, aplicada al TEXTO. Si este baseline gratis
    empata al filtro de workspace, el lens no aporta para esta tarea; si el
    workspace gana, el claim de la Idea 1 queda blindado. No requiere GPU."""
    anclas = anclas_de_contexto(contexto)
    toks = [t.lower() for t in _PALABRA_RE.findall(dream.texto)]
    if not toks:
        s = 0.0
    else:
        s = round(sum(1 for t in toks if _matchea_ancla(t, anclas)) / len(toks), 3)
    dream.scores["lexical"] = s
    return s


def score_jspace(rt, dream: Dream, *, firma_objetivo: str = "computo",
                 contexto: str | None = None,
                 rastrear: list[str] | None = None,
                 top_k: int = 30, max_posiciones: int = 20) -> float:
    """Score J-Space del dream; muta el dream (scores/firmas).

    - firma estática ("computo", "verificacion", …): fracción de intensidad
      del pizarrón en las anclas de esa firma (léxico extendido) — para
      tareas de razonamiento.
    - firma "fidelidad": anclas dinámicas extraídas del contexto fuente —
      para tareas factuales (knowledge incorporation), donde `computo` no
      aplica. Lección de exp2: la firma debe estar alineada con la tarea.
    """
    from ..signatures_ext import detectar_ext

    ws = rt.leer_pizarron(dream.texto, rastrear=rastrear, top_k=top_k,
                          max_posiciones=max_posiciones)
    firmas = detectar_ext(ws.top)
    dream.firmas = firmas
    dream.firmas["_top"] = [t["token"] for t in ws.top[:12]]  # debug
    if firma_objetivo == "fidelidad":
        if contexto is None:
            raise ValueError("firma 'fidelidad' requiere el contexto fuente")
        s = puntuar_top_con_anclas(ws.top, anclas_de_contexto(contexto))
        dream.firmas = {**firmas, "fidelidad": {"score": s}}
    else:
        s = float(firmas[firma_objetivo]["score"])
    dream.scores["jspace"] = s
    return s


# ---------- score por gradiente (paper §3.4; pensado para GPU en Colab) ----------

def score_gradiente(model, tokenizer, dream: Dream, *, device=None,
                    max_length: int = 384) -> float:
    """Norma del gradiente de L_SFT(dream) sobre los parámetros entrenables
    (importancia ω del paper). Un backward por dream; limpia los grads."""
    device = device or next(model.parameters()).device
    ids = tokenizer(dream.texto, return_tensors="pt", truncation=True,
                    max_length=max_length).input_ids.to(device)
    entrenables = [p for p in model.parameters() if p.requires_grad]
    if not entrenables:  # sin adaptador: usar todo el modelo es carísimo; avisar
        raise RuntimeError("score_gradiente requiere parámetros entrenables (LoRA)")
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    total = 0.0
    for p in entrenables:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm() ** 2)
    model.zero_grad(set_to_none=True)
    del out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    g = total ** 0.5
    dream.scores["gradiente"] = g
    return g


# ---------- selección (lógica pura, testeable sin modelo) ----------

def _ranks(valores: list[float]) -> list[float]:
    """Rango normalizado en [0,1] (mayor valor -> rango mayor)."""
    orden = sorted(range(len(valores)), key=lambda i: valores[i])
    r = [0.0] * len(valores)
    for pos, i in enumerate(orden):
        r[i] = pos / max(len(valores) - 1, 1)
    return r


def seleccionar(dreams: list[Dream], *, k: int, b_random: int = 1,
                modo: str = "combinado", seed: int = 0) -> list[Dream]:
    """Top-k por score según el modo + b aleatorios de los restantes.

    modos: "none" (k+b al azar), "grad", "jspace", "combinado"
    (promedio de rangos de ambos scores).
    """
    if not dreams:
        return []
    rng = random.Random(seed)
    if modo == "none":
        n = min(k + b_random, len(dreams))
        return rng.sample(dreams, n)

    if modo == "grad":
        claves = [d.scores.get("gradiente", 0.0) for d in dreams]
    elif modo == "jspace":
        claves = [d.scores.get("jspace", 0.0) for d in dreams]
    elif modo == "lexical":
        claves = [d.scores.get("lexical", 0.0) for d in dreams]
    elif modo == "combinado":
        rg = _ranks([d.scores.get("gradiente", 0.0) for d in dreams])
        rj = _ranks([d.scores.get("jspace", 0.0) for d in dreams])
        claves = [(a + c) / 2 for a, c in zip(rg, rj)]
    else:
        raise ValueError(f"modo desconocido: {modo}")

    for d, c in zip(dreams, claves):
        d.scores["combinado" if modo == "combinado" else f"clave_{modo}"] = round(float(c), 4)

    orden = sorted(range(len(dreams)), key=lambda i: -claves[i])
    top = [dreams[i] for i in orden[:k]]
    resto = [dreams[i] for i in orden[k:]]
    if b_random and resto:
        top += rng.sample(resto, min(b_random, len(resto)))
    return top


@torch.no_grad()
def descartar_alucinaciones(dreams: list[Dream], *, firma_objetivo: str = "computo",
                            umbral: float = 0.05) -> tuple[list[Dream], list[Dream]]:
    """Corte duro de la Idea 1: dreams cuyo workspace no enciende la firma
    objetivo por encima del umbral se descartan (no razonó, alucinó)."""
    ok, fuera = [], []
    for d in dreams:
        (ok if d.scores.get("jspace", 0.0) >= umbral else fuera).append(d)
    return ok, fuera
