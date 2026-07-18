"""Configuración compartida local/Colab.

La política del proyecto: todo lo que requiera fine-tuning (SFT, GKD,
workspace distillation, ReSTEM) corre primero en Google Colab; lo local
(lens, pizarrón, router, evaluación, preparación de bundles) corre en la Mac.
"""

from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "resultados"
BUNDLES = RAIZ / "bundles"          # handoff local -> Colab (JSON)
TASKS = RAIZ / "tasks"

# Modelo sujeto. "qwen" tiene lens pre-ajustado en neuronpedia/jacobian-lens
# (ver harness.runtime.MODELS). Para iterar rápido en Colab con menos VRAM se
# puede usar QLoRA (cargar en 4-bit) sobre el mismo modelo.
MODEL_KEY = "qwen"
HF_MODEL = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
LENS_REVISION = "qwen-n1000"

# ---- Hiperparámetros por defecto (nombres del paper §3.3-3.4) ----

LORA = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

GKD = {
    "lam": 0.5,        # λ: fracción on-policy (muestras del student) vs data del teacher
    "divergencia": "jsd",
    "temperatura": 1.0,
}

LTI = {
    "gamma": 0.5,      # γ: peso reward semántico vs absoluto (Ec. 3)
    "z0": 0.6,         # umbral Levenshtein normalizado (Ec. 4): si dist_norm > z0, r_abs = 0
    "frac_prefijo": (0.2, 0.6),  # rango de la fracción del dream usada como prefijo
}

DREAMING = {
    "m": 8,            # dreams generados por contexto
    "top_k": 4,        # seleccionados por score
    "b_random": 1,     # extra aleatorios para diversidad (paper §3.4)
    "max_new_tokens": 220,
    "steer_alpha": 0.04,   # intensidad del steering (0.10 degeneraba 12/12 dreams)
    "firma_objetivo": "computo",
}

WORKSPACE_LOSS = {
    "top_k_vocab": 96,     # tokens del sub-vocabulario por capa (top del teacher + rastreados)
    "temperatura": 2.0,
    "peso": 0.3,           # coeficiente del término de workspace en la loss total
}

SCHEDULER_CHUNKS = {
    # nivel -> C^(l): cada cuántos pasos de wake consolida ese nivel.
    # "rapida" es el contexto (se consolida cada sesión), "media" el LoRA de
    # sesión, "lenta" la fusión al modelo base.
    "rapida": 4,
    "media": 16,
    "lenta": 64,
}
