"""Helpers de fine-tuning para los notebooks de Colab.

Política del proyecto: TODO fine-tuning corre primero en Google Colab.
Estos helpers mantienen la lógica en el paquete (testeable, versionada) y
dejan a los notebooks como orquestadores finos.

- sft_lora: SFT clásico sobre textos (dreams seleccionados / pares LTI).
- entrenar_ks: Knowledge Seeding = GKD sobre pares teacher/student, con
  término opcional de Workspace Distillation (Idea 3).
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from .nrem import ParKS, perdida_gkd
from .workspace_loss import WorkspaceDistillLoss


def _batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def sft_lora(model, tokenizer, textos: list[str], *, epochs: int = 2,
             lr: float = 1e-4, batch_size: int = 2, max_length: int = 384,
             seed: int = 0, log=print) -> list[float]:
    """SFT sobre los parámetros entrenables (el adaptador LoRA activo).

    Mezcla los textos por época con semilla explícita: en exp1f, dos corridas
    con ~el mismo set de dreams pero distinto orden divergieron 0.417 vs
    0.167 de incorporación — el orden es un confound real con pocos pasos.
    """
    import random as _random
    rng = _random.Random(seed)
    device = next(model.parameters()).device
    if device.type == "cuda":
        try:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except (AttributeError, TypeError):
            pass
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    perdidas = []
    for ep in range(epochs):
        textos = list(textos)
        rng.shuffle(textos)
        for lote in _batches(textos, batch_size):
            enc = tokenizer(lote, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(device)
            labels = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)
            out = model(**enc, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            perdidas.append(float(out.loss.detach()))
            del out
            if device.type == "cuda":
                torch.cuda.empty_cache()
        log(f"  sft epoch {ep + 1}/{epochs} loss={perdidas[-1]:.4f}")
    try:
        model.gradient_checkpointing_disable()
    except AttributeError:
        pass
    model.eval()
    return perdidas


def entrenar_ks(model, tokenizer, pares: list[ParKS], *, lens=None,
                capas: list[int] | None = None, vocab_ids: list[int] | None = None,
                lam: float = 0.5, divergencia: str = "jsd",
                peso_ws: float = 0.0, temperatura_ws: float = 2.0,
                peso_ce: float = 1.0, peso_gkd: float = 1.0,
                seed: int = 0,
                epochs: int = 2, lr: float = 1e-4, max_length: int = 384,
                max_resp_tokens: int = 144,
                max_new_tokens_onpolicy: int = 128, log=print) -> dict:
    """Knowledge Seeding: GKD teacher(con contexto) → student(sin contexto).

    El teacher y el student son el MISMO modelo con distinto prompt; solo el
    adaptador nuevo es entrenable (LoraStack.entrenar_solo). Con peso_ws > 0
    y un lens, agrega Workspace Distillation: las distribuciones
    lens-decodificadas del teacher y del student deben coincidir sobre los
    tokens de la respuesta.

    Mezcla λ de GKD: con prob. (1-λ) la secuencia evaluada es la respuesta
    del teacher (off-policy); con prob. λ es una muestra del student
    (on-policy), en ambos casos puntuada con la divergencia teacher∥student.

    Importante (§3.3, compute-consolidate-update): el forward del teacher se
    hace con los adaptadores DESACTIVADOS (estado pre-update congelado); si
    no, el teacher derivaría junto con el student durante el entrenamiento.

    peso_ce (calibración exp3 corrida 1): con teacher-forcing sobre el mismo
    texto de respuesta, la divergencia con/sin contexto es casi nula tras los
    primeros tokens (JSD≈0.05) y no inyecta conocimiento. El término de CE
    sobre la respuesta del teacher es la señal directa de incorporación;
    la divergencia y el término de workspace moldean CÓMO se codifica.
    """
    device = next(model.parameters()).device
    ws_loss = None
    if peso_ws > 0:
        if lens is None or not capas or not vocab_ids:
            raise ValueError("peso_ws > 0 requiere lens, capas y vocab_ids")
        unembed = model.get_output_embeddings().weight.detach()
        ws_loss = WorkspaceDistillLoss(lens, capas, unembed, vocab_ids,
                                       temperatura=temperatura_ws, device=device)

    # T4 (14.5 GB): sin checkpointing, las activaciones del student en
    # secuencias largas + logits full-vocab fp32 de la divergencia no entran.
    if device.type == "cuda":
        try:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except (AttributeError, TypeError):
            pass

    import random as _random
    rng = _random.Random(seed)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    historia = {"gkd": [], "ws": [], "ce": []}

    for ep in range(epochs):
        pares = list(pares)
        rng.shuffle(pares)
        for par in pares:
            on_policy = torch.rand(1, generator=gen).item() < lam
            if on_policy:
                ids_s = tokenizer(par.prompt_student, return_tensors="pt").to(device)
                with torch.no_grad():
                    muestras = model.generate(
                        **ids_s, max_new_tokens=max_new_tokens_onpolicy,
                        do_sample=True, top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id)
                respuesta = tokenizer.decode(
                    muestras[0, ids_s.input_ids.shape[1]:], skip_special_tokens=True)
                if not respuesta.strip():
                    continue
            else:
                respuesta = par.respuesta_teacher

            # misma respuesta bajo ambos prompts; la divergencia se computa
            # solo sobre los tokens de la respuesta
            def tokenizar(prompt):
                p = tokenizer(prompt, return_tensors="pt",
                              truncation=True, max_length=max_length)
                r = tokenizer(respuesta, return_tensors="pt", add_special_tokens=False,
                              truncation=True, max_length=max_resp_tokens)
                ids = torch.cat([p.input_ids, r.input_ids], dim=1).to(device)
                n_prompt = p.input_ids.shape[1]
                return ids, n_prompt

            ids_t, np_t = tokenizar(par.prompt_teacher)
            ids_e, np_e = tokenizar(par.prompt_student)
            n_resp = ids_t.shape[1] - np_t

            ctx_teacher = (model.disable_adapter()
                           if hasattr(model, "disable_adapter")
                           else contextlib.nullcontext())
            with torch.no_grad(), ctx_teacher:
                out_t = model(input_ids=ids_t, output_hidden_states=ws_loss is not None)
            out_e = model(input_ids=ids_e, output_hidden_states=ws_loss is not None)

            # logits que predicen los tokens de la respuesta
            lt = out_t.logits[:, np_t - 1: np_t - 1 + n_resp].detach()
            le = out_e.logits[:, np_e - 1: np_e - 1 + n_resp]
            mask = torch.ones(lt.shape[:2], device=device)
            loss_gkd = perdida_gkd(lt, le, mask, divergencia=divergencia)
            historia["gkd"].append(float(loss_gkd.detach()))
            loss = peso_gkd * loss_gkd  # peso_gkd=0 -> brazo ce_only

            if peso_ce > 0 and not on_policy:  # CE solo sobre data del teacher
                labels = ids_e[:, np_e: np_e + n_resp]
                ce = F.cross_entropy(
                    le.reshape(-1, le.shape[-1]).float(), labels.reshape(-1))
                historia["ce"].append(float(ce.detach()))
                loss = loss + peso_ce * ce

            if ws_loss is not None:
                ht = {l: out_t.hidden_states[l + 1][:, np_t: np_t + n_resp]
                      for l in ws_loss.capas}
                he = {l: out_e.hidden_states[l + 1][:, np_e: np_e + n_resp]
                      for l in ws_loss.capas}
                lw = ws_loss(ht, he)
                historia["ws"].append(float(lw.detach()))
                loss = loss + peso_ws * lw

            del out_t  # liberar los logits del teacher antes del backward
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            del out_e, loss, lt, le
            if device.type == "cuda":
                torch.cuda.empty_cache()
        log(f"  ks epoch {ep + 1}/{epochs} gkd={historia['gkd'][-1]:.4f}"
            + (f" ce={historia['ce'][-1]:.4f}" if historia["ce"] else "")
            + (f" ws={historia['ws'][-1]:.4f}" if historia["ws"] else ""))
    try:
        model.gradient_checkpointing_disable()  # no penalizar el generate posterior
    except AttributeError:
        pass
    model.eval()
    return historia
