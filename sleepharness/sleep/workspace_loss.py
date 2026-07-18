"""Workspace Distillation (Idea 3 del crossover).

La destilación del paper compara distribuciones de SALIDA (logits). Acá se
agrega un término que compara el WORKSPACE: en capas medias-tardías, la
distribución lens-decodificada del teacher y del student deben coincidir:

    L_ws = mean_l KL( softmax(U_sub · J_l · h_teacher / T)
                    ∥ softmax(U_sub · J_l · h_student / T) )

restringida a un sub-vocabulario (conceptos rastreados + top del teacher)
para que sea barata. Enseña no solo QUÉ responder sino QUÉ conceptos
encender en el pizarrón. Como teacher y student comparten el modelo base
(± LoRA), el mismo J_l aplica a ambos.

Optimización clave: W_l = U_sub @ J_l se precomputa una vez ([V_sub, d]);
cada paso de entrenamiento es un matmul chico por capa.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sub_vocab(tokenizer, conceptos: list[str], extra_ids: list[int] | None = None) -> list[int]:
    """Ids de vocabulario para los conceptos que son token único + extras."""
    ids = []
    for c in conceptos:
        enc = tokenizer(c, add_special_tokens=False).input_ids
        if len(enc) == 1:
            ids.append(enc[0])
    ids += extra_ids or []
    return sorted(set(ids))


def hiddens_de(model, input_ids: torch.Tensor, capas: list[int],
               con_grad: bool = False) -> dict[int, torch.Tensor]:
    """Estados del residual stream a la salida de cada capa pedida.

    Con output_hidden_states=True, hidden_states[l+1] es la salida de la
    capa l (hidden_states[0] son los embeddings) — misma convención de
    lectura que ActivationRecorder de jlens sobre model.layers[l].
    """
    ctx = torch.enable_grad() if con_grad else torch.no_grad()
    with ctx:
        out = model(input_ids=input_ids, output_hidden_states=True)
    return {l: out.hidden_states[l + 1] for l in capas}


class WorkspaceDistillLoss:
    def __init__(self, lens, capas: list[int], unembed_weight: torch.Tensor,
                 vocab_ids: list[int], *, temperatura: float = 2.0,
                 device=None, dtype=torch.float32):
        """lens: jlens.JacobianLens; unembed_weight: lm_head.weight [V, d]."""
        self.capas = [l for l in capas if l in lens.jacobians]
        if not self.capas:
            raise ValueError("ninguna capa pedida tiene Jacobiano en el lens")
        self.temperatura = temperatura
        self.vocab_ids = torch.tensor(vocab_ids, dtype=torch.long)
        device = device or unembed_weight.device
        u_sub = unembed_weight[self.vocab_ids].to(device=device, dtype=dtype)  # [V_sub, d]
        self.W = {
            l: (u_sub @ lens.jacobians[l].to(device=device, dtype=dtype))      # [V_sub, d]
            for l in self.capas
        }

    def lens_logits(self, hidden: torch.Tensor, capa: int) -> torch.Tensor:
        """[B, T, d] -> [B, T, V_sub] en el sub-vocabulario."""
        return hidden.to(self.W[capa].dtype) @ self.W[capa].T

    def __call__(self, hidden_teacher: dict[int, torch.Tensor],
                 hidden_student: dict[int, torch.Tensor],
                 mask: torch.Tensor | None = None) -> torch.Tensor:
        """KL(teacher ∥ student) sobre el sub-vocabulario, promediada en
        capas y posiciones (mask [B, T] opcional, 1 = posición de respuesta)."""
        total = 0.0
        for l in self.capas:
            log_p = F.log_softmax(self.lens_logits(hidden_teacher[l], l).detach()
                                  / self.temperatura, dim=-1)
            log_q = F.log_softmax(self.lens_logits(hidden_student[l], l)
                                  / self.temperatura, dim=-1)
            kl = (log_p.exp() * (log_p - log_q)).sum(-1)          # [B, T]
            if mask is not None:
                m = mask.float()
                kl = (kl * m).sum() / m.sum().clamp(min=1.0)
            else:
                kl = kl.mean()
            total = total + kl
        return total / len(self.capas)
