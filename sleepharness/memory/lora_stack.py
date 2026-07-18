"""Jerarquía de memoria como stack de adaptadores LoRA (proxy de CMS/Hope).

Aproximación práctica de la jerarquía de frecuencias del paper (§2.2) sobre
un Transformer HF estándar:

- memoria rápida  = el contexto/KV (no vive acá; es el prompt de la sesión)
- memoria media   = adaptadores LoRA, uno nuevo por ciclo de sueño
                    ("expansión de parámetros", §3.2: el expert low-rank nuevo)
- memoria lenta   = el modelo base con adaptadores viejos fusionados

Reglas del protocolo compute-consolidate-update (§3.3):
- al dormir se agrega un adaptador nuevo y SOLO ese es entrenable
  (los viejos congelados => el conocimiento previo no se pisa);
- tras consolidar, la "poda sináptica" borra adaptadores del nivel rápido;
- periódicamente (nivel "lenta") los adaptadores estables se fusionan al base.

Requiere `peft` (extra [train]); pensado para correr en Colab, aunque también
anda local para cargar/evaluar adaptadores ya entrenados.
"""

from __future__ import annotations

import json
from pathlib import Path


class LoraStack:
    def __init__(self, model, lora_config: dict, adapter_prefix: str = "sleep"):
        from peft import LoraConfig, get_peft_model

        self._LoraConfig = LoraConfig
        self.prefix = adapter_prefix
        self.historial: list[dict] = []
        self._n = 0
        nombre = self._nombre(0)
        self.model = get_peft_model(model, LoraConfig(**lora_config), adapter_name=nombre)
        self.lora_config = lora_config
        self.activos = [nombre]
        self.historial.append({"evento": "nuevo", "adaptador": nombre})

    def _nombre(self, i: int) -> str:
        return f"{self.prefix}_{i}"

    @property
    def actual(self) -> str:
        return self.activos[-1]

    # ---- expansión de parámetros (nuevo expert low-rank por ciclo) ----

    def nuevo_adaptador(self) -> str:
        """Agrega un adaptador nuevo (congela los anteriores) y lo activa."""
        self._n += 1
        nombre = self._nombre(self._n)
        self.model.add_adapter(nombre, self._LoraConfig(**self.lora_config))
        self.activos.append(nombre)
        self.entrenar_solo(nombre)
        self.historial.append({"evento": "nuevo", "adaptador": nombre})
        return nombre

    def entrenar_solo(self, nombre: str) -> None:
        """Deja entrenables únicamente los parámetros del adaptador `nombre`.

        Todos los adaptadores en `self.activos` siguen participando del
        forward (conocimiento previo presente), pero solo el nuevo recibe
        gradiente — el requisito de estabilidad/plasticidad de §3.3.
        """
        self.model.base_model.set_adapter(self.activos)  # forward: todos
        for n, p in self.model.named_parameters():
            p.requires_grad = f".{nombre}." in n or n.endswith(f".{nombre}")

    # ---- consolidación al nivel lento ----

    def fusionar_al_base(self, nombres: list[str] | None = None) -> None:
        """Fusiona adaptadores estables en los pesos base (memoria lenta)."""
        nombres = nombres or self.activos[:-1] or self.activos
        self.model.base_model.merge_adapter(adapter_names=nombres)
        self.historial.append({"evento": "fusion", "adaptadores": list(nombres)})

    # ---- poda sináptica (§3.3 paso c: reset de los experts rápidos) ----

    def poda_sinaptica(self, nombres: list[str]) -> None:
        for n in nombres:
            self.model.delete_adapter(n)
            self.activos.remove(n)
        self.historial.append({"evento": "poda", "adaptadores": list(nombres)})

    # ---- persistencia ----

    def guardar(self, dir_out: str | Path) -> None:
        dir_out = Path(dir_out)
        dir_out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(dir_out))
        (dir_out / "stack.json").write_text(json.dumps(
            {"activos": self.activos, "historial": self.historial,
             "lora_config": self.lora_config}, ensure_ascii=False, indent=2))

    def parametros_entrenables(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
