"""Fase activa (wake): el agente sujeto corre tareas instrumentado.

Envuelve harness.Runtime + AgentSpec + TraceWriter de jlens-harness y
acumula, además de la traza JSONL, el "diario de sesión": los contextos
vistos y los veredictos, que son la materia prima del sueño (NREM consolida
el diario; REM sueña sobre él; el router decide la ruta de mejora).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from harness.agentspec import AgentSpec, componer_prompt
from harness.trace import TraceWriter, registro

from ..signatures_ext import detectar_ext as detectar


@dataclass
class PasoWake:
    paso_id: str
    tarea: str
    salida: str
    truncada: bool
    verificacion: dict | None
    firmas_prompt: dict
    firmas_salida: dict | None


@dataclass
class Diario:
    """Lo que la fase de sueño necesita saber del día."""

    contextos: list[str] = field(default_factory=list)   # info nueva vista en wake
    pasos: list[PasoWake] = field(default_factory=list)

    def fallos(self) -> list[PasoWake]:
        return [p for p in self.pasos if p.verificacion and not p.verificacion.get("ok")]

    def to_dict(self) -> dict:
        return {
            "contextos": self.contextos,
            "pasos": [{
                "paso_id": p.paso_id, "tarea": p.tarea, "salida": p.salida,
                "truncada": p.truncada, "verificacion": p.verificacion,
                "firmas_prompt": p.firmas_prompt, "firmas_salida": p.firmas_salida,
            } for p in self.pasos],
        }


class WakeSession:
    def __init__(self, rt, spec: AgentSpec, trace_path: str | Path,
                 rastrear: list[str] | None = None):
        self.rt = rt
        self.spec = spec
        self.rastrear = rastrear or []
        self.trace = TraceWriter(trace_path)
        self.diario = Diario()
        self._n = 0

    def observar_contexto(self, texto: str) -> None:
        """Registra información nueva vista durante la sesión (candidata a
        consolidarse en sueño). Análogo del episodio en el hipocampo."""
        self.diario.contextos.append(texto)

    def paso(self, tarea: str, verificar=None, **kwargs) -> PasoWake:
        """Corre una tarea con la spec actual, lee el pizarrón y verifica."""
        self._n += 1
        paso_id = f"wake_{self._n:03d}_{time.strftime('%H%M%S')}"
        prompt = componer_prompt(self.spec, tarea, self.rt.arranque)
        kwargs.setdefault("max_posiciones_salida", 20)  # menos varianza de firmas
        r = self.rt.step(prompt, max_new_tokens=self.spec.output_budget,
                         rastrear=self.rastrear, **kwargs)
        verificacion = verificar(r.salida) if verificar else None
        self.trace.write(registro(paso_id, self.spec.to_dict(), tarea, r, verificacion))

        p = PasoWake(
            paso_id=paso_id, tarea=tarea, salida=r.salida, truncada=r.truncada,
            verificacion=verificacion,
            firmas_prompt=detectar(r.ws_prompt.top),
            firmas_salida=detectar(r.ws_salida.top) if r.ws_salida else None,
        )
        self.diario.pasos.append(p)
        return p

    def cerrar(self, diario_path: str | Path | None = None) -> Diario:
        self.trace.close()
        if diario_path:
            Path(diario_path).parent.mkdir(parents=True, exist_ok=True)
            Path(diario_path).write_text(
                json.dumps(self.diario.to_dict(), ensure_ascii=False, indent=2))
        return self.diario
