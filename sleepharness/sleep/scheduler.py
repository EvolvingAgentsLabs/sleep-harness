"""Cuándo dormir: el calendario de consolidación por niveles (§3.2 del paper).

Cada nivel de memoria tiene un chunk C^(l); el sueño se dispara en los pasos
divisibles por algún C^(l) y consolida exactamente los niveles vencidos, del
más rápido al más lento. Antes de actualizar un bloque hay que consolidar su
conocimiento hacia el nivel siguiente (protocolo compute-consolidate-update).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SleepScheduler:
    chunks: dict[str, int]                 # nivel -> C^(l), ej. {"rapida": 4, "media": 16, "lenta": 64}
    paso: int = 0
    historial: list[dict] = field(default_factory=list)

    def __post_init__(self):
        orden = sorted(self.chunks.items(), key=lambda kv: kv[1])
        self._niveles = [n for n, _ in orden]
        for (na, ca), (nb, cb) in zip(orden, orden[1:]):
            if cb % ca != 0:  # supuesto del paper: C^(l) divisible por C^(l-1)
                raise ValueError(f"C[{nb}]={cb} debe ser divisible por C[{na}]={ca}")

    def registrar_paso(self) -> list[str]:
        """Avanza un paso de wake; devuelve los niveles a consolidar ahora
        (orden rápido→lento; vacío si no toca dormir)."""
        self.paso += 1
        vencidos = [n for n in self._niveles if self.paso % self.chunks[n] == 0]
        if vencidos:
            self.historial.append({"paso": self.paso, "niveles": vencidos})
        return vencidos

    def proximo_sueno(self) -> int:
        """Pasos que faltan para el próximo sueño (del nivel más rápido)."""
        c = min(self.chunks.values())
        return c - (self.paso % c)
