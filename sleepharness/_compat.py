"""Bootstrap de imports: jlens-harness no es un paquete instalable.

Busca el repo hermano (o el indicado por SLEEPHARNESS_JLENS_HARNESS) y lo
inserta en sys.path para poder importar `harness.runtime`, `harness.signatures`,
etc. En Colab el zip del bundle preserva la estructura de hermanos, así que
este mismo mecanismo funciona sin cambios.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _localizar_jlens_harness() -> Path | None:
    env = os.environ.get("SLEEPHARNESS_JLENS_HARNESS")
    candidatos = [Path(env)] if env else []
    aqui = Path(__file__).resolve()
    # sleep-harness/sleepharness/_compat.py -> hermano ../jlens-harness
    candidatos.append(aqui.parents[2] / "jlens-harness")
    candidatos.append(Path.cwd() / "jlens-harness")
    for c in candidatos:
        if (c / "harness" / "runtime.py").exists():
            return c
    return None


_repo = _localizar_jlens_harness()
if _repo is not None and str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

try:  # harness queda disponible para el resto del paquete
    import harness  # noqa: F401

    JLENS_HARNESS_DISPONIBLE = True
except ImportError:  # módulos puros (router, scheduler, rewards) siguen andando
    JLENS_HARNESS_DISPONIBLE = False
