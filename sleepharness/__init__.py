"""sleep-harness: paradigma de sueño (arXiv:2606.03979) sobre jlens-harness.

Wake = el agente sujeto corre tareas instrumentado con el Jacobian lens.
Sleep = NREM (consolidación con Knowledge Seeding) + REM (dreaming filtrado
por J-Space). Todo fine-tuning corre en Google Colab (ver notebooks/); el
código local prepara bundles, lee el pizarrón y evalúa.
"""

from . import _compat  # noqa: F401  (hace importable `harness` de jlens-harness)

__version__ = "0.1.0"
