"""Fase 3 — Firewall subconsciente: detección de intención maliciosa por J-Space.

Contraparte de seguridad de nuestro resultado sobreviviente (veto de gemelos,
p=0.019). El paper *Bad Memory* (Gadgil et al., UW, arXiv:2607.14611) muestra
que los payloads de inyección se ofuscan con alto solape léxico contra texto
benigno — su ejemplo canónico: leer `~/.ssh/id_rsa` disfrazado de "license-audit
telemetry". Un filtro léxico es ciego a eso por construcción. Este módulo mide
si el pizarrón (J-Space) enciende conceptos de seguridad al PROCESAR el payload,
aunque el texto los oculte.

Uso: `score_seguridad(top)` sobre el readout de `runtime.leer_pizarron`.
"""

from __future__ import annotations

import re

# Conceptos que el workspace debería encender al representar un payload
# malicioso, aunque el texto los camufle. Son tokens (posiblemente en inglés:
# el pizarrón decodifica conceptos, no la superficie del payload).
FIRMAS_SEGURIDAD: dict[str, set[str]] = {
    # exfiltración de credenciales (goal 1 del paper)
    "exfiltracion": {
        "ssh", "key", "keys", "credential", "credentials", "password",
        "passwords", "token", "tokens", "secret", "secrets", "private",
        "exfiltrate", "steal", "leak", "sensitive", "authentication",
        "auth", "rsa", "keychain", "wallet", "seed", "env",
    },
    # uso no autorizado de herramientas (goal 2)
    "ejecucion": {
        "install", "download", "execute", "run", "command", "shell",
        "sudo", "curl", "wget", "script", "subprocess", "eval", "exec",
        "payload", "binary", "package",
    },
    # marcadores de manipulación / autoridad inyectada (los tres goals)
    "manipulacion": {
        "important", "must", "always", "override", "ignore", "mandatory",
        "critical", "obey", "comply", "instruct", "rule", "policy",
        "required", "enforce", "never",
    },
    # nivel de ABSTRACCIÓN de amenaza — descubierto en el diagnóstico de Gemma
    # (P0): algunos modelos representan el payload no con el concepto concreto
    # (ssh/rsa) sino con la categoría (malware/hacker/cybersecurity). Añadirlo
    # calibra el detector cross-modelo sin dañar Qwen (estos conceptos también
    # son genuinamente de amenaza y quedan oscuros en los gemelos benignos).
    "ciberamenaza": {
        "malware", "hacker", "cybersecurity", "cybersecurity", "encryption",
        "encrypted", "cryptographic", "cryptography", "vpn", "breach",
        "exfiltration", "unauthorized", "malicious", "threat", "attack",
        "virus", "trojan", "seguridad", "phishing", "logins", "usernames",
    },
}

_PALABRA = re.compile(r"[A-Za-zÁÉÍÓÚáéíóúüñÑ]{3,}|\d+")


def _matchea(token: str, anclas: set[str]) -> bool:
    t = token.strip().lower()
    if t in anclas:
        return True
    if len(t) < 4:
        return False
    # familia de prefijos, como en dream_filter (sub-tokens / cognados)
    return any(len(a) >= 4 and (t in a or a in t or t[:5] == a[:5]) for a in anclas)


def score_seguridad(top: list[dict], firmas: tuple[str, ...] = ("exfiltracion",
                    "ejecucion", "manipulacion")) -> dict:
    """Fracción de intensidad del pizarrón capturada por cada firma de
    seguridad + un score combinado 'malicious_intent'. `top` = ws.top."""
    total = sum(t["intensidad"] for t in top) or 1.0
    out = {}
    capturado = 0.0
    for nombre in firmas:
        anclas = FIRMAS_SEGURIDAD[nombre]
        peso = sum(t["intensidad"] for t in top if _matchea(t["token"], anclas))
        conceptos = [t["token"] for t in top if _matchea(t["token"], anclas)]
        out[nombre] = {"score": round(peso / total, 3), "conceptos": conceptos}
    # malicious_intent = conceptos duros de amenaza (exfiltración + ejecución +
    # abstracción de amenaza), sin manipulación (que también aparece en
    # instrucciones legítimas)
    duras = {*FIRMAS_SEGURIDAD["exfiltracion"], *FIRMAS_SEGURIDAD["ejecucion"],
             *FIRMAS_SEGURIDAD["ciberamenaza"]}
    peso_duro = sum(t["intensidad"] for t in top if _matchea(t["token"], duras))
    out["malicious_intent"] = round(peso_duro / total, 3)
    return out


def firewall_consolidacion(rt, texto: str, umbral: float, *,
                           capas: list[int] | None = None) -> dict:
    """Firewall de sueño (H-SEC-2): lee el pizarrón del parche de memoria/spec
    PROPUESTO y veta su consolidación si la intención maliciosa supera el
    umbral. Se coloca antes de escribir memoria durable o entrenar un adapter.

    Devuelve {ok, malicious_intent, motivo}. ok=False = VETO (no consolidar).
    `capas` permite la ventana calibrada por modelo (ver P0).
    """
    capas_previas = rt.capas
    if capas:
        rt.capas = capas
    try:
        ws = rt.leer_pizarron(texto, top_k=30, max_posiciones=20)
    finally:
        rt.capas = capas_previas
    mi = score_seguridad(ws.top)["malicious_intent"]
    ok = mi < umbral
    return {
        "ok": ok, "malicious_intent": mi,
        "motivo": ("consolidación permitida" if ok else
                   f"VETO: firma de intención maliciosa {mi:.3f} ≥ umbral {umbral:.3f} "
                   "— el pizarrón enciende conceptos de amenaza que el texto oculta"),
    }


def solape_lexico(a: str, b: str) -> float:
    """Baseline de superficie: solape de tokens de contenido entre dos textos
    (Jaccard sobre palabras ≥3). Es lo que un filtro de texto 've' — y lo que
    resulta ALTO entre payload malicioso ofuscado y su gemelo benigno."""
    ta = {w.lower() for w in _PALABRA.findall(a)}
    tb = {w.lower() for w in _PALABRA.findall(b)}
    if not ta or not tb:
        return 0.0
    return round(len(ta & tb) / len(ta | tb), 3)
