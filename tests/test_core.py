"""Tests de lógica pura: corren sin modelo ni GPU."""

import pytest
import torch

from sleepharness.eval.forgetting import comparar, puntuar
from sleepharness.eval.incorporation import acierta
from sleepharness.eval.verifiers import verificar_numerico
from sleepharness.evolution.router import Ruta, decidir, sintetizar_parche
from sleepharness.sleep.dream_filter import descartar_alucinaciones, seleccionar
from sleepharness.sleep.dreams import Dream
from sleepharness.sleep.nrem import (levenshtein_norm, perdida_gkd, r_abs,
                                     recompensa_lti)
from sleepharness.sleep.scheduler import SleepScheduler
from sleepharness.wake.tools import calc


# ---------------- scheduler ----------------

def test_scheduler_niveles():
    s = SleepScheduler({"rapida": 2, "media": 4, "lenta": 8})
    vencidos = [s.registrar_paso() for _ in range(8)]
    assert vencidos[1] == ["rapida"]
    assert vencidos[3] == ["rapida", "media"]
    assert vencidos[7] == ["rapida", "media", "lenta"]
    assert vencidos[0] == vencidos[2] == []


def test_scheduler_divisibilidad():
    with pytest.raises(ValueError):
        SleepScheduler({"rapida": 3, "media": 4})


# ---------------- router (Idea 2) ----------------

def _firmas(computo=0.0, missing=0.0):
    return {"computo": {"score": computo, "conceptos": []},
            "missing_info": {"score": missing, "conceptos": []}}


def test_router_ok_es_nada():
    d = decidir({"ok": True}, _firmas(computo=0.5))
    assert d.ruta is Ruta.NADA


def test_router_missing_info_es_contexto():
    d = decidir({"ok": False}, _firmas(missing=0.3))
    assert d.ruta is Ruta.CONTEXTO


def test_router_missing_no_compite_con_firma_tarea():
    # exp2 ciclo 2: computo=0.209 > missing=0.144, pero missing supera el
    # umbral absoluto -> CONTEXTO igual (en tareas de cómputo la firma de
    # tarea siempre enciende; no debe bloquear la ruta de contexto)
    d = decidir({"ok": False}, _firmas(computo=0.209, missing=0.144))
    assert d.ruta is Ruta.CONTEXTO


def test_router_pide_datos_gana_a_tool():
    # exp2 ciclo 1: la salida rechaza por falta de datos aunque el pizarrón
    # no encienda missing_info -> la señal del verificador manda
    d = decidir({"ok": False, "pide_datos": True}, _firmas(computo=0.169))
    assert d.ruta is Ruta.CONTEXTO


def test_router_truncada_en_pleno_calculo_es_formato():
    # exp2 corrida 2, ciclo 2 (valores reales): salida cortada definiendo
    # variables, n_numeros=10, firma bajo umbral por léxico angosto
    d = decidir({"ok": False, "n_numeros": 10}, _firmas(computo=0.084),
                truncada=True)
    assert d.ruta is Ruta.FORMATO


def test_router_intento_sin_firma_es_tool():
    # números en la salida = intento de cómputo, aunque el pizarrón no llegue
    # al umbral -> TOOL, no PESOS
    d = decidir({"ok": False, "n_numeros": 5}, _firmas(computo=0.05))
    assert d.ruta is Ruta.TOOL


def test_router_pesos_solo_como_ultimo_recurso():
    d = decidir({"ok": False, "n_numeros": 1}, _firmas(computo=0.05))
    assert d.ruta is Ruta.PESOS


def test_parche_formato_brevedad_y_budget():
    d = decidir({"ok": False, "n_numeros": 10}, _firmas(computo=0.3),
                truncada=True)
    spec = {"instructions": "Sos analista.", "data": "d", "tools": [],
            "output_budget": 160}
    parche = sintetizar_parche(d, spec)
    assert parche["output_budget"] == 320
    assert "breve" in parche["instructions"]
    assert sintetizar_parche(d, parche) is None  # idempotente


def test_router_firma_sana_falla_es_tool():
    d = decidir({"ok": False}, _firmas(computo=0.4))
    assert d.ruta is Ruta.TOOL


def test_router_sin_firma_es_pesos():
    d = decidir({"ok": False}, _firmas())
    assert d.ruta is Ruta.PESOS


def test_parche_contexto_inyecta_datos():
    d = decidir({"ok": False}, _firmas(missing=0.3))
    spec = {"instructions": "x", "data": "", "tools": [], "output_budget": 160}
    parche = sintetizar_parche(d, spec, datos_reales="Costo: $20")
    assert parche["data"] == "Costo: $20"
    assert sintetizar_parche(d, parche, datos_reales="Costo: $20") is None  # idempotente


def test_parche_tool_agrega_calc():
    d = decidir({"ok": False}, _firmas(computo=0.4))
    spec = {"instructions": "x", "data": "d", "tools": [], "output_budget": 160}
    parche = sintetizar_parche(d, spec)
    assert parche["tools"][0]["name"] == "calc"
    assert "TOOL:" in parche["instructions"]


# ---------------- filtrado de dreams (Idea 1) ----------------

def _dreams():
    return [
        Dream("a", "qa", 0, scores={"gradiente": 1.0, "jspace": 0.30}),
        Dream("b", "qa", 0, scores={"gradiente": 3.0, "jspace": 0.02}),
        Dream("c", "qa", 0, scores={"gradiente": 2.0, "jspace": 0.20}),
        Dream("d", "qa", 0, scores={"gradiente": 0.5, "jspace": 0.01}),
    ]


def test_seleccion_grad_vs_jspace():
    ds = _dreams()
    assert [d.texto for d in seleccionar(ds, k=1, b_random=0, modo="grad")] == ["b"]
    assert [d.texto for d in seleccionar(ds, k=1, b_random=0, modo="jspace")] == ["a"]


def test_seleccion_combinada_promedia_rangos():
    # "d" es el peor en ambos rankings: nunca entra al top combinado
    top = seleccionar(_dreams(), k=2, b_random=0, modo="combinado")
    assert len(top) == 2 and "d" not in {d.texto for d in top}
    for d in top:
        assert "combinado" in d.scores


def test_seleccion_none_respeta_k_mas_b():
    assert len(seleccionar(_dreams(), k=2, b_random=1, modo="none")) == 3


def test_descartar_alucinaciones():
    ok, fuera = descartar_alucinaciones(_dreams(), umbral=0.05)
    assert {d.texto for d in ok} == {"a", "c"}
    assert {d.texto for d in fuera} == {"b", "d"}


def test_anclas_de_contexto_contenido_y_numeros():
    from sleepharness.sleep.dream_filter import anclas_de_contexto
    ctx = ("Vantar Dynamics fue fundada en 2019 en Rosario. La membrana "
           "opera a 41 grados y reduce el consumo un 37 por ciento.")
    anclas = anclas_de_contexto(ctx)
    assert {"vantar", "membrana", "rosario", "2019", "41", "37"} <= anclas
    assert "en" not in anclas and "por" not in anclas  # stopwords/cortas fuera


def test_puntuar_top_con_anclas():
    from sleepharness.sleep.dream_filter import puntuar_top_con_anclas
    top = [{"token": "membrana", "intensidad": 6.0},
           {"token": "41", "intensidad": 3.0},
           {"token": "music", "intensidad": 1.0}]
    # dream fiel: 9 de 10 de intensidad en anclas del contexto
    assert puntuar_top_con_anclas(top, {"membrana", "41"}) == 0.9
    # dream alucinado: nada del contexto encendido
    assert puntuar_top_con_anclas(top, {"heliox", "rosario"}) == 0.0


def test_score_lexico_y_modo_lexical():
    # control V2+: baseline léxico sin lens, y su modo en seleccionar
    from sleepharness.sleep.dream_filter import score_lexico, seleccionar
    ctx = "Vantar Dynamics fue fundada en 2019 en Rosario."
    fiel = Dream("La empresa Vantar se fundó en Rosario en 2019.", "qa", 0)
    aluc = Dream("El clima estuvo templado y la música sonaba fuerte.", "qa", 0)
    assert score_lexico(fiel, ctx) > 0.4
    assert score_lexico(aluc, ctx) == 0.0
    top = seleccionar([fiel, aluc], k=1, b_random=0, modo="lexical")
    assert top[0] is fiel


def test_matcheo_subtokens_y_cognados():
    # calibración exp1: nombres propios sub-tokenizados y conceptos en inglés
    from sleepharness.sleep.dream_filter import puntuar_top_con_anclas
    anclas = {"vantar", "membrana", "rosario", "41"}
    top = [{"token": " Vant", "intensidad": 4.0},      # sub-token de Vantar
           {"token": "membrane", "intensidad": 4.0},   # cognado EN
           {"token": " 41", "intensidad": 2.0},        # número exacto
           {"token": " 37", "intensidad": 5.0},        # número NO anclado
           {"token": "music", "intensidad": 5.0}]      # irrelevante
    assert puntuar_top_con_anclas(top, anclas) == 0.5  # 10 de 20


def test_limpiar_y_degenerado():
    from sleepharness.sleep.dreams import es_degenerado, limpiar_dream
    assert limpiar_dream("<think>\nhmm\n</think>\n\nP: ¿qué? / R: eso") == "P: ¿qué? / R: eso"
    assert es_degenerado("membrane " * 40)
    assert not es_degenerado("uno dos tres cuatro cinco seis siete ocho nueve "
                             "diez once doce trece")


# ---------------- rewards LTI (Ec. 3-4) ----------------

def test_levenshtein_norm():
    assert levenshtein_norm("abc", "abc") == 0.0
    assert levenshtein_norm("abc", "xyz") == 1.0


def test_r_abs_umbral():
    assert r_abs("hola mundo", "hola mundo") == 1.0
    assert r_abs("aaaa", "zzzz", z0=0.6) == 0.0  # dist 1.0 > z0


def test_recompensa_lti_mezcla():
    r = recompensa_lti("hola mundo", "hola mundo", gamma=0.5)
    assert r == 1.0
    r2 = recompensa_lti("x", "hola mundo cruel", gamma=0.5)
    assert 0.0 <= r2 < 0.5


# ---------------- GKD ----------------

def test_gkd_cero_si_iguales():
    logits = torch.randn(1, 5, 11)
    mask = torch.ones(1, 5)
    for div in ("fkl", "rkl", "jsd"):
        assert perdida_gkd(logits, logits.clone(), mask, divergencia=div) < 1e-5


def test_gkd_positiva_si_distintas():
    a, b = torch.randn(1, 5, 11), torch.randn(1, 5, 11)
    assert perdida_gkd(a, b, torch.ones(1, 5), divergencia="jsd") > 0


# ---------------- verificador y sondas ----------------

def test_verificar_numerico_pricing():
    v = verificar_numerico("El precio óptimo es $47.50 y la ganancia $3,025 por mes.",
                           {"precio": 47.5, "ganancia": 3025})
    assert v["ok"]
    v2 = verificar_numerico("Necesito más información sobre el costo del producto.",
                            {"precio": 47.5})
    assert not v2["ok"] and v2["pide_datos"]


def test_pide_datos_formulacion_negativa():
    # regresión exp2 ciclo 1 (salida real del modelo)
    salida = ("Como analista de negocios, debo ser honesto contigo: **no puedo "
              "darte un precio exacto ni una ganancia esperada específica** sin "
              "datos fundamentales de tu negocio.")
    assert verificar_numerico(salida, {"precio": 47.5})["pide_datos"]


def test_puntuar_y_comparar():
    assert puntuar("La capital es París.", {"contains": ["par"]}) == 1.0
    assert puntuar("Au", {"regex": r"\bAu\b"}) == 1.0
    r = comparar({"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 1.0})
    assert r["olvido_medio"] == 0.5 and r["sondas_olvidadas"] == ["a"]


def test_acierta_alternativas():
    assert acierta("Fue Lucía Ferreyra quien la fundó", ["Lucía Ferreyra", "Ferreyra"])
    assert not acierta("no sé", "Ferreyra")


# ---------------- calc (tool) ----------------

def test_calc_aritmetica():
    assert calc("(30 - 5*1.5)*(100 + 20*1.5)") == "2925"
    assert calc("-100/(2*-100)") == "0.5"
    assert calc("__import__('os')").startswith("error")


def test_dedupe_llamadas():
    # exp2 corrida 3 ciclo 4: el modelo repite el mismo bloque de TOOLs
    from sleepharness.wake.tools import dedupe_llamadas
    llamadas = [("calc", "40 - 20"), ("calc", "100 + 20 * 2"),
                ("calc", "40 - 20"), ("calc", "100 + 20 * 2")]
    assert dedupe_llamadas(llamadas) == [("calc", "40 - 20"),
                                         ("calc", "100 + 20 * 2")]
