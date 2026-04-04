"""
=============================================================================
ENTREGA 2 - PARTE B: GMMHMM CON ESTRATEGIA DE ESCALADO ITERATIVO
=============================================================================
Mejora sobre Entrega 1 y Parte A: usar mezclas de gaussianas por estado
(GMMHMM) en lugar de una sola gaussiana (GaussianHMM), con covarianza
diagonal unicamente.

Estrategia de escalado iterativo:
  Fase 1: Prueba rapida - probar n_mix=1,2,3 con config de Entrega 1
           Si mejora >= 0.5pp -> evaluacion final
  Fase 2: Grid search sobre (n_mix, n_estados, prob_autolazo) con features fijas
           Si mejora >= 0.5pp -> evaluacion final
  Fase 3: Seleccion de features (ultimo recurso)
           -> evaluacion final con la mejor combinacion global

Evaluacion final: N=74 y N=47, matrices de confusion, comparativa con E1/E2.

Restriccion: solo covarianza diagonal ("diag").

Uso:
  python3 clasificador_digitos_v3.py
=============================================================================
"""

import sys
import os
import time
import json
import warnings
from collections import defaultdict

import numpy as np
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")
import logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# -- Importar funciones base de Entrega 1 --
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENTREGA1_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Entrega1")
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
EXTRACTOR_DIR = os.path.join(
    REPO_ROOT, "Extractores_adaptados", "Extractores", "Extractor Local"
)
sys.path.insert(0, EXTRACTOR_DIR)
sys.path.insert(0, ENTREGA1_DIR)

from clasificador_digitos import (
    cargar_base_datos, iterar_muestras, preprocesar, extraer_features,
    NormalizadorZScore, clasificar_lote,
    graficar_confusion, FEATURE_SUBSETS,
    _startprob_left_right, _transmat_left_right,
)

# =============================================================================
# CONFIGURACION
# =============================================================================

DB_PATH = os.path.join(REPO_ROOT, "e-BioDigit_DB", "e-BioDigit_DB")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "resultados")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_TRAIN_74 = 74
N_TRAIN_47 = 47
N_OPT_TRAIN = 78
N_OPT_VAL = 15

# -- Umbral de mejora (0.5 puntos porcentuales) --
MEJORA_THRESHOLD = 0.005

# -- Fase 1: config fija de Entrega 1, solo varia n_mix --
FASE1_N_MIX = [1, 2, 3]
FASE1_CONFIG = {
    "n_estados": 7,
    "prob_autolazo": 0.6,
    "features": "med",
}

# -- Fase 2: grid search (diag only, features fijas) --
FASE2_GRID = {
    "n_mix":         [1, 2, 3],
    "n_estados":     [5, 7, 10, 12, 15],
    "prob_autolazo": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

# -- Fase 3: subsets de features --
FASE3_FEATURES = ["min", "med", "full"]
CUSTOM_SUBSETS = {
    "med_noxy": [7, 8, 19, 20, 10, 4, 5, 6, 11, 21],          # 10D: med sin x,y
    "med_plus": [7, 8, 19, 20, 10, 4, 5, 6, 11, 21, 0, 1, 12, 13],  # 14D: med + drho, da
}

# -- Resultados de entregas anteriores (referencia) --
RESULTADOS_ANTERIORES = {
    "E1_N74": 0.9289,
    "E1_N47": 0.8978,
    "E2_N74": 0.9270,
    "E2_N47": 0.8989,
}


# =============================================================================
# FUNCIONES BASE (reutilizadas de v3 original)
# =============================================================================

def entrenar_gmmhmm_digito(secuencias, n_estados=7, n_mix=2,
                            tipo_covarianza="diag", n_iter=100,
                            n_restarts=5, prob_autolazo=0.6,
                            semilla_base=0, min_covar=1e-2):
    """Entrena un GMMHMM (o GaussianHMM si n_mix=1) left-right para un digito.

    Restriccion: solo covarianza diagonal.
    """
    if tipo_covarianza != "diag":
        raise ValueError(f"Solo covarianza 'diag' permitida, recibido: "
                         f"'{tipo_covarianza}'")

    longitudes = [len(s) for s in secuencias]
    X = np.concatenate(secuencias)

    mejor_score = -np.inf
    mejor_modelo = None

    for restart in range(n_restarts):
        semilla = semilla_base + restart

        if n_mix == 1:
            modelo = hmm.GaussianHMM(
                n_components=n_estados,
                covariance_type=tipo_covarianza,
                n_iter=n_iter,
                random_state=semilla,
                init_params="mc",
                params="mct",
                verbose=False,
            )
        else:
            modelo = hmm.GMMHMM(
                n_components=n_estados,
                n_mix=n_mix,
                covariance_type=tipo_covarianza,
                n_iter=n_iter,
                random_state=semilla,
                init_params="mcw",
                params="mcwt",
                min_covar=min_covar,
                verbose=False,
            )

        # Forzar topologia left-right
        modelo.startprob_ = _startprob_left_right(n_estados)
        modelo.transmat_ = _transmat_left_right(n_estados, prob_autolazo)

        try:
            modelo.fit(X, longitudes)
            score = modelo.score(X, longitudes)
            if score > mejor_score:
                mejor_score = score
                mejor_modelo = modelo
        except (ValueError, np.linalg.LinAlgError):
            continue

    return mejor_modelo, mejor_score


def preparar_features_por_usuario(db, user_ids, indices_features,
                                   n_resample=80, suavizar=True):
    """Pre-computa features agrupadas por usuario y digito."""
    datos = {}
    for uid in user_ids:
        muestras = []
        for _, digito, _, muestra in iterar_muestras(db, [uid]):
            x, y, ts = preprocesar(
                muestra["x"], muestra["y"], muestra["timestamp"],
                n_resample=n_resample, suavizar=suavizar)
            feats = extraer_features(x, y, muestra["presion"],
                                     indices_features=indices_features)
            muestras.append((feats, digito))
        datos[uid] = muestras
    return datos


def preparar_features_multisubset(db, user_ids, n_resample=80, suavizar=True):
    """Pre-computa features para TODOS los subsets (min, med, full).

    Returns:
        dict {subset_name: {uid: [(feats_array, digito), ...]}}
    """
    datos_todos = {name: {} for name in FEATURE_SUBSETS}

    for uid in user_ids:
        muestras_raw = []
        for _, digito, _, muestra in iterar_muestras(db, [uid]):
            x, y, ts = preprocesar(
                muestra["x"], muestra["y"], muestra["timestamp"],
                n_resample=n_resample, suavizar=suavizar)
            feats_all = extraer_features(x, y, muestra["presion"],
                                          indices_features=None)
            muestras_raw.append((feats_all, digito))

        for name, indices in FEATURE_SUBSETS.items():
            datos_todos[name][uid] = [
                (feats[:, indices], digito) for feats, digito in muestras_raw
            ]

    return datos_todos


def preparar_features_custom(db, user_ids, custom_subsets,
                              n_resample=80, suavizar=True):
    """Pre-computa features para subsets personalizados.

    Returns:
        dict {subset_name: {uid: [(feats_array, digito), ...]}}
    """
    datos_todos = {name: {} for name in custom_subsets}

    for uid in user_ids:
        muestras_raw = []
        for _, digito, _, muestra in iterar_muestras(db, [uid]):
            x, y, ts = preprocesar(
                muestra["x"], muestra["y"], muestra["timestamp"],
                n_resample=n_resample, suavizar=suavizar)
            feats_all = extraer_features(x, y, muestra["presion"],
                                          indices_features=None)
            muestras_raw.append((feats_all, digito))

        for name, indices in custom_subsets.items():
            datos_todos[name][uid] = [
                (feats[:, indices], digito) for feats, digito in muestras_raw
            ]

    return datos_todos


def reunir_secuencias(datos_por_usuario, user_ids):
    """Reune secuencias y etiquetas de un conjunto de usuarios."""
    secuencias = []
    etiquetas = []
    por_digito = defaultdict(list)
    for uid in user_ids:
        for feats, digito in datos_por_usuario[uid]:
            secuencias.append(feats)
            etiquetas.append(digito)
            por_digito[digito].append(feats)
    return secuencias, etiquetas, por_digito


def entrenar_y_evaluar(train_por_digito_norm, val_norm, val_labels,
                        n_estados=7, n_mix=2, tipo_covarianza="diag",
                        prob_autolazo=0.6, n_iter=50, n_restarts=3):
    """Entrena 10 modelos y evalua accuracy en validacion."""
    modelos = {}
    for digito in range(10):
        seqs = train_por_digito_norm[digito]
        if len(seqs) == 0:
            modelos[digito] = None
            continue
        modelo, _ = entrenar_gmmhmm_digito(
            seqs, n_estados=n_estados, n_mix=n_mix,
            tipo_covarianza=tipo_covarianza, n_iter=n_iter,
            n_restarts=n_restarts, prob_autolazo=prob_autolazo,
            semilla_base=digito * 100)
        modelos[digito] = modelo

    _, _, accuracy = clasificar_lote(modelos, val_norm, val_labels)
    return accuracy, modelos


# -- Checkpointing --

def _checkpoint_path(tag):
    return os.path.join(RESULTS_DIR, f"checkpoint_{tag}.json")


def _cargar_checkpoint(tag):
    path = _checkpoint_path(tag)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _guardar_checkpoint(tag, data):
    with open(_checkpoint_path(tag), "w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# FASE 1: PRUEBA RAPIDA DE GMMHMM
# =============================================================================

def fase1_prueba_rapida(train_por_digito_norm, val_norm, val_labels):
    """Prueba rapida: n_mix=1,2,3 con config fija de Entrega 1.

    Returns:
        (best_n_mix, best_acc, mejora, resultados_dict)
        mejora=True si GMMHMM (n_mix>1) supera baseline por >= MEJORA_THRESHOLD
    """
    tag = "fase1"
    ckpt = _cargar_checkpoint(tag)

    if ckpt and ckpt.get("completada"):
        print("  (cargado de checkpoint)", flush=True)
        resultados = ckpt["resultados"]
    else:
        resultados_previos = ckpt["resultados"] if ckpt else []
        n_done = len(resultados_previos)
        resultados = list(resultados_previos)

        for i, n_mix in enumerate(FASE1_N_MIX):
            if i < n_done:
                continue

            t0 = time.time()
            acc, _ = entrenar_y_evaluar(
                train_por_digito_norm, val_norm, val_labels,
                n_estados=FASE1_CONFIG["n_estados"], n_mix=n_mix,
                tipo_covarianza="diag",
                prob_autolazo=FASE1_CONFIG["prob_autolazo"],
                n_iter=50, n_restarts=5)
            elapsed = time.time() - t0

            resultado = {"n_mix": n_mix, "accuracy": acc, "tiempo": elapsed}
            resultados.append(resultado)

            print(f"    n_mix={n_mix}: {acc * 100:.2f}% ({elapsed:.0f}s)",
                  flush=True)

            _guardar_checkpoint(tag, {
                "resultados": resultados,
                "completada": i == len(FASE1_N_MIX) - 1,
            })

    # Analizar resultados
    acc_baseline = None
    best_gmm_acc = 0.0
    best_gmm_nmix = 1

    for r in resultados:
        if r["n_mix"] == 1:
            acc_baseline = r["accuracy"]
        else:
            if r["accuracy"] > best_gmm_acc:
                best_gmm_acc = r["accuracy"]
                best_gmm_nmix = r["n_mix"]

    # Determinar el mejor global (incluyendo n_mix=1)
    best_acc = max(r["accuracy"] for r in resultados)
    best_n_mix = [r for r in resultados if r["accuracy"] == best_acc][0]["n_mix"]

    mejora = best_gmm_acc > acc_baseline + MEJORA_THRESHOLD

    print(f"\n    Baseline (n_mix=1): {acc_baseline * 100:.2f}%")
    print(f"    Mejor GMMHMM: n_mix={best_gmm_nmix} -> {best_gmm_acc * 100:.2f}%")
    print(f"    Mejora >= {MEJORA_THRESHOLD*100:.1f}pp? "
          f"{'SI' if mejora else 'NO'}")

    return best_n_mix, best_acc, mejora, resultados


# =============================================================================
# FASE 2: GRID SEARCH (n_mix, n_estados, prob_autolazo)
# =============================================================================

def fase2_busqueda_hiperparametros(datos_feat, opt_train_users, opt_val_users,
                                     acc_referencia):
    """Grid search sobre (n_mix, n_estados, prob_autolazo), diag, features fijas.

    Args:
        datos_feat: dict {uid: [(feats, digito), ...]} para el subset fijo
        opt_train_users: lista de user_ids de entrenamiento
        opt_val_users: lista de user_ids de validacion
        acc_referencia: accuracy de referencia de Fase 1

    Returns:
        (mejor_config, mejor_acc, mejora, resultados)
        mejora=True si best > acc_referencia + MEJORA_THRESHOLD
    """
    tag = "fase2"

    # Generar todas las combinaciones
    configs = []
    for n_mix in FASE2_GRID["n_mix"]:
        for n_est in FASE2_GRID["n_estados"]:
            for p_auto in FASE2_GRID["prob_autolazo"]:
                configs.append({
                    "n_mix": n_mix,
                    "n_estados": n_est,
                    "prob_autolazo": p_auto,
                })

    total = len(configs)
    print(f"    Total configuraciones: {total}", flush=True)

    # Preparar datos (una sola vez, features fijas)
    train_seqs, train_labels, _ = reunir_secuencias(datos_feat, opt_train_users)
    val_seqs, val_labels, _ = reunir_secuencias(datos_feat, opt_val_users)

    norm = NormalizadorZScore()
    train_norm = norm.ajustar_y_transformar(train_seqs)
    val_norm = norm.transformar(val_seqs)

    train_por_digito = defaultdict(list)
    for seq, label in zip(train_norm, train_labels):
        train_por_digito[label].append(seq)

    # Cargar checkpoint
    ckpt = _cargar_checkpoint(tag)
    resultados = ckpt.get("resultados", []) if ckpt else []
    n_done = len(resultados)
    if n_done > 0:
        print(f"    Checkpoint: {n_done}/{total} ya evaluadas", flush=True)

    mejor_acc = max((r[1] for r in resultados), default=0.0)
    mejor_config = None
    for r in resultados:
        if r[1] == mejor_acc:
            mejor_config = r[0]

    for i, cfg in enumerate(configs):
        if i < n_done:
            continue

        t0 = time.time()
        acc, _ = entrenar_y_evaluar(
            train_por_digito, val_norm, val_labels,
            n_estados=cfg["n_estados"], n_mix=cfg["n_mix"],
            tipo_covarianza="diag",
            prob_autolazo=cfg["prob_autolazo"],
            n_iter=50, n_restarts=3)
        elapsed = time.time() - t0

        resultados.append((cfg, acc))

        marker = " ***" if acc > mejor_acc else ""
        print(f"    [{i+1:3d}/{total}] n_mix={cfg['n_mix']}, "
              f"n_est={cfg['n_estados']:2d}, "
              f"p={cfg['prob_autolazo']:.1f} "
              f"-> {acc*100:.2f}% ({elapsed:.0f}s){marker}", flush=True)

        if acc > mejor_acc:
            mejor_acc = acc
            mejor_config = cfg

        # Checkpoint cada 5 configs
        if (i + 1) % 5 == 0 or i == total - 1:
            _guardar_checkpoint(tag, {
                "resultados": resultados,
                "mejor_config": mejor_config,
                "mejor_acc": mejor_acc,
            })

    mejora = mejor_acc > acc_referencia + MEJORA_THRESHOLD

    print(f"\n    Referencia Fase 1: {acc_referencia * 100:.2f}%")
    print(f"    Mejor Fase 2: {mejor_acc * 100:.2f}% "
          f"(n_mix={mejor_config['n_mix']}, "
          f"n_est={mejor_config['n_estados']}, "
          f"p={mejor_config['prob_autolazo']})")
    print(f"    Mejora >= {MEJORA_THRESHOLD*100:.1f}pp? "
          f"{'SI' if mejora else 'NO'}")

    return mejor_config, mejor_acc, mejora, resultados


# =============================================================================
# FASE 3: SELECCION DE FEATURES (ULTIMO RECURSO)
# =============================================================================

def fase3_seleccion_features(db, user_ids, datos_multi, opt_train_users,
                               opt_val_users, config_base):
    """Prueba distintos subsets de features con la mejor config de Fase 2.

    Args:
        db: base de datos cargada
        user_ids: todos los user_ids
        datos_multi: features pre-computadas (min, med, full)
        opt_train_users: user_ids de entrenamiento
        opt_val_users: user_ids de validacion
        config_base: dict con n_mix, n_estados, prob_autolazo de Fase 2

    Returns:
        (best_features_name, best_acc, resultados)
    """
    tag = "fase3"

    # Preparar subsets custom
    print("    Preparando subsets personalizados...", flush=True)
    datos_custom = preparar_features_custom(db, user_ids, CUSTOM_SUBSETS)

    # Combinar todos los subsets a probar
    todos_subsets = {}
    for name in FASE3_FEATURES:
        todos_subsets[name] = datos_multi[name]
    for name in CUSTOM_SUBSETS:
        todos_subsets[name] = datos_custom[name]

    subset_names = list(todos_subsets.keys())
    total = len(subset_names)
    print(f"    Subsets a probar: {subset_names}", flush=True)

    # Cargar checkpoint
    ckpt = _cargar_checkpoint(tag)
    resultados = ckpt.get("resultados", []) if ckpt else []
    n_done = len(resultados)
    if n_done > 0:
        print(f"    Checkpoint: {n_done}/{total} ya evaluados", flush=True)

    mejor_acc = max((r[1] for r in resultados), default=0.0)
    mejor_feat = None
    for r in resultados:
        if r[1] == mejor_acc:
            mejor_feat = r[0]

    for i, feat_name in enumerate(subset_names):
        if i < n_done:
            continue

        datos_feat = todos_subsets[feat_name]

        train_seqs, train_labels, _ = reunir_secuencias(
            datos_feat, opt_train_users)
        val_seqs, val_labels, _ = reunir_secuencias(
            datos_feat, opt_val_users)

        norm = NormalizadorZScore()
        train_norm = norm.ajustar_y_transformar(train_seqs)
        val_norm = norm.transformar(val_seqs)

        train_por_digito = defaultdict(list)
        for seq, label in zip(train_norm, train_labels):
            train_por_digito[label].append(seq)

        t0 = time.time()
        acc, _ = entrenar_y_evaluar(
            train_por_digito, val_norm, val_labels,
            n_estados=config_base["n_estados"],
            n_mix=config_base["n_mix"],
            tipo_covarianza="diag",
            prob_autolazo=config_base["prob_autolazo"],
            n_iter=50, n_restarts=5)
        elapsed = time.time() - t0

        n_dims = train_seqs[0].shape[1] if train_seqs else "?"
        marker = " ***" if acc > mejor_acc else ""
        print(f"    [{i+1}/{total}] {feat_name:10s} ({n_dims}D) "
              f"-> {acc*100:.2f}% ({elapsed:.0f}s){marker}", flush=True)

        resultados.append((feat_name, acc))

        if acc > mejor_acc:
            mejor_acc = acc
            mejor_feat = feat_name

        _guardar_checkpoint(tag, {
            "resultados": resultados,
            "config_base": config_base,
            "mejor_feat": mejor_feat,
            "mejor_acc": mejor_acc,
        })

    print(f"\n    Mejor subset: {mejor_feat} -> {mejor_acc * 100:.2f}%")

    return mejor_feat, mejor_acc, resultados


# =============================================================================
# EVALUACION FINAL
# =============================================================================

def evaluar_escenario(datos_por_usuario, train_users, test_users,
                      config, nombre="escenario"):
    """Entrena con config dada y evalua en test."""
    print(f"\n  {'=' * 60}")
    print(f"  Escenario: {nombre}")
    print(f"  Train: {len(train_users)}, Test: {len(test_users)}")
    print(f"  Config: n_mix={config['n_mix']}, n_est={config['n_estados']}, "
          f"cov=diag, p={config['prob_autolazo']}, "
          f"feat={config['features']}")
    print(f"  {'=' * 60}", flush=True)

    t0 = time.time()

    train_seqs, train_labels, _ = reunir_secuencias(
        datos_por_usuario, train_users)
    test_seqs, test_labels, _ = reunir_secuencias(
        datos_por_usuario, test_users)

    norm = NormalizadorZScore()
    train_norm = norm.ajustar_y_transformar(train_seqs)
    test_norm = norm.transformar(test_seqs)

    train_por_digito = defaultdict(list)
    for seq, label in zip(train_norm, train_labels):
        train_por_digito[label].append(seq)

    print("  Entrenando modelos finales...", flush=True)
    modelos = {}
    for digito in range(10):
        seqs = train_por_digito[digito]
        print(f"    Digito {digito}: {len(seqs)} seqs...", end=" ", flush=True)
        modelo, score = entrenar_gmmhmm_digito(
            seqs, n_estados=config["n_estados"], n_mix=config["n_mix"],
            tipo_covarianza="diag", n_iter=100,
            n_restarts=8, prob_autolazo=config["prob_autolazo"],
            semilla_base=digito * 100)
        modelos[digito] = modelo
        print(f"LL={score:.1f}", flush=True)

    predicciones, puntuaciones, accuracy = clasificar_lote(
        modelos, test_norm, test_labels)
    cm = sk_confusion_matrix(test_labels, predicciones, labels=list(range(10)))

    elapsed = time.time() - t0
    print(f"\n  >>> Accuracy: {accuracy * 100:.2f}% ({elapsed:.0f}s)")

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "modelos": modelos,
        "normalizador": norm,
        "predicciones": predicciones,
        "test_labels": test_labels,
        "tiempo": elapsed,
    }


# =============================================================================
# GRAFICAS
# =============================================================================

def graficar_top_configs(resultados, filepath, top_n=20):
    """Grafica las top N configuraciones del grid search (Fase 2)."""
    sorted_res = sorted(resultados, key=lambda x: x[1], reverse=True)[:top_n]

    labels = []
    accs = []
    for cfg, acc in sorted_res:
        label = (f"mix={cfg['n_mix']} est={cfg['n_estados']} "
                 f"p={cfg['prob_autolazo']}")
        labels.append(label)
        accs.append(acc * 100)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(accs)), accs, color="steelblue", edgecolor="navy")
    ax.set_yticks(range(len(accs)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(f"Top {top_n} configuraciones - Fase 2 Grid Search (diag)")
    ax.invert_yaxis()

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"      Guardado: {filepath}")


def graficar_fases(res_fase1, res_fase2, res_fase3, filepath):
    """Resumen visual de las 3 fases."""
    n_subplots = 1 + (1 if res_fase2 else 0) + (1 if res_fase3 else 0)
    fig, axes = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 5))
    if n_subplots == 1:
        axes = [axes]

    idx = 0

    # -- Fase 1: n_mix comparison --
    ax = axes[idx]
    n_mixs = [r["n_mix"] for r in res_fase1]
    accs = [r["accuracy"] * 100 for r in res_fase1]
    colors = ["#2ecc71" if r["n_mix"] == 1 else "#3498db" for r in res_fase1]
    bars = ax.bar(n_mixs, accs, color=colors, edgecolor="navy", width=0.6)
    ax.set_xlabel("n_mix")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Fase 1: Prueba rapida n_mix")
    ax.set_xticks(n_mixs)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{acc:.1f}%", ha="center", fontsize=9)
    idx += 1

    # -- Fase 2: top 10 configs (si se ejecuto) --
    if res_fase2:
        ax = axes[idx]
        sorted_res = sorted(res_fase2, key=lambda x: x[1],
                             reverse=True)[:10]
        labels_f2 = []
        accs_f2 = []
        for cfg, acc in sorted_res:
            label = (f"m={cfg['n_mix']} e={cfg['n_estados']} "
                     f"p={cfg['prob_autolazo']}")
            labels_f2.append(label)
            accs_f2.append(acc * 100)
        bars = ax.barh(range(len(accs_f2)), accs_f2,
                        color="steelblue", edgecolor="navy")
        ax.set_yticks(range(len(accs_f2)))
        ax.set_yticklabels(labels_f2, fontsize=7)
        ax.set_xlabel("Accuracy (%)")
        ax.set_title("Fase 2: Top 10 configs")
        ax.invert_yaxis()
        for bar, acc in zip(bars, accs_f2):
            ax.text(bar.get_width() + 0.1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{acc:.1f}%", va="center", fontsize=7)
        idx += 1

    # -- Fase 3: feature subsets (si se ejecuto) --
    if res_fase3:
        ax = axes[idx]
        feat_names = [r[0] for r in res_fase3]
        accs_f3 = [r[1] * 100 for r in res_fase3]
        bars = ax.bar(range(len(feat_names)), accs_f3,
                       color="#e67e22", edgecolor="navy", width=0.6)
        ax.set_xticks(range(len(feat_names)))
        ax.set_xticklabels(feat_names, fontsize=8, rotation=30)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Fase 3: Seleccion features")
        for bar, acc in zip(bars, accs_f3):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{acc:.1f}%", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"      Guardado: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_inicio = time.time()

    print("=" * 70)
    print("  ENTREGA 2B: GMMHMM - ESCALADO ITERATIVO (SOLO DIAG)")
    print("=" * 70, flush=True)

    # =====================================================================
    # [1] Cargar datos
    # =====================================================================
    print("\n[1] Cargando base de datos...", flush=True)
    db, user_ids = cargar_base_datos(DB_PATH)
    total = sum(1 for _ in iterar_muestras(db, user_ids))
    print(f"  Usuarios: {len(user_ids)}, Muestras: {total}", flush=True)

    # =====================================================================
    # [2] Pre-computar features para todos los subsets estandar
    # =====================================================================
    print("\n[2] Pre-computando features (min, med, full)...", flush=True)
    datos_multi = preparar_features_multisubset(db, user_ids)
    print("  OK", flush=True)

    # Split de optimizacion
    opt_train = user_ids[:N_OPT_TRAIN]
    opt_val = user_ids[N_OPT_TRAIN:N_OPT_TRAIN + N_OPT_VAL]

    # =====================================================================
    # [3] FASE 1: Prueba rapida de GMMHMM
    # =====================================================================
    print("\n" + "=" * 70)
    print("  FASE 1: PRUEBA RAPIDA (n_mix=1,2,3 con config E1)")
    print(f"  Config fija: {FASE1_CONFIG}")
    print("=" * 70, flush=True)

    # Preparar datos med para Fase 1
    datos_med = datos_multi["med"]
    train_seqs, train_labels, _ = reunir_secuencias(datos_med, opt_train)
    val_seqs, val_labels, _ = reunir_secuencias(datos_med, opt_val)

    norm = NormalizadorZScore()
    train_norm = norm.ajustar_y_transformar(train_seqs)
    val_norm = norm.transformar(val_seqs)

    train_por_digito = defaultdict(list)
    for seq, label in zip(train_norm, train_labels):
        train_por_digito[label].append(seq)

    best_nmix_f1, best_acc_f1, mejora_f1, res_fase1 = \
        fase1_prueba_rapida(train_por_digito, val_norm, val_labels)

    # Variables para rastrear resultados de cada fase
    res_fase2 = None
    res_fase3 = None
    fase_final = 1

    if mejora_f1:
        # GMMHMM mejoro -> usar directamente
        config_final = {
            "n_mix": best_nmix_f1,
            "n_estados": FASE1_CONFIG["n_estados"],
            "covarianza": "diag",
            "prob_autolazo": FASE1_CONFIG["prob_autolazo"],
            "features": FASE1_CONFIG["features"],
        }
        print(f"\n  >>> FASE 1 exitosa. Config final: n_mix={best_nmix_f1}")
        print("  >>> Saltando Fases 2 y 3.")
    else:
        # =================================================================
        # [4] FASE 2: Grid search (n_mix, n_estados, prob_autolazo)
        # =================================================================
        print("\n\n" + "=" * 70)
        print("  FASE 2: GRID SEARCH (n_mix x n_estados x prob_autolazo)")
        print(f"  Features fijas: {FASE1_CONFIG['features']}")
        print(f"  Covarianza: diag")
        print(f"  Referencia Fase 1: {best_acc_f1 * 100:.2f}%")
        print("=" * 70, flush=True)

        mejor_cfg_f2, mejor_acc_f2, mejora_f2, res_fase2 = \
            fase2_busqueda_hiperparametros(
                datos_med, opt_train, opt_val, best_acc_f1)

        graficar_top_configs(
            res_fase2,
            os.path.join(RESULTS_DIR, "top_configs_fase2.png"))

        fase_final = 2

        if mejora_f2:
            config_final = {
                "n_mix": mejor_cfg_f2["n_mix"],
                "n_estados": mejor_cfg_f2["n_estados"],
                "covarianza": "diag",
                "prob_autolazo": mejor_cfg_f2["prob_autolazo"],
                "features": FASE1_CONFIG["features"],
            }
            print(f"\n  >>> FASE 2 exitosa. Saltando Fase 3.")
        else:
            # =============================================================
            # [5] FASE 3: Seleccion de features (ultimo recurso)
            # =============================================================
            print("\n\n" + "=" * 70)
            print("  FASE 3: SELECCION DE FEATURES (ULTIMO RECURSO)")
            print(f"  Config base: n_mix={mejor_cfg_f2['n_mix']}, "
                  f"n_est={mejor_cfg_f2['n_estados']}, "
                  f"p={mejor_cfg_f2['prob_autolazo']}")
            print("=" * 70, flush=True)

            mejor_feat_f3, mejor_acc_f3, res_fase3 = \
                fase3_seleccion_features(
                    db, user_ids, datos_multi, opt_train, opt_val,
                    mejor_cfg_f2)

            config_final = {
                "n_mix": mejor_cfg_f2["n_mix"],
                "n_estados": mejor_cfg_f2["n_estados"],
                "covarianza": "diag",
                "prob_autolazo": mejor_cfg_f2["prob_autolazo"],
                "features": mejor_feat_f3,
            }
            fase_final = 3

    # =====================================================================
    # [6] EVALUACION FINAL
    # =====================================================================
    print("\n\n" + "=" * 70)
    print("  EVALUACION FINAL")
    print(f"  Config: {config_final}")
    print(f"  Decidida en Fase {fase_final}")
    print("=" * 70, flush=True)

    # Obtener datos del subset de features correcto
    feat_name = config_final["features"]
    if feat_name in datos_multi:
        datos_eval = datos_multi[feat_name]
    else:
        # Subset personalizado
        datos_custom = preparar_features_custom(
            db, user_ids, {feat_name: CUSTOM_SUBSETS[feat_name]})
        datos_eval = datos_custom[feat_name]

    train_74 = user_ids[:N_TRAIN_74]
    test_74 = user_ids[N_TRAIN_74:]
    train_47 = user_ids[:N_TRAIN_47]
    test_47 = user_ids[N_TRAIN_47:]

    # -- N=74 --
    print(f"\n  [6a] ESCENARIO N=74")
    res_74 = evaluar_escenario(datos_eval, train_74, test_74,
                                config_final, nombre="N=74 (mejor config)")

    graficar_confusion(
        res_74["confusion_matrix"],
        f"Confusion N=74 GMMHMM ({res_74['accuracy']*100:.1f}%)",
        os.path.join(RESULTS_DIR, "confusion_N74_v3.png"))

    # -- N=47 --
    print(f"\n  [6b] ESCENARIO N=47")
    res_47 = evaluar_escenario(datos_eval, train_47, test_47,
                                config_final, nombre="N=47 (mejor config)")

    graficar_confusion(
        res_47["confusion_matrix"],
        f"Confusion N=47 GMMHMM ({res_47['accuracy']*100:.1f}%)",
        os.path.join(RESULTS_DIR, "confusion_N47_v3.png"))

    # =====================================================================
    # [7] GRAFICAS RESUMEN
    # =====================================================================
    print("\n  Generando graficas resumen...", flush=True)
    graficar_fases(res_fase1, res_fase2, res_fase3,
                    os.path.join(RESULTS_DIR, "comparativa_fases.png"))

    # =====================================================================
    # [8] GUARDAR RESULTADOS
    # =====================================================================
    resumen = {
        "config_final": config_final,
        "fase_final": fase_final,
        "fase1": {
            "resultados": res_fase1,
            "mejor_n_mix": best_nmix_f1,
            "mejor_acc": best_acc_f1,
            "mejora": mejora_f1,
        },
        "fase2": {
            "total_configs": len(res_fase2) if res_fase2 else 0,
            "top_10": sorted(res_fase2, key=lambda x: x[1],
                             reverse=True)[:10] if res_fase2 else [],
        } if res_fase2 is not None else None,
        "fase3": {
            "resultados": res_fase3,
        } if res_fase3 is not None else None,
        "evaluacion_final": {
            "N74": {"accuracy": res_74["accuracy"]},
            "N47": {"accuracy": res_47["accuracy"]},
        },
        "comparativa": {
            "E1_N74": RESULTADOS_ANTERIORES["E1_N74"],
            "E1_N47": RESULTADOS_ANTERIORES["E1_N47"],
            "E2_N74": RESULTADOS_ANTERIORES["E2_N74"],
            "E2_N47": RESULTADOS_ANTERIORES["E2_N47"],
            "E3_N74": res_74["accuracy"],
            "E3_N47": res_47["accuracy"],
        },
    }
    with open(os.path.join(RESULTS_DIR, "resultados_v3.json"), "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"  Resultados guardados en resultados_v3.json")

    # =====================================================================
    # [9] RESUMEN FINAL
    # =====================================================================
    elapsed = time.time() - t_inicio

    print("\n" + "=" * 70)
    print("  RESUMEN ENTREGA 2B - ESCALADO ITERATIVO")
    print("=" * 70)

    print(f"\n  Fase 1 (prueba rapida n_mix con config E1):")
    for r in res_fase1:
        marker = " <-- mejor" if r["n_mix"] == best_nmix_f1 else ""
        print(f"    n_mix={r['n_mix']}: {r['accuracy'] * 100:.2f}%{marker}")
    print(f"    Mejora? {'SI' if mejora_f1 else 'NO'}")

    if res_fase2 is not None:
        top3 = sorted(res_fase2, key=lambda x: x[1], reverse=True)[:3]
        print(f"\n  Fase 2 (grid search, {len(res_fase2)} configs):")
        for cfg, acc in top3:
            print(f"    n_mix={cfg['n_mix']}, n_est={cfg['n_estados']}, "
                  f"p={cfg['prob_autolazo']} -> {acc*100:.2f}%")

    if res_fase3 is not None:
        print(f"\n  Fase 3 (seleccion features):")
        for feat_name, acc in res_fase3:
            marker = " <-- mejor" if feat_name == config_final["features"] \
                else ""
            print(f"    {feat_name:10s}: {acc * 100:.2f}%{marker}")

    print(f"\n  Config final (decidida en Fase {fase_final}):")
    for k, v in config_final.items():
        print(f"    {k}: {v}")

    print(f"\n  Comparativa final:")
    print(f"  {'Escenario':<15} {'E1':>10} {'E2':>10} {'E3':>10}")
    print("  " + "-" * 48)
    print(f"  {'N=74':<15} "
          f"{RESULTADOS_ANTERIORES['E1_N74']*100:>9.2f}% "
          f"{RESULTADOS_ANTERIORES['E2_N74']*100:>9.2f}% "
          f"{res_74['accuracy']*100:>9.2f}%")
    print(f"  {'N=47':<15} "
          f"{RESULTADOS_ANTERIORES['E1_N47']*100:>9.2f}% "
          f"{RESULTADOS_ANTERIORES['E2_N47']*100:>9.2f}% "
          f"{res_47['accuracy']*100:>9.2f}%")

    print(f"\n  Tiempo total: {elapsed / 60:.1f} minutos")
    print("=" * 70)


if __name__ == "__main__":
    main()
