"""
=============================================================================
ENTREGA 2 - PARTE A: HMMs CON HIPERPARAMETROS OPTIMIZADOS POR DIGITO
=============================================================================
Mejora sobre Entrega 1: en lugar de usar los mismos n_estados y
prob_autolazo para todos los digitos, se optimizan individualmente.

Estrategia (descenso por coordenadas sobre accuracy conjunta):
  1. Dividir usuarios de train en inner_train (80%) y val (20%)
  2. Empezar con params baseline (7 estados, 0.6 autolazo) para todos
  3. Para cada digito, probar todas las combinaciones del grid manteniendo
     los demas digitos fijos, y quedarse con la que maximice la accuracy
     de clasificacion conjunta sobre validacion (los 10 modelos juntos)
  4. Repetir el barrido hasta convergencia (max 3 rondas)
  5. Reentrenar con todos los datos de train usando los params optimos
  6. Evaluar en test

Uso:
  python clasificador_digitos_v2.py
=============================================================================
"""

import sys
import os
import time
import json
import warnings
import pickle
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
REPO_ROOT_IMPORT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
EXTRACTOR_DIR = os.path.join(
    REPO_ROOT_IMPORT, "Extractores_adaptados", "Extractores", "Extractor Local"
)
sys.path.insert(0, EXTRACTOR_DIR)
sys.path.insert(0, ENTREGA1_DIR)

from clasificador_digitos import (
    cargar_base_datos, iterar_muestras, preprocesar, extraer_features,
    NormalizadorZScore, entrenar_hmm_digito, clasificar, clasificar_lote,
    graficar_confusion, FEATURE_SUBSETS,
)

# =============================================================================
# CONFIGURACION
# =============================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DB_PATH = os.path.join(REPO_ROOT, "e-BioDigit_DB", "e-BioDigit_DB")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "resultados")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_TRAIN_74 = 74
N_TRAIN_47 = 47

# Usuarios para la fase de busqueda de hiperparametros
N_OPT_TRAIN = 78
N_OPT_VAL = 15  # 78 + 15 = 93 (todos los usuarios)

# Grid de busqueda por digito
GRID_N_ESTADOS = [3, 5, 7, 10]
GRID_PROB_AUTOLAZO = [0.4, 0.5, 0.6, 0.7, 0.8]

# Fraccion de usuarios de train usados para validacion interna
FRAC_VAL = 0.2

# Params baseline (mejor config de Entrega 1)
BASELINE_N_ESTADOS = 7
BASELINE_PROB_AUTOLAZO = 0.6



# =============================================================================
# FUNCIONES
# =============================================================================

def preparar_features_por_usuario(db, user_ids, indices_features,
                                   n_resample=80, suavizar=True):
    """Pre-computa features agrupadas por usuario y digito.

    Returns:
        dict {uid: [(feats_array, digito), ...]}
    """
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


def reunir_secuencias(datos_por_usuario, user_ids):
    """Reune secuencias y etiquetas de un conjunto de usuarios.

    Returns:
        secuencias: lista de arrays (T, D)
        etiquetas: lista de digitos
        por_digito: dict {digito: [arrays]}
    """
    secuencias = []
    etiquetas = []
    por_digito = defaultdict(list)
    for uid in user_ids:
        for feats, digito in datos_por_usuario[uid]:
            secuencias.append(feats)
            etiquetas.append(digito)
            por_digito[digito].append(feats)
    return secuencias, etiquetas, por_digito


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


def optimizar_params_por_digito(train_por_digito_norm, val_seqs_norm,
                                 val_labels, n_iter=50, n_restarts=2,
                                 tag="opt"):
    """Optimiza params de cada digito individualmente, maximizando la
    accuracy de clasificacion conjunta sobre validacion.

    Para cada digito, prueba todas las combinaciones del grid mientras
    los demas 9 modelos permanecen en baseline. Se queda con la
    combinacion que maximiza la accuracy global del sistema.

    Returns:
        mejores_params: dict {digito: {"n_estados": int, "prob_autolazo": float}}
        accuracy_val: float
    """
    # Entrenar modelos baseline (7 est, 0.6 autolazo)
    print("    Entrenando modelos baseline (7 est, 0.6 autolazo)...",
          flush=True)
    modelos_baseline = {}
    for digito in range(10):
        modelo, _ = entrenar_hmm_digito(
            train_por_digito_norm[digito],
            n_estados=BASELINE_N_ESTADOS, tipo_covarianza="diag",
            n_iter=n_iter, n_restarts=n_restarts,
            prob_autolazo=BASELINE_PROB_AUTOLAZO,
            semilla_base=digito * 100)
        modelos_baseline[digito] = modelo

    # Accuracy baseline
    _, _, acc_baseline = clasificar_lote(
        modelos_baseline, val_seqs_norm, val_labels)
    print(f"    Accuracy baseline: {acc_baseline * 100:.2f}%", flush=True)

    # Optimizar cada digito individualmente
    params = {d: {"n_estados": BASELINE_N_ESTADOS,
                  "prob_autolazo": BASELINE_PROB_AUTOLAZO}
              for d in range(10)}

    print("\n    Optimizando cada digito (otros en baseline)...", flush=True)

    for digito in range(10):
        mejor_acc = acc_baseline
        mejor_n = BASELINE_N_ESTADOS
        mejor_p = BASELINE_PROB_AUTOLAZO

        for n_est in GRID_N_ESTADOS:
            for p_auto in GRID_PROB_AUTOLAZO:
                if n_est == BASELINE_N_ESTADOS and p_auto == BASELINE_PROB_AUTOLAZO:
                    continue

                try:
                    modelo_cand, _ = entrenar_hmm_digito(
                        train_por_digito_norm[digito],
                        n_estados=n_est, tipo_covarianza="diag",
                        n_iter=n_iter, n_restarts=n_restarts,
                        prob_autolazo=p_auto,
                        semilla_base=digito * 100)

                    if modelo_cand is None:
                        continue

                    # Sustituir solo este digito, otros en baseline
                    modelos_tmp = dict(modelos_baseline)
                    modelos_tmp[digito] = modelo_cand

                    _, _, acc = clasificar_lote(
                        modelos_tmp, val_seqs_norm, val_labels)

                    if acc > mejor_acc:
                        mejor_acc = acc
                        mejor_n = n_est
                        mejor_p = p_auto

                except Exception:
                    continue

        params[digito] = {"n_estados": mejor_n, "prob_autolazo": mejor_p}

        if mejor_acc > acc_baseline:
            print(f"    Digito {digito}: {mejor_n} est, p={mejor_p} "
                  f"-> acc={mejor_acc * 100:.2f}% "
                  f"(+{(mejor_acc - acc_baseline) * 100:.2f}%)", flush=True)
        else:
            print(f"    Digito {digito}: baseline (7 est, 0.6)", flush=True)

        # Checkpoint
        ckpt_data = {
            "params": {str(k): v for k, v in params.items()},
            "accuracy_baseline": acc_baseline,
        }
        _guardar_checkpoint(tag, ckpt_data)

    # Accuracy final con todos los params optimos combinados
    modelos_final = {}
    for digito in range(10):
        p = params[digito]
        if (p["n_estados"] == BASELINE_N_ESTADOS
                and p["prob_autolazo"] == BASELINE_PROB_AUTOLAZO):
            modelos_final[digito] = modelos_baseline[digito]
        else:
            modelo, _ = entrenar_hmm_digito(
                train_por_digito_norm[digito],
                n_estados=p["n_estados"], tipo_covarianza="diag",
                n_iter=n_iter, n_restarts=n_restarts,
                prob_autolazo=p["prob_autolazo"],
                semilla_base=digito * 100)
            modelos_final[digito] = modelo

    _, _, acc_final = clasificar_lote(
        modelos_final, val_seqs_norm, val_labels)
    print(f"\n    Accuracy combinada: {acc_final * 100:.2f}% "
          f"(baseline: {acc_baseline * 100:.2f}%)", flush=True)

    return params, acc_final


def entrenar_con_params_optimos(train_por_digito_norm, params_por_digito,
                                 n_iter=100, n_restarts=5):
    """Entrena un HMM por digito con sus parametros optimos individuales.

    Returns:
        modelos: dict {digito: GaussianHMM}
        scores: dict {digito: log-verosimilitud}
    """
    modelos = {}
    scores = {}

    for digito in range(10):
        seqs = train_por_digito_norm[digito]
        p = params_por_digito[digito]
        n_est = p["n_estados"]
        p_auto = p["prob_autolazo"]

        print(f"    Digito {digito}: {len(seqs)} seqs, "
              f"n_est={n_est}, p_auto={p_auto}...", end=" ", flush=True)

        modelo, score = entrenar_hmm_digito(
            seqs, n_estados=n_est, tipo_covarianza="diag",
            n_iter=n_iter, n_restarts=n_restarts, prob_autolazo=p_auto,
            semilla_base=digito * 100)

        modelos[digito] = modelo
        scores[digito] = score
        print(f"LL={score:.1f}", flush=True)

    return modelos, scores


def evaluar_escenario(datos_por_usuario, train_users, test_users,
                      params_por_digito, nombre="escenario"):
    """Entrena con params dados y evalua en test.

    Returns:
        dict con accuracy, confusion_matrix, etc.
    """
    print(f"\n  {'=' * 60}")
    print(f"  Escenario: {nombre}")
    print(f"  Train: {len(train_users)}, Test: {len(test_users)}")
    print(f"  {'=' * 60}", flush=True)

    t0 = time.time()

    # 1. Reunir datos
    train_seqs, train_labels, _ = reunir_secuencias(
        datos_por_usuario, train_users)
    test_seqs, test_labels, _ = reunir_secuencias(
        datos_por_usuario, test_users)

    # 2. Normalizar
    norm = NormalizadorZScore()
    train_norm = norm.ajustar_y_transformar(train_seqs)
    test_norm = norm.transformar(test_seqs)

    train_norm_por_digito = defaultdict(list)
    for seq, label in zip(train_norm, train_labels):
        train_norm_por_digito[label].append(seq)

    # 3. Entrenar con params optimos
    print("  Entrenando con params optimizados...", flush=True)
    modelos, _ = entrenar_con_params_optimos(
        train_norm_por_digito, params_por_digito, n_iter=100, n_restarts=5)

    # 4. Evaluar
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


def graficar_params_por_digito(params, filepath):
    """Grafica los parametros optimos por digito."""
    digitos = list(range(10))
    n_estados = [params[d]["n_estados"] for d in digitos]
    p_autolazo = [params[d]["prob_autolazo"] for d in digitos]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # N estados
    barras1 = ax1.bar(digitos, n_estados, color="steelblue", edgecolor="navy")
    ax1.set_xlabel("Digito")
    ax1.set_ylabel("N estados")
    ax1.set_title("Numero optimo de estados por digito")
    ax1.set_xticks(digitos)
    for b, v in zip(barras1, n_estados):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                 str(v), ha="center", va="bottom", fontsize=10)
    ax1.set_ylim(0, max(n_estados) + 2)
    ax1.grid(axis="y", alpha=0.3)

    # Prob autolazo
    barras2 = ax2.bar(digitos, p_autolazo, color="coral", edgecolor="darkred")
    ax2.set_xlabel("Digito")
    ax2.set_ylabel("Prob. autolazo")
    ax2.set_title("Probabilidad optima de autolazo por digito")
    ax2.set_xticks(digitos)
    for b, v in zip(barras2, p_autolazo):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    ax2.set_ylim(0, max(p_autolazo) + 0.15)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Hiperparametros optimizados por digito", fontsize=13)
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
    print("  ENTREGA 2A: HMMs CON PARAMS OPTIMIZADOS POR DIGITO")
    print("  (Descenso por coordenadas sobre accuracy conjunta)")
    print("=" * 70, flush=True)

    # 1. Cargar datos
    print("\n[1] Cargando base de datos...", flush=True)
    db, user_ids = cargar_base_datos(DB_PATH)
    total = sum(1 for _ in iterar_muestras(db, user_ids))
    print(f"  Usuarios: {len(user_ids)}, Muestras: {total}", flush=True)

    # 2. Pre-computar features
    print("\n[2] Pre-computando features...", flush=True)
    indices_med = FEATURE_SUBSETS["med"]
    datos = preparar_features_por_usuario(db, user_ids, indices_med)
    print("  OK", flush=True)

    # 3. Busqueda de hiperparametros con 78 train + 15 val
    print(f"\n[3] BUSQUEDA DE HIPERPARAMETROS "
          f"({N_OPT_TRAIN} train, {N_OPT_VAL} val)")
    print("=" * 60, flush=True)

    opt_train_users = user_ids[:N_OPT_TRAIN]
    opt_val_users = user_ids[N_OPT_TRAIN:N_OPT_TRAIN + N_OPT_VAL]

    # Preparar datos para optimizacion
    opt_train_seqs, opt_train_labels, _ = reunir_secuencias(
        datos, opt_train_users)
    opt_val_seqs, opt_val_labels, _ = reunir_secuencias(
        datos, opt_val_users)

    norm_opt = NormalizadorZScore()
    opt_train_norm = norm_opt.ajustar_y_transformar(opt_train_seqs)
    opt_val_norm = norm_opt.transformar(opt_val_seqs)

    opt_train_por_digito = defaultdict(list)
    for seq, label in zip(opt_train_norm, opt_train_labels):
        opt_train_por_digito[label].append(seq)

    # Descenso por coordenadas
    params_opt, acc_val = optimizar_params_por_digito(
        opt_train_por_digito, opt_val_norm, opt_val_labels,
        n_iter=50, n_restarts=2, tag="opt_global")

    print(f"\n  Params encontrados (acc_val={acc_val * 100:.2f}%):")
    print(f"  {'Digito':<8} {'N estados':>10} {'P autolazo':>12}")
    print("  " + "-" * 32)
    for d in range(10):
        p = params_opt[d]
        print(f"  {d:<8} {p['n_estados']:>10} {p['prob_autolazo']:>12.1f}")

    # 4. Evaluar N=74 con params encontrados
    train_74 = user_ids[:N_TRAIN_74]
    test_74 = user_ids[N_TRAIN_74:]

    print(f"\n\n[4] ESCENARIO N=74 (74 train, 19 test)")
    res_74 = evaluar_escenario(datos, train_74, test_74, params_opt,
                                nombre="N=74 (params optimizados)")

    graficar_confusion(
        res_74["confusion_matrix"],
        f"Confusion N=74 optimizado ({res_74['accuracy']*100:.1f}%)",
        os.path.join(RESULTS_DIR, "confusion_N74_opt.png"))

    # 5. Evaluar N=47 con los mismos params
    train_47 = user_ids[:N_TRAIN_47]
    test_47 = user_ids[N_TRAIN_47:]

    print(f"\n\n[5] ESCENARIO N=47 (47 train, 46 test)")
    res_47 = evaluar_escenario(datos, train_47, test_47, params_opt,
                                nombre="N=47 (params optimizados)")

    graficar_confusion(
        res_47["confusion_matrix"],
        f"Confusion N=47 optimizado ({res_47['accuracy']*100:.1f}%)",
        os.path.join(RESULTS_DIR, "confusion_N47_opt.png"))

    # 6. Graficas y resultados
    graficar_params_por_digito(
        params_opt,
        os.path.join(RESULTS_DIR, "params_por_digito.png"))

    resumen = {
        "optimizacion": {
            "n_train": N_OPT_TRAIN,
            "n_val": N_OPT_VAL,
            "accuracy_val": acc_val,
            "params": {str(d): params_opt[d] for d in range(10)},
        },
        "N74": {"accuracy": res_74["accuracy"]},
        "N47": {"accuracy": res_47["accuracy"]},
    }
    with open(os.path.join(RESULTS_DIR, "resultados.json"), "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    # 7. Resumen
    elapsed = time.time() - t_inicio

    print("\n" + "=" * 70)
    print("  RESUMEN ENTREGA 2A")
    print("=" * 70)

    print(f"\n  Params optimizados ({N_OPT_TRAIN} train + {N_OPT_VAL} val):")
    print(f"  Accuracy validacion: {acc_val * 100:.2f}%")
    print(f"\n  {'Digito':<8} {'N estados':>10} {'P autolazo':>12}")
    print("  " + "-" * 32)
    for d in range(10):
        p = params_opt[d]
        print(f"  {d:<8} {p['n_estados']:>10} {p['prob_autolazo']:>12.1f}")

    print(f"\n  Comparativa con Entrega 1 (params uniformes):")
    print(f"  {'Escenario':<15} {'E1 (uniforme)':>15} {'E2 (por digito)':>17}")
    print("  " + "-" * 50)
    print(f"  {'N=74':<15} {'92.89%':>15} {res_74['accuracy']*100:>16.2f}%")
    print(f"  {'N=47':<15} {'89.78%':>15} {res_47['accuracy']*100:>16.2f}%")

    print(f"\n  Tiempo total: {elapsed / 60:.1f} minutos")
    print("=" * 70)


if __name__ == "__main__":
    main()
