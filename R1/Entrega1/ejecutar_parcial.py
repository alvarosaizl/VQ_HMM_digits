"""
Ejecucion parcial: solo los experimentos nuevos.
  1. Tabla de preprocesado completa (con centrado/escalado on/off)
  2. Los 3 escenarios finales: N=74, N=47, LOO CV
"""
import time
import os
import json
import numpy as np
from collections import defaultdict
from sklearn.model_selection import LeaveOneOut

from clasificador_digitos import (
    cargar_base_datos, iterar_muestras, preparar_datos, NormalizadorZScore,
    FEATURE_SUBSETS, RESULTS_DIR, DB_PATH, N_TRAIN_74, N_TRAIN_47,
    graficar_comparativa, graficar_confusion, graficar_comparativa_global,
    _entrenar_y_evaluar,
)

def main():
    t_inicio = time.time()

    # ------------------------------------------------------------------
    # PASO 1: Cargar base de datos
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  EJECUCION PARCIAL: PREPROCESADO + 3 ESCENARIOS")
    print("=" * 70, flush=True)

    print("\n[PASO 1] Cargando base de datos...", flush=True)
    db, user_ids = cargar_base_datos(DB_PATH)
    total = sum(1 for _ in iterar_muestras(db, user_ids))
    print(f"  Usuarios: {len(user_ids)}, Muestras: {total}", flush=True)

    train_74 = user_ids[:N_TRAIN_74]
    test_74 = user_ids[N_TRAIN_74:]
    train_47 = user_ids[:N_TRAIN_47]
    test_47 = user_ids[N_TRAIN_47:]

    todos_resultados = {}
    resultados_busqueda = []
    datos_cache = {}

    def _preparar_cache(subset, suavizar, n_resample, train_users, test_users,
                        etiqueta=None, centrar=True, escalar=True):
        clave = (subset, suavizar, n_resample, len(train_users), centrar, escalar)
        if clave in datos_cache:
            return datos_cache[clave]

        indices = FEATURE_SUBSETS[subset]
        if etiqueta:
            print(f"    Preparando: {etiqueta}...", end=" ", flush=True)

        _, train_seqs, train_labels = preparar_datos(
            db, train_users, indices, n_resample, suavizar, centrar, escalar)
        _, test_seqs, test_labels = preparar_datos(
            db, test_users, indices, n_resample, suavizar, centrar, escalar)

        normalizador = NormalizadorZScore()
        train_norm = normalizador.ajustar_y_transformar(train_seqs)
        test_norm = normalizador.transformar(test_seqs)

        train_norm_por_digito = defaultdict(list)
        for seq, label in zip(train_norm, train_labels):
            train_norm_por_digito[label].append(seq)

        result = (train_norm_por_digito, test_norm, test_labels, normalizador)
        datos_cache[clave] = result
        if etiqueta:
            print("OK", flush=True)
        return result

    # ------------------------------------------------------------------
    # PASO 2: Tabla de preprocesado completa
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [PASO 2] EFECTOS DEL PREPROCESADO (tabla completa)")
    print("=" * 70, flush=True)

    # (nombre, suavizar, n_resample, centrar, escalar)
    preproc_configs = [
        ("con_todo",       True,  80, True,  True),
        ("sin_suavizado",  False, 80, True,  True),
        ("sin_resample",   True,  0,  True,  True),
        ("sin_centrado",   True,  80, False, True),
        ("sin_escalado",   True,  80, True,  False),
        ("sin_cent_esc",   True,  80, False, False),
        ("sin_nada",       False, 0,  False, False),
    ]
    res_preproc = []
    for nombre, suav, nres, cent, esc in preproc_configs:
        t0 = time.time()
        tr, te, tl, _ = _preparar_cache(
            "med", suav, nres, train_74, test_74, nombre, cent, esc)
        r = _entrenar_y_evaluar(tr, te, tl, n_restarts=2, n_iter=50)
        dt = time.time() - t0
        res_preproc.append({"valor": nombre, "accuracy": r["accuracy"]})
        todos_resultados[f"preproc={nombre}"] = r["accuracy"]
        resultados_busqueda.append({"param": "preprocesado", "valor": nombre,
                                    "accuracy": r["accuracy"]})
        print(f"    preprocesado={nombre:>16s}: accuracy={r['accuracy']*100:5.1f}%  "
              f"({dt:.0f}s)", flush=True)

    graficar_comparativa(res_preproc, "Preprocesado",
                         os.path.join(RESULTS_DIR, "comp_preprocesado.png"))

    # ------------------------------------------------------------------
    # PASO 3: 3 escenarios con mejor configuracion
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [PASO 3] EVALUACION CON MEJOR CONFIGURACION")
    print("=" * 70, flush=True)

    # --- Escenario N=74 ---
    print("\n  >>> Escenario N=74 (74 train, 19 test) <<<", flush=True)
    med_train, med_test, med_labels, med_norm = _preparar_cache(
        "med", True, 80, train_74, test_74, "med(12D) N=74")
    res_74 = _entrenar_y_evaluar(med_train, med_test, med_labels,
                                 n_estados=7, tipo_covarianza="diag",
                                 n_iter=100, n_restarts=5, prob_autolazo=0.6)
    todos_resultados["MEJOR_N74"] = res_74["accuracy"]
    print(f"    Accuracy: {res_74['accuracy']*100:.2f}%", flush=True)

    graficar_confusion(
        res_74["confusion_matrix"], "Matriz de Confusion - N=74 (mejor config)",
        os.path.join(RESULTS_DIR, "confusion_N74.png"))

    # --- Escenario N=47 ---
    print("\n  >>> Escenario N=47 (47 train, 46 test) <<<", flush=True)
    tr47, te47, tl47, _ = _preparar_cache(
        "med", True, 80, train_47, test_47, "med(12D) N=47")
    res_47 = _entrenar_y_evaluar(tr47, te47, tl47,
                                 n_estados=7, tipo_covarianza="diag",
                                 n_iter=100, n_restarts=5, prob_autolazo=0.6)
    todos_resultados["MEJOR_N47"] = res_47["accuracy"]
    print(f"    Accuracy: {res_47['accuracy']*100:.2f}%", flush=True)

    graficar_confusion(
        res_47["confusion_matrix"], "Matriz de Confusion - N=47 (mejor config)",
        os.path.join(RESULTS_DIR, "confusion_N47.png"))

    # --- Escenario ALL: Leave-One-User-Out ---
    print("\n  >>> Escenario ALL (Leave-One-User-Out CV) <<<", flush=True)
    loo = LeaveOneOut()
    n_users = len(user_ids)
    accuracies_loo = []
    cm_acumulada = np.zeros((10, 10), dtype=int)
    indices_med = FEATURE_SUBSETS["med"]

    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(user_ids)):
        train_users_fold = [user_ids[i] for i in train_idx]
        test_users_fold = [user_ids[i] for i in test_idx]

        _, train_seqs, train_labels = preparar_datos(
            db, train_users_fold, indices_med, n_resample=80, suavizar=True)
        _, test_seqs, test_labels = preparar_datos(
            db, test_users_fold, indices_med, n_resample=80, suavizar=True)

        normalizador_fold = NormalizadorZScore()
        train_norm = normalizador_fold.ajustar_y_transformar(train_seqs)
        test_norm = normalizador_fold.transformar(test_seqs)

        train_norm_por_digito = defaultdict(list)
        for seq, label in zip(train_norm, train_labels):
            train_norm_por_digito[label].append(seq)

        r_fold = _entrenar_y_evaluar(train_norm_por_digito, test_norm,
                                     test_labels, n_estados=7,
                                     tipo_covarianza="diag", n_iter=100,
                                     n_restarts=5, prob_autolazo=0.6)
        accuracies_loo.append(r_fold["accuracy"])
        cm_acumulada += r_fold["confusion_matrix"]
        print(f"    Usuario {fold_idx+1}/{n_users} ({test_users_fold[0]}): "
              f"accuracy={r_fold['accuracy']*100:.1f}%", flush=True)

    acc_media_loo = np.mean(accuracies_loo)
    acc_std_loo = np.std(accuracies_loo)
    todos_resultados["MEJOR_LOO_ALL"] = acc_media_loo
    print(f"    Media: {acc_media_loo*100:.2f}% (+/- {acc_std_loo*100:.2f}%)",
          flush=True)

    graficar_confusion(
        cm_acumulada,
        f"Matriz de Confusion - LOO CV ({acc_media_loo*100:.1f}% media)",
        os.path.join(RESULTS_DIR, "confusion_LOO.png"))

    # ------------------------------------------------------------------
    # RESUMEN
    # ------------------------------------------------------------------
    elapsed = time.time() - t_inicio

    print("\n" + "=" * 70)
    print("  RESUMEN")
    print("=" * 70)

    print(f"\n  --- Tabla de preprocesado ---")
    print(f"  {'Config':<18} {'Suav':>5} {'Res':>5} {'Cent':>5} {'Esc':>5} {'Accuracy':>10}")
    print("  " + "-" * 52)
    for (nombre, suav, nres, cent, esc), res in zip(preproc_configs, res_preproc):
        s = "si" if suav else "no"
        r = str(nres) if nres > 0 else "no"
        c = "si" if cent else "no"
        e = "si" if esc else "no"
        print(f"  {nombre:<18} {s:>5} {r:>5} {c:>5} {e:>5} {res['accuracy']*100:>9.1f}%")

    print(f"\n  --- 3 Escenarios (mejor config) ---")
    print(f"  {'Escenario':<25} {'Accuracy':>10}")
    print("  " + "-" * 38)
    print(f"  {'N=74 (74 train/19 test)':<25} {res_74['accuracy'] * 100:>9.2f}%")
    print(f"  {'N=47 (47 train/46 test)':<25} {res_47['accuracy'] * 100:>9.2f}%")
    print(f"  {'ALL (LOO CV)':<25} {acc_media_loo * 100:>8.2f}% +/- {acc_std_loo * 100:.2f}%")

    print(f"\n  Tiempo total: {elapsed / 60:.1f} minutos")
    print("=" * 70)


if __name__ == "__main__":
    main()
