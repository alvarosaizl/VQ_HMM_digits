"""
LOO Cross Validation optimizado con checkpoint.
Pre-computa features una sola vez y guarda progreso en disco
para poder reanudar si se interrumpe.
"""
import time
import os
import json
import numpy as np
from collections import defaultdict

from clasificador_digitos import (
    cargar_base_datos, iterar_muestras, preprocesar, extraer_features,
    NormalizadorZScore, FEATURE_SUBSETS, RESULTS_DIR, DB_PATH,
    graficar_confusion, entrenar_todos_los_digitos, clasificar_lote,
)
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "loo_checkpoint.json")


def cargar_checkpoint():
    """Carga el checkpoint si existe."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
        data["cm_acumulada"] = np.array(data["cm_acumulada"])
        return data
    return None


def guardar_checkpoint(resultados_por_uid, cm_acumulada):
    """Guarda el progreso actual a disco."""
    data = {
        "resultados_por_uid": resultados_por_uid,
        "cm_acumulada": cm_acumulada.tolist(),
    }
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f)


def main():
    t_inicio = time.time()

    print("=" * 70)
    print("  LEAVE-ONE-USER-OUT CV (5 restarts, 100 iter, con checkpoint)")
    print("=" * 70, flush=True)

    # Cargar checkpoint si existe
    ckpt = cargar_checkpoint()
    if ckpt:
        resultados_por_uid = ckpt["resultados_por_uid"]
        cm_acumulada = ckpt["cm_acumulada"]
        print(f"\n  Checkpoint encontrado: {len(resultados_por_uid)}/93 usuarios "
              f"completados. Reanudando...", flush=True)
    else:
        resultados_por_uid = {}
        cm_acumulada = np.zeros((10, 10), dtype=int)
        print("\n  Sin checkpoint previo. Empezando desde cero.", flush=True)

    print("\nCargando base de datos...", flush=True)
    db, user_ids = cargar_base_datos(DB_PATH)
    n_users = len(user_ids)
    print(f"  Usuarios: {n_users}", flush=True)

    indices_med = FEATURE_SUBSETS["med"]

    # Pre-computar features para TODOS los usuarios una sola vez
    print("  Pre-computando features para todos los usuarios...", flush=True)
    datos_por_usuario = {}
    for uid in user_ids:
        muestras_usuario = []
        for _, digito, sesion, muestra in iterar_muestras(db, [uid]):
            x, y, ts = preprocesar(
                muestra["x"], muestra["y"], muestra["timestamp"],
                n_resample=80, suavizar=True)
            feats = extraer_features(x, y, muestra["presion"],
                                     indices_features=indices_med)
            muestras_usuario.append((feats, digito))
        datos_por_usuario[uid] = muestras_usuario
    print("  Features pre-computadas.", flush=True)

    # LOO: dejar 1 usuario fuera cada vez
    pendientes = [uid for uid in user_ids if uid not in resultados_por_uid]
    completados = len(resultados_por_uid)
    print(f"  Pendientes: {len(pendientes)} usuarios\n", flush=True)

    for fold_idx, test_uid in enumerate(pendientes):
        t0 = time.time()
        pos = completados + fold_idx + 1

        # Recopilar train (todos menos el usuario test)
        train_seqs = []
        train_labels = []
        for uid in user_ids:
            if uid == test_uid:
                continue
            for feats, digito in datos_por_usuario[uid]:
                train_seqs.append(feats)
                train_labels.append(digito)

        # Test: solo el usuario excluido
        test_seqs = [feats for feats, _ in datos_por_usuario[test_uid]]
        test_labels = [digito for _, digito in datos_por_usuario[test_uid]]

        # Normalizar
        norm = NormalizadorZScore()
        train_norm = norm.ajustar_y_transformar(train_seqs)
        test_norm = norm.transformar(test_seqs)

        # Agrupar train por digito
        train_por_digito = defaultdict(list)
        for seq, label in zip(train_norm, train_labels):
            train_por_digito[label].append(seq)

        # Entrenar y evaluar (config completa: 5 restarts, 100 iter)
        modelos, _ = entrenar_todos_los_digitos(
            train_por_digito, n_estados=7, tipo_covarianza="diag",
            n_iter=100, n_restarts=5, prob_autolazo=0.6, verbose=False)

        predicciones, _, accuracy = clasificar_lote(modelos, test_norm, test_labels)
        cm = sk_confusion_matrix(test_labels, predicciones, labels=list(range(10)))

        # Acumular resultados
        resultados_por_uid[test_uid] = accuracy
        cm_acumulada += cm

        # Guardar checkpoint
        guardar_checkpoint(resultados_por_uid, cm_acumulada)

        dt = time.time() - t0
        acc_parcial = np.mean(list(resultados_por_uid.values()))
        print(f"    Usuario {pos}/{n_users} ({test_uid}): "
              f"{accuracy*100:5.1f}%  (media parcial: {acc_parcial*100:.1f}%)  "
              f"({dt:.0f}s)", flush=True)

    # Resultado final
    accuracies = list(resultados_por_uid.values())
    acc_media = np.mean(accuracies)
    acc_std = np.std(accuracies)
    elapsed = time.time() - t_inicio

    print(f"\n    Media LOO: {acc_media*100:.2f}% (+/- {acc_std*100:.2f}%)")
    print(f"    Tiempo esta sesion: {elapsed/60:.1f} minutos", flush=True)

    graficar_confusion(
        cm_acumulada,
        f"Matriz de Confusion - LOO CV ({acc_media*100:.1f}% media)",
        os.path.join(RESULTS_DIR, "confusion_LOO.png"))

    print("\n" + "=" * 70)
    print(f"  LOO CV: {acc_media*100:.2f}% +/- {acc_std*100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
