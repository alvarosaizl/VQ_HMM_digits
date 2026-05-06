"""
Actualiza el serial original (HMM->VQ en espacio LL) con las LL del
GMMHMM bien entrenado.

Estrategia:
  - Test LLs vienen de ensemble_{N74,N47}_full.json (HMM 100 iter, 10 restarts).
  - Train LLs para VQ vienen de ensemble_K5fold.json (HMM 80 iter, 6 restarts;
    LLs out-of-fold sobre los 93 usuarios sin leakage).
  - Para cada escenario se filtran las train LLs a los usuarios train
    correspondientes (74 o 47 primeros usuarios ordenados).
  - Se usa la mejor config del grid (kmeans, softmax, k=2). La normalizacion
    softmax es invariante a la escala absoluta de las LL y por tanto es
    razonablemente robusta a la diferencia de hyperparametros del HMM
    (100,10 vs 80,6).

Salida:
  Serial/resultados/serial_N74.json  (sobrescribe)
  Serial/resultados/serial_N47.json  (sobrescribe)
"""

import os, json
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ENTREGA3    = os.path.dirname(SCRIPT_DIR)
PARALEL_DIR = os.path.join(ENTREGA3, "Paralel", "resultados")
SERIAL_DIR  = os.path.join(SCRIPT_DIR, "resultados")

VQ_ALGO   = "kmeans"
VQ_NORM   = "softmax"
VQ_K      = 2

N_TRAIN = {"N74": 74, "N47": 47}


def normalize_lls(lls, mode):
    if mode == "softmax":
        s = lls - np.max(lls, axis=1, keepdims=True)
        e = np.exp(s)
        return e / np.sum(e, axis=1, keepdims=True)
    if mode == "shift":
        return lls - np.max(lls, axis=1, keepdims=True)
    if mode == "zscore":
        mu = np.mean(lls, axis=1, keepdims=True)
        sd = np.std(lls, axis=1, keepdims=True) + 1e-12
        return (lls - mu) / sd
    if mode == "raw":
        return lls
    raise ValueError(mode)


def train_vq_on_lls(lls, labels, k):
    by_d = defaultdict(list)
    for v, l in zip(lls, labels):
        by_d[int(l)].append(v)
    cbs = {}
    for d in range(10):
        if not by_d.get(d):
            cbs[d] = None; continue
        X = np.vstack(by_d[d])
        k_eff = min(k, len(X))
        km = KMeans(n_clusters=k_eff, init="k-means++", n_init=10,
                    max_iter=300, random_state=42)
        km.fit(X)
        cbs[d] = km.cluster_centers_
    return cbs


def vq_scores(cbs, lls):
    out = np.full((len(lls), 10), -np.inf)
    for i, v in enumerate(lls):
        for d in range(10):
            cb = cbs.get(d)
            if cb is None: continue
            out[i, d] = -float(np.min(np.sum((cb - v[None, :]) ** 2, axis=1)))
    return out


def run(tag):
    in_path  = os.path.join(PARALEL_DIR, f"ensemble_{tag}_full.json")
    k5_path  = os.path.join(PARALEL_DIR, "ensemble_K5fold.json")
    out_path = os.path.join(SERIAL_DIR, f"serial_{tag}.json")

    with open(in_path) as f:
        ensemble_test = json.load(f)
    with open(k5_path) as f:
        k5 = json.load(f)

    rows_test  = ensemble_test["rows"]
    rows_train = k5["rows"]

    test_uids = sorted(set(r["user_id"] for r in rows_test))
    n_train   = N_TRAIN[tag]
    all_uids  = sorted(set(r["user_id"] for r in rows_train))
    train_uids = set(all_uids[:n_train])
    rows_train_filt = [r for r in rows_train if r["user_id"] in train_uids]

    lls_train = np.array([r["hmm_lls"] for r in rows_train_filt], dtype=float)
    y_train   = np.array([r["y_true"]  for r in rows_train_filt], dtype=int)
    lls_test  = np.array([r["hmm_lls"] for r in rows_test], dtype=float)
    y_test    = np.array([r["y_true"]  for r in rows_test], dtype=int)

    print(f"[{tag}] train(VQ): {len(rows_train_filt)} muestras de "
          f"{n_train} usuarios (LLs OOF del K5fold)")
    print(f"[{tag}] test:      {len(rows_test)} muestras de "
          f"{len(test_uids)} usuarios (LLs del HMM 100/10)")

    lls_train_n = normalize_lls(lls_train, VQ_NORM)
    lls_test_n  = normalize_lls(lls_test,  VQ_NORM)

    cbs    = train_vq_on_lls(lls_train_n, y_train, VQ_K)
    scores = vq_scores(cbs, lls_test_n)

    pred_serial = np.argmax(scores, axis=1)
    pred_hmm    = np.argmax(lls_test, axis=1)
    acc_serial  = float(accuracy_score(y_test, pred_serial))
    acc_hmm     = float(accuracy_score(y_test, pred_hmm))

    print(f"[{tag}] HMM baseline (argmax LL): {acc_hmm*100:6.2f}%")
    print(f"[{tag}] Serial HMM->VQ (new HMM): {acc_serial*100:6.2f}%")

    rows_out = []
    for i, r in enumerate(rows_test):
        rows_out.append({
            "user_id": r["user_id"], "digit": r["digit"],
            "session": r["session"], "sample_id": r["sample_id"],
            "y_true": r["y_true"],
            "hmm_lls":       [float(x) for x in lls_test[i]],
            "scores_serial": [float(x) for x in scores[i]],
            "pred_hmm":    int(pred_hmm[i]),
            "pred_serial": int(pred_serial[i]),
        })

    out = {
        "tag": tag,
        "source": (f"test LLs: ensemble_{tag}_full.json (HMM 100/10); "
                   f"train LLs: ensemble_K5fold.json (HMM 80/6)"),
        "hmm_config":   ensemble_test["hmm_config"],
        "hmm_fit_test": ensemble_test["hmm_fit"],
        "hmm_fit_train_oof": k5["hmm_fit"],
        "vq_algorithm": VQ_ALGO, "vq_norm": VQ_NORM, "vq_k": VQ_K,
        "acc_hmm_baseline": acc_hmm,
        "acc_serial":       acc_serial,
        "n_train_users":    n_train,
        "n_test_users":     len(test_uids),
        "rows": rows_out,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{tag}] Guardado: {out_path}\n")


if __name__ == "__main__":
    for tag in ("N74", "N47"):
        run(tag)
