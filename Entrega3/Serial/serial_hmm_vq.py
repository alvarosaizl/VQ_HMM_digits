"""
=============================================================================
CASCADA SERIAL: HMM (likelihoods) -> VQ
=============================================================================
Idea: usar la salida del HMM (vector de 10 log-verosimilitudes) como espacio
de features para un clasificador VQ.  Si el HMM funciona razonablemente,
para una muestra del digito d la dimension d sera la mas alta -> los
vectores LL de la misma clase se agrupan en una zona del espacio 10D
("desplazamiento por dimension"). Un VQ con codebook por clase puede
capturar agrupaciones por estilo de escritura dentro de cada digito.

Pipeline:

  1. K-fold sobre USUARIOS de train (por defecto K=5):
     - En cada fold se entrena un HMM con K-1 partes y se calculan las LL
       para los usuarios de la parte holdout.
     - Asi obtenemos LL "out-of-fold" para todas las muestras de train sin
       leakage.
  2. Grid search VQ sobre las LL de train, validado con un split interno
     train/val por usuarios.  Busca:
       - algoritmo: kmeans, mbkmeans, lbg
       - k centroides por digito: 1, 2, 4, 8, 16, 32
       - normalizacion del vector LL: raw, zscore, softmax, shift
  3. Reentrenamiento final:
       - HMM se entrena con TODOS los usuarios train.
       - VQ con la mejor config sobre las LL out-of-fold de train.
  4. Evaluacion en test (HMM final aplicado a test -> LL -> VQ -> digito).

Modelo HMM (mejor segun informe E2): GMMHMM, n_mix=2, 7 estados, p=0.6,
features=med (12D).

Salida:
  resultados/grid_<TAG>.json     - resultados del grid search
  resultados/serial_<TAG>.json   - predicciones y accuracy final
  resultados/summary.json        - resumen por escenario
  plots/cm_<TAG>_<algo>.png      - matriz de confusion del VQ final y baseline
=============================================================================
"""

import os
import re
import sys
import json
import time
import warnings
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")
import logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENTREGA3_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(ENTREGA3_DIR)
ENTREGA1_DIR = os.path.join(REPO_ROOT, "Entrega1")
ENTREGA2_DIR = os.path.join(REPO_ROOT, "Entrega2")
HMM_DIR = os.path.join(ENTREGA2_DIR, "HMM")
EXTRACTOR_DIR = os.path.join(
    REPO_ROOT, "Extractores_adaptados", "Extractores", "Extractor Local"
)
DB_PATH = os.path.join(ENTREGA2_DIR, "e-BioDigit_DB", "e-BioDigit_DB")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "resultados")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

sys.path.insert(0, EXTRACTOR_DIR)
sys.path.insert(0, ENTREGA1_DIR)
sys.path.insert(0, HMM_DIR)

from clasificador_digitos import (  # noqa: E402
    cargar_base_datos, preprocesar, extraer_features,
    NormalizadorZScore, FEATURE_SUBSETS,
)
from clasificador_digitos_v3 import entrenar_gmmhmm_digito  # noqa: E402


# =============================================================================
# Configuracion
# =============================================================================

HMM_CFG = dict(
    n_mix=2, n_estados=7, cov="diag", prob_autolazo=0.6, features="med",
)
# IMPORTANTE: K-fold y HMM final usan el MISMO fit para que las LL
# train (out-of-fold) y test compartan distribucion. De lo contrario el
# VQ entrena en un espacio LL distinto al de test (distribution shift).
HMM_FIT_KFOLD = dict(n_iter=40, n_restarts=2)
HMM_FIT_FINAL = dict(n_iter=40, n_restarts=2)
# Para LOO: 93 folds, sin K-fold OOF. Pipeline simplificado.
HMM_FIT_LOO = dict(n_iter=15, n_restarts=1)
# Fallback si no hay grid_N74.json. Normalizaciones invariantes a escala.
VQ_FALLBACK = dict(algorithm="mbkmeans", norm="zscore", k=8)

K_FOLDS = 3
GRID_ALGORITHMS = ["kmeans", "mbkmeans", "lbg"]
# "raw" se descarta: depende de la escala absoluta de las LL y por tanto
# no generaliza si train y test usan HMM con distinto entrenamiento.
GRID_NORMALIZATIONS = ["zscore", "softmax", "shift"]
GRID_K_CENTROIDS = [1, 2, 4, 8, 16, 32]

INNER_VAL_FRAC = 0.20  # fraccion de usuarios train para validacion interna del grid

N_TRAIN_74 = 74
N_TRAIN_47 = 47

_FNAME_RE = re.compile(r"u(\d+)_digit_(\d)_(\d+)\.txt")


# =============================================================================
# Dataset (solo necesitamos features HMM para esta cascada)
# =============================================================================

class Sample:
    __slots__ = ("user_id", "digit", "session", "sample_id", "hmm_feats")

    def __init__(self, uid, d, s, sid, hf):
        self.user_id = uid
        self.digit = d
        self.session = s
        self.sample_id = sid
        self.hmm_feats = hf


def _parse_sample_id(filepath):
    m = _FNAME_RE.match(os.path.basename(filepath))
    return int(m.group(3)) if m else None


def build_dataset():
    print("Cargando base de datos...", flush=True)
    db, uids = cargar_base_datos(DB_PATH)
    indices_hmm = FEATURE_SUBSETS[HMM_CFG["features"]]

    by_user = defaultdict(list)
    for uid_str in uids:
        uid = int(uid_str)
        for digit in range(10):
            if digit not in db[uid_str]:
                continue
            for session in (1, 2):
                if session not in db[uid_str][digit]:
                    continue
                for muestra in db[uid_str][digit][session]:
                    sid = _parse_sample_id(muestra["filepath"])
                    if sid is None:
                        continue
                    x_h, y_h, _ = preprocesar(
                        muestra["x"], muestra["y"], muestra["timestamp"],
                        n_resample=80, suavizar=True,
                    )
                    hmm_feats = extraer_features(
                        x_h, y_h, muestra["presion"],
                        indices_features=indices_hmm,
                    )
                    by_user[uid].append(Sample(
                        uid, digit, session, sid, hmm_feats))

    user_ids = sorted(by_user.keys())
    print(f"  {len(user_ids)} usuarios, "
          f"{sum(len(v) for v in by_user.values())} muestras", flush=True)
    return by_user, user_ids


# =============================================================================
# HMM helpers
# =============================================================================

def train_hmm(train_samples, fit_opts):
    seqs = [s.hmm_feats for s in train_samples]
    labels = [s.digit for s in train_samples]
    norm = NormalizadorZScore()
    seqs_n = norm.ajustar_y_transformar(seqs)

    by_d = defaultdict(list)
    for sq, l in zip(seqs_n, labels):
        by_d[l].append(sq)

    modelos = {}
    for d in range(10):
        if not by_d.get(d):
            modelos[d] = None
            continue
        m, _ = entrenar_gmmhmm_digito(
            by_d[d],
            n_estados=HMM_CFG["n_estados"],
            n_mix=HMM_CFG["n_mix"],
            tipo_covarianza=HMM_CFG["cov"],
            n_iter=fit_opts["n_iter"],
            n_restarts=fit_opts["n_restarts"],
            prob_autolazo=HMM_CFG["prob_autolazo"],
            semilla_base=d * 100,
        )
        modelos[d] = m
    return modelos, norm


def hmm_log_likelihoods(modelos, norm, samples):
    """(N, 10) matrix of HMM log-likelihoods per digit.

    Rechaza scores patologicos (|LL| > 1e6) que indican que un modelo HMM
    colapso durante el entrenamiento. Estos pasan a -inf y luego al
    suelo finito, para que no contaminen normalizaciones posteriores.
    """
    seqs_n = norm.transformar([s.hmm_feats for s in samples])
    out = np.full((len(samples), 10), -np.inf, dtype=float)
    SANE_LIMIT = 1e6
    for i, sq in enumerate(seqs_n):
        for d in range(10):
            mdl = modelos.get(d)
            if mdl is None:
                continue
            try:
                s = mdl.score(sq)
                if not np.isfinite(s) or abs(s) > SANE_LIMIT:
                    continue
                out[i, d] = s
            except (ValueError, np.linalg.LinAlgError):
                pass
    finite = np.isfinite(out)
    if finite.any():
        floor = float(np.min(out[finite])) - 1e3
        out[~finite] = floor
    else:
        out[~finite] = -1e6
    return out


# =============================================================================
# Out-of-fold LL extraction sobre usuarios train (K-fold sin leakage)
# =============================================================================

def kfold_user_split(user_ids, k):
    """Devuelve lista de (train_uids, val_uids) con K particiones de usuarios."""
    rng = np.random.default_rng(0)
    perm = rng.permutation(user_ids)
    folds = np.array_split(perm, k)
    splits = []
    for i in range(k):
        val = list(map(int, folds[i]))
        train = [int(u) for j in range(k) if j != i for u in folds[j]]
        splits.append((train, val))
    return splits


def extract_oof_lls(by_user, train_user_ids, fit_opts, tag, k_folds=K_FOLDS):
    """Out-of-fold log-likelihoods para todas las muestras de los usuarios train.

    Devuelve (lls, samples) alineados; samples contiene la muestra original.
    """
    splits = kfold_user_split(train_user_ids, k_folds)
    out_lls = []
    out_samples = []
    for fold_idx, (tr_uids, val_uids) in enumerate(splits, 1):
        t0 = time.time()
        tr_samples = [s for u in tr_uids for s in by_user[u]]
        val_samples = [s for u in val_uids for s in by_user[u]]
        print(f"  [{tag}] K-fold {fold_idx}/{len(splits)} "
              f"(train usuarios={len(tr_uids)}, val usuarios={len(val_uids)})...",
              flush=True)
        modelos, norm = train_hmm(tr_samples, fit_opts)
        lls = hmm_log_likelihoods(modelos, norm, val_samples)
        out_lls.append(lls)
        out_samples.extend(val_samples)
        # Accuracy del HMM "tal cual" sobre la particion val (referencia)
        acc = float(np.mean(
            np.argmax(lls, axis=1) ==
            np.array([s.digit for s in val_samples])))
        print(f"  [{tag}] fold {fold_idx} HMM acc={acc*100:.2f}% "
              f"({time.time()-t0:.0f}s)", flush=True)
    return np.vstack(out_lls), out_samples


# =============================================================================
# Normalizaciones del vector LL
# =============================================================================

def normalize_lls(lls, mode):
    if mode == "raw":
        return lls
    if mode == "shift":
        return lls - np.max(lls, axis=1, keepdims=True)
    if mode == "softmax":
        s = lls - np.max(lls, axis=1, keepdims=True)
        e = np.exp(s)
        return e / np.sum(e, axis=1, keepdims=True)
    if mode == "zscore":
        mu = np.mean(lls, axis=1, keepdims=True)
        sd = np.std(lls, axis=1, keepdims=True) + 1e-12
        return (lls - mu) / sd
    raise ValueError(f"normalizacion desconocida: {mode}")


# =============================================================================
# VQ sobre el espacio LL (per-class codebook, single 10D vector per sample)
# =============================================================================

def build_codebook(X, k, algorithm, random_state=42):
    if len(X) == 0:
        return None
    k_eff = min(k, len(X))
    if algorithm == "kmeans":
        km = KMeans(n_clusters=k_eff, init="k-means++", n_init=10,
                    max_iter=300, random_state=random_state)
        km.fit(X)
        return km.cluster_centers_
    if algorithm == "mbkmeans":
        km = MiniBatchKMeans(
            n_clusters=k_eff, init="k-means++", n_init=10,
            max_iter=300, batch_size=min(1024, len(X)),
            random_state=random_state)
        km.fit(X)
        return km.cluster_centers_
    if algorithm == "lbg":
        return _lbg(X, k_eff)
    raise ValueError(f"algoritmo desconocido: {algorithm}")


def _lbg(X, k, epsilon=0.01, max_lloyd=50):
    centroids = np.mean(X, axis=0, keepdims=True)
    while len(centroids) < k:
        perturb = epsilon * (np.std(X, axis=0, keepdims=True) + 1e-8)
        new_c = np.vstack([centroids + perturb, centroids - perturb])
        if len(new_c) > k:
            new_c = new_c[:k]
        centroids = new_c
        for _ in range(max_lloyd):
            dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            assign = np.argmin(dists, axis=1)
            updated = np.zeros_like(centroids)
            for i in range(len(centroids)):
                mask = assign == i
                if np.any(mask):
                    updated[i] = np.mean(X[mask], axis=0)
                else:
                    updated[i] = X[np.random.randint(len(X))]
            if np.allclose(centroids, updated, atol=1e-6):
                break
            centroids = updated
    return centroids


def train_vq_on_lls(lls, labels, algorithm, k):
    """Entrena un codebook por digito sobre vectores LL 10D."""
    by_d = defaultdict(list)
    for v, l in zip(lls, labels):
        by_d[l].append(v)
    codebooks = {}
    for d in range(10):
        if not by_d.get(d):
            codebooks[d] = None
            continue
        X = np.vstack(by_d[d])
        codebooks[d] = build_codebook(X, k, algorithm)
    return codebooks


def predict_vq_on_lls(codebooks, lls):
    """Para cada vector LL devuelve la clase con menor distancia al
    centroide mas cercano de su codebook."""
    scores = vq_scores_on_lls(codebooks, lls)
    return np.argmax(scores, axis=1)


def vq_scores_on_lls(codebooks, lls):
    """Score por clase = -min_distancia_a_centroide_clase. (N, 10).

    Mayor = mas probable (utilizable directamente para AUC/EER one-vs-rest).
    """
    N = len(lls)
    out = np.full((N, 10), -np.inf, dtype=float)
    for i, v in enumerate(lls):
        for d in range(10):
            cb = codebooks.get(d)
            if cb is None:
                continue
            dists = np.sum((cb - v[None, :]) ** 2, axis=1)
            out[i, d] = -float(np.min(dists))
    return out


# =============================================================================
# Grid search sobre LL train (validacion por hold-out de usuarios)
# =============================================================================

def split_users_inner(samples, val_frac=INNER_VAL_FRAC, seed=0):
    uids = sorted({s.user_id for s in samples})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uids)
    n_val = max(1, int(round(len(uids) * val_frac)))
    val_set = set(map(int, perm[:n_val]))
    train_set = set(map(int, perm[n_val:]))
    return train_set, val_set


def grid_search_vq(lls_train, samples_train):
    """Devuelve (lista de resultados ordenada, mejor config)."""
    train_set, val_set = split_users_inner(samples_train)
    idx_tr = [i for i, s in enumerate(samples_train) if s.user_id in train_set]
    idx_va = [i for i, s in enumerate(samples_train) if s.user_id in val_set]

    labels_train = np.array([s.digit for s in samples_train])
    y_tr = labels_train[idx_tr]
    y_va = labels_train[idx_va]

    results = []
    total = len(GRID_ALGORITHMS) * len(GRID_NORMALIZATIONS) * len(GRID_K_CENTROIDS)
    print(f"  Grid search VQ: {total} configuraciones "
          f"(inner train={len(idx_tr)}, val={len(idx_va)} muestras)", flush=True)
    cnt = 0
    for norm_mode in GRID_NORMALIZATIONS:
        lls_n = normalize_lls(lls_train, norm_mode)
        for algo in GRID_ALGORITHMS:
            for k in GRID_K_CENTROIDS:
                cnt += 1
                t0 = time.time()
                cb = train_vq_on_lls(lls_n[idx_tr], y_tr, algo, k)
                preds = predict_vq_on_lls(cb, lls_n[idx_va])
                acc = float(accuracy_score(y_va, preds))
                results.append({
                    "algorithm": algo, "norm": norm_mode, "k": k,
                    "accuracy": acc, "time_s": time.time() - t0,
                })
                print(f"    [{cnt:3d}/{total}] norm={norm_mode:7s} "
                      f"algo={algo:9s} k={k:3d} -> "
                      f"acc={acc*100:5.2f}% ({time.time()-t0:.1f}s)",
                      flush=True)
    results.sort(key=lambda r: -r["accuracy"])
    best = results[0]
    print(f"\n  Mejor config grid: algo={best['algorithm']} "
          f"norm={best['norm']} k={best['k']} acc={best['accuracy']*100:.2f}%",
          flush=True)
    return results, best


# =============================================================================
# Plots
# =============================================================================

def plot_confusion(cm, title, filepath):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10), ax=ax)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_grid(results, filepath, top_n=20):
    top = results[:top_n]
    labels = [f"{r['algorithm'][:3]}|{r['norm'][:3]}|k={r['k']}" for r in top]
    accs = [r["accuracy"] * 100 for r in top]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top)), accs, color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy validacion (%)")
    ax.set_title(f"Top {top_n} configuraciones VQ sobre LL")
    for i, a in enumerate(accs):
        ax.text(a + 0.05, i, f"{a:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


# =============================================================================
# Escenario completo
# =============================================================================

def run_scenario(by_user, user_ids, n_train, tag):
    train_uids = list(user_ids[:n_train])
    test_uids = list(user_ids[n_train:])
    train_samples = [s for u in train_uids for s in by_user[u]]
    test_samples = [s for u in test_uids for s in by_user[u]]

    print(f"\n[{tag}] train usuarios={len(train_uids)} ({len(train_samples)}m) "
          f"| test usuarios={len(test_uids)} ({len(test_samples)}m)", flush=True)

    # 1. K-fold OOF likelihoods sobre train
    print(f"\n[{tag}] Extrayendo LL out-of-fold sobre train ({K_FOLDS}-fold)...",
          flush=True)
    t0 = time.time()
    lls_train, ord_train_samples = extract_oof_lls(
        by_user, train_uids, HMM_FIT_KFOLD, tag)
    print(f"[{tag}] OOF train listo ({time.time()-t0:.0f}s)", flush=True)

    # 2. Grid search VQ
    print(f"\n[{tag}] Grid search VQ sobre LL...", flush=True)
    t0 = time.time()
    grid_results, best = grid_search_vq(lls_train, ord_train_samples)
    print(f"[{tag}] Grid search listo ({(time.time()-t0)/60:.1f} min)", flush=True)

    plot_grid(grid_results,
              os.path.join(PLOTS_DIR, f"grid_top_{tag}.png"))

    with open(os.path.join(RESULTS_DIR, f"grid_{tag}.json"), "w") as f:
        json.dump({
            "tag": tag, "best": best, "all": grid_results,
            "k_folds": K_FOLDS,
            "hmm_config": HMM_CFG, "hmm_fit_kfold": HMM_FIT_KFOLD,
        }, f, indent=2)

    # 3. HMM final sobre TODOS los usuarios train
    print(f"\n[{tag}] Entrenando HMM final ({HMM_FIT_FINAL})...", flush=True)
    t0 = time.time()
    hmm_models, hmm_norm = train_hmm(train_samples, HMM_FIT_FINAL)
    print(f"[{tag}] HMM final listo ({time.time()-t0:.0f}s)", flush=True)

    # 4. LL sobre test
    lls_test = hmm_log_likelihoods(hmm_models, hmm_norm, test_samples)
    y_test = np.array([s.digit for s in test_samples])

    # Baseline: HMM puro sobre test
    pred_hmm = np.argmax(lls_test, axis=1)
    acc_hmm = float(accuracy_score(y_test, pred_hmm))

    # 5. Entrenar VQ con la mejor config sobre TODAS las LL OOF de train
    norm_mode = best["norm"]
    algo = best["algorithm"]
    k = best["k"]
    lls_train_n = normalize_lls(lls_train, norm_mode)
    lls_test_n = normalize_lls(lls_test, norm_mode)
    y_train = np.array([s.digit for s in ord_train_samples])

    cb_final = train_vq_on_lls(lls_train_n, y_train, algo, k)
    serial_scores = vq_scores_on_lls(cb_final, lls_test_n)
    pred_vq = np.argmax(serial_scores, axis=1)
    acc_vq = float(accuracy_score(y_test, pred_vq))

    print(f"\n[{tag}] Resultados test:")
    print(f"    HMM baseline (argmax LL):      {acc_hmm*100:6.2f}%")
    print(f"    Serial HMM->VQ ({algo}/{norm_mode}/k={k}): {acc_vq*100:6.2f}%")

    # Confusion matrices
    plot_confusion(
        confusion_matrix(y_test, pred_hmm, labels=list(range(10))),
        f"{tag} - HMM baseline ({acc_hmm*100:.2f}%)",
        os.path.join(PLOTS_DIR, f"cm_{tag}_hmm_baseline.png"),
    )
    plot_confusion(
        confusion_matrix(y_test, pred_vq, labels=list(range(10))),
        f"{tag} - Serial HMM->VQ {algo} norm={norm_mode} k={k} "
        f"({acc_vq*100:.2f}%)",
        os.path.join(PLOTS_DIR, f"cm_{tag}_serial.png"),
    )

    # Persistencia (incluye scores 10D para AUC/EER)
    rows = []
    for i, s in enumerate(test_samples):
        rows.append({
            "user_id": s.user_id, "digit": s.digit,
            "session": s.session, "sample_id": s.sample_id,
            "y_true": s.digit,
            "hmm_lls": [float(x) for x in lls_test[i]],
            "scores_serial": [float(x) for x in serial_scores[i]],
            "pred_hmm": int(pred_hmm[i]),
            "pred_serial": int(pred_vq[i]),
        })

    summary = {
        "tag": tag,
        "hmm_config": HMM_CFG,
        "hmm_fit_kfold": HMM_FIT_KFOLD,
        "hmm_fit_final": HMM_FIT_FINAL,
        "k_folds": K_FOLDS,
        "best_vq": best,
        "acc_hmm_baseline": acc_hmm,
        "acc_serial": acc_vq,
        "n_train_users": len(train_uids),
        "n_test_users": len(test_uids),
    }
    with open(os.path.join(RESULTS_DIR, f"serial_{tag}.json"), "w") as f:
        json.dump({**summary, "rows": rows}, f, indent=2)
    return summary


# =============================================================================
# Leave-One-User-Out (pipeline simplificado, sin K-fold OOF)
# =============================================================================

def _load_best_vq_config():
    """Lee la mejor config del grid si grid_N74.json existe.
    En caso contrario usa VQ_FALLBACK."""
    grid_path = os.path.join(RESULTS_DIR, "grid_N74.json")
    if os.path.exists(grid_path):
        with open(grid_path) as f:
            data = json.load(f)
        best = data["best"]
        print(f"  [LOO] usando config del grid N74: {best['algorithm']} "
              f"norm={best['norm']} k={best['k']} "
              f"(val acc={best['accuracy']*100:.2f}%)", flush=True)
        return best["algorithm"], best["norm"], best["k"]
    print(f"  [LOO] grid_N74.json no encontrado, usando fallback "
          f"{VQ_FALLBACK}", flush=True)
    return VQ_FALLBACK["algorithm"], VQ_FALLBACK["norm"], VQ_FALLBACK["k"]


def run_loo(by_user, user_ids):
    """LOO con pipeline simplificado: HMM entrenado en 92 usuarios; LL
    extraidas para esos 92 (con leakage controlado) entrenan el VQ; luego
    se evalua en el usuario holdout. Esto evita el K-fold anidado, que
    seria prohibitivamente caro (5 x 93 = 465 entrenamientos HMM)."""
    tag = "LOO"
    out_path = os.path.join(RESULTS_DIR, f"serial_{tag}.json")
    ckpt_path = os.path.join(RESULTS_DIR, f"checkpoint_{tag}.json")

    algo, norm_mode, k = _load_best_vq_config()

    rows_by_uid = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            rows_by_uid = {int(k_): v for k_, v in json.load(f).items()}
        print(f"  [LOO] checkpoint: {len(rows_by_uid)}/{len(user_ids)} "
              f"usuarios ya completados", flush=True)

    for i, test_uid in enumerate(user_ids, 1):
        if test_uid in rows_by_uid:
            continue
        t0 = time.time()
        train_samples = [s for u in user_ids if u != test_uid
                         for s in by_user[u]]
        test_samples = by_user[test_uid]

        # 1. HMM en 92 usuarios
        hmm_models, hmm_norm = train_hmm(train_samples, HMM_FIT_LOO)

        # 2. LL para todos los samples de train (leaky) + test
        lls_train = hmm_log_likelihoods(hmm_models, hmm_norm, train_samples)
        lls_test = hmm_log_likelihoods(hmm_models, hmm_norm, test_samples)
        y_train = np.array([s.digit for s in train_samples])
        y_test = np.array([s.digit for s in test_samples])

        # 3. VQ en LL train (con leakage; ver justificacion en docstring)
        lls_train_n = normalize_lls(lls_train, norm_mode)
        lls_test_n = normalize_lls(lls_test, norm_mode)
        codebooks = train_vq_on_lls(lls_train_n, y_train, algo, k)

        # 4. Inferencia
        scores_serial = vq_scores_on_lls(codebooks, lls_test_n)
        pred_serial = np.argmax(scores_serial, axis=1)
        pred_hmm = np.argmax(lls_test, axis=1)

        user_rows = []
        for j, s in enumerate(test_samples):
            user_rows.append({
                "user_id": s.user_id, "digit": s.digit,
                "session": s.session, "sample_id": s.sample_id,
                "y_true": s.digit,
                "hmm_lls": [float(x) for x in lls_test[j]],
                "scores_serial": [float(x) for x in scores_serial[j]],
                "pred_hmm": int(pred_hmm[j]),
                "pred_serial": int(pred_serial[j]),
            })
        rows_by_uid[test_uid] = user_rows

        with open(ckpt_path, "w") as f:
            json.dump(rows_by_uid, f)

        acc_h = float(np.mean(pred_hmm == y_test))
        acc_s = float(np.mean(pred_serial == y_test))
        print(f"  [LOO {len(rows_by_uid):2d}/{len(user_ids)}] uid={test_uid} "
              f"hmm={acc_h*100:5.1f}% serial={acc_s*100:5.1f}% "
              f"({time.time()-t0:.0f}s)", flush=True)

    # Flatten
    all_rows = []
    for uid in user_ids:
        all_rows.extend(rows_by_uid[uid])
    y_true = np.array([r["y_true"] for r in all_rows])
    pred_hmm = np.array([r["pred_hmm"] for r in all_rows])
    pred_serial = np.array([r["pred_serial"] for r in all_rows])

    acc_hmm = float(accuracy_score(y_true, pred_hmm))
    acc_serial = float(accuracy_score(y_true, pred_serial))

    print(f"\n[LOO] HMM baseline: {acc_hmm*100:.2f}%")
    print(f"[LOO] Serial:       {acc_serial*100:.2f}%")

    plot_confusion(
        confusion_matrix(y_true, pred_hmm, labels=list(range(10))),
        f"LOO - HMM baseline ({acc_hmm*100:.2f}%)",
        os.path.join(PLOTS_DIR, f"cm_{tag}_hmm_baseline.png"),
    )
    plot_confusion(
        confusion_matrix(y_true, pred_serial, labels=list(range(10))),
        f"LOO - Serial HMM->VQ ({acc_serial*100:.2f}%)",
        os.path.join(PLOTS_DIR, f"cm_{tag}_serial.png"),
    )

    summary = {
        "tag": tag,
        "hmm_config": HMM_CFG,
        "hmm_fit_loo": HMM_FIT_LOO,
        "vq_algorithm": algo,
        "vq_norm": norm_mode,
        "vq_k": k,
        "acc_hmm_baseline": acc_hmm,
        "acc_serial": acc_serial,
        "n_folds": len(user_ids),
        "loo_pipeline": "simplified (no inner K-fold OOF; "
                         "VQ trained on leaky train LLs)",
    }
    with open(out_path, "w") as f:
        json.dump({**summary, "rows": all_rows}, f, indent=2)
    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  CASCADA SERIAL HMM (LL) -> VQ")
    print("=" * 70, flush=True)

    by_user, user_ids = build_dataset()
    scenarios = os.environ.get("SCENARIOS", "N74,N47,LOO").split(",")

    summary = {}
    if "N74" in scenarios:
        summary["N74"] = run_scenario(by_user, user_ids, N_TRAIN_74, "N74")
    if "N47" in scenarios:
        summary["N47"] = run_scenario(by_user, user_ids, N_TRAIN_47, "N47")
    if "LOO" in scenarios:
        summary["LOO"] = run_loo(by_user, user_ids)

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
