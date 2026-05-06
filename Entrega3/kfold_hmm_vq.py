#!/usr/bin/env python3
"""
=============================================================================
KFOLD HMM + VQ  —  K=5 pliegues por usuario
=============================================================================
Equivalente aproximado al LOO pero con modelos bien entrenados:
  - GMMHMM: n_iter=80, n_restarts=6  (vs 15/1 del LOO del ensemble)
  - VQ:     K=128, 5D pos_ang_curv   (igual que ensemble_paralelo)

Cada fold entrena con ~74 usuarios y testa con ~19, cubriendo los 93 en
total. El resultado es directamente comparable con ensemble_LOO.json.

Checkpoint por fold → se puede interrumpir y retomar sin perder trabajo.

Salida:
  Paralel/resultados/ensemble_K5fold.json  — formato idéntico a ensemble_LOO.json
  kfold_checkpoint.json                    — progreso (fold→rows)
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")
import logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)
ENTREGA1_DIR = os.path.join(REPO_ROOT, "Entrega1")
ENTREGA2_DIR = os.path.join(REPO_ROOT, "Entrega2")
HMM_DIR      = os.path.join(ENTREGA2_DIR, "HMM")
VQ_DIR       = os.path.join(ENTREGA2_DIR, "VQ")
EXTRACTOR_DIR = os.path.join(
    REPO_ROOT, "Extractores_adaptados", "Extractores", "Extractor Local"
)
DB_PATH      = os.path.join(ENTREGA2_DIR, "e-BioDigit_DB", "e-BioDigit_DB")

RESULTS_DIR  = os.path.join(SCRIPT_DIR, "Paralel", "resultados")
PLOTS_DIR    = os.path.join(SCRIPT_DIR, "Paralel", "plots")
CKPT_PATH    = os.path.join(SCRIPT_DIR, "kfold_checkpoint.json")
OUT_PATH     = os.path.join(RESULTS_DIR, "ensemble_K5fold.json")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

sys.path.insert(0, EXTRACTOR_DIR)
sys.path.insert(0, ENTREGA1_DIR)
sys.path.insert(0, HMM_DIR)
sys.path.insert(0, VQ_DIR)

from clasificador_digitos import (           # noqa: E402
    cargar_base_datos, preprocesar, extraer_features,
    NormalizadorZScore, FEATURE_SUBSETS,
)
from clasificador_digitos_v3 import entrenar_gmmhmm_digito   # noqa: E402
from busqueda_VQ import (                    # noqa: E402
    preprocess_trace as vq_preprocess,
    compute_full_features,
)

# ── Configuración ─────────────────────────────────────────────────────────────

K_FOLDS       = 5
RANDOM_SEED   = 42

HMM_CFG = dict(
    n_mix=2, n_estados=7, cov="diag",
    prob_autolazo=0.6, features="med",
)
HMM_FIT = dict(n_iter=80, n_restarts=6)   # parámetros completos

VQ_FEATURES_IDX = [0, 1, 19, 20, 10]      # x, y, sin, cos, dtheta  (5D)
VQ_K            = 128
VQ_FEATURE_NAME = "pos_ang_curv"

_FNAME_RE = re.compile(r"u(\d+)_digit_(\d)_(\d+)\.txt")

# ── Dataset ───────────────────────────────────────────────────────────────────

class Sample:
    __slots__ = ("user_id", "digit", "session", "sample_id", "hmm_feats", "vq_feats")
    def __init__(self, uid, d, s, sid, hf, vf):
        self.user_id = uid; self.digit = d; self.session = s
        self.sample_id = sid; self.hmm_feats = hf; self.vq_feats = vf


def build_dataset():
    print("Cargando base de datos...", flush=True)
    db, uids = cargar_base_datos(DB_PATH)
    idx_hmm  = FEATURE_SUBSETS[HMM_CFG["features"]]
    by_user  = defaultdict(list)

    for uid_str in uids:
        uid = int(uid_str)
        for digit in range(10):
            if digit not in db[uid_str]:
                continue
            for session in (1, 2):
                if session not in db[uid_str][digit]:
                    continue
                for m in db[uid_str][digit][session]:
                    m_id = _FNAME_RE.match(os.path.basename(m["filepath"]))
                    sid  = int(m_id.group(3)) if m_id else None
                    if sid is None:
                        continue
                    x_h, y_h, _ = preprocesar(
                        m["x"], m["y"], m["timestamp"], n_resample=80, suavizar=True)
                    hf = extraer_features(x_h, y_h, m["presion"], indices_features=idx_hmm)

                    x_v, y_v, _ = vq_preprocess(
                        np.array(m["x"], float), np.array(m["y"], float),
                        np.array(m["timestamp"], float))
                    full = compute_full_features(x_v, y_v, m["presion"])
                    vf   = full[:, VQ_FEATURES_IDX]

                    by_user[uid].append(Sample(uid, digit, session, sid, hf, vf))

    user_ids = sorted(by_user.keys())
    print(f"  {len(user_ids)} usuarios, "
          f"{sum(len(v) for v in by_user.values())} muestras", flush=True)
    return by_user, user_ids


def make_folds(user_ids, k=5, seed=42):
    """Divide los usuarios en k pliegues balanceados."""
    rng   = np.random.default_rng(seed)
    ids   = list(user_ids)
    rng.shuffle(ids)
    return [ids[i::k] for i in range(k)]

# ── Entrenamiento HMM ─────────────────────────────────────────────────────────

def train_hmm(train_samples):
    seqs   = [s.hmm_feats for s in train_samples]
    labels = [s.digit     for s in train_samples]
    norm   = NormalizadorZScore()
    seqs_n = norm.ajustar_y_transformar(seqs)
    by_d   = defaultdict(list)
    for sq, l in zip(seqs_n, labels):
        by_d[l].append(sq)

    modelos = {}
    t0 = time.time()
    for d in range(10):
        if not by_d.get(d):
            modelos[d] = None; continue
        model, score = entrenar_gmmhmm_digito(
            by_d[d],
            n_estados    = HMM_CFG["n_estados"],
            n_mix        = HMM_CFG["n_mix"],
            tipo_covarianza = HMM_CFG["cov"],
            n_iter       = HMM_FIT["n_iter"],
            n_restarts   = HMM_FIT["n_restarts"],
            prob_autolazo = HMM_CFG["prob_autolazo"],
            semilla_base = d * 100,
        )
        modelos[d] = model
        print(f"    dígito {d}: {len(by_d[d])} seqs  LL={score:.0f}  "
              f"({time.time()-t0:.0f}s acum.)", flush=True)
    return modelos, norm


def hmm_log_likelihoods(modelos, norm, samples):
    seqs_n = norm.transformar([s.hmm_feats for s in samples])
    out    = np.full((len(samples), 10), -np.inf)
    LIMIT  = 1e6
    for i, sq in enumerate(seqs_n):
        for d in range(10):
            mdl = modelos.get(d)
            if mdl is None: continue
            try:
                sc = mdl.score(sq)
                if np.isfinite(sc) and abs(sc) < LIMIT:
                    out[i, d] = sc
            except Exception:
                pass
    finite = np.isfinite(out)
    floor  = float(np.min(out[finite])) - 1e3 if finite.any() else -1e6
    out[~finite] = floor
    return out

# ── Entrenamiento VQ ──────────────────────────────────────────────────────────

def train_vq(train_samples):
    by_d = defaultdict(list)
    for s in train_samples:
        by_d[s.digit].append(s.vq_feats)
    cbs = {}
    for d in range(10):
        if not by_d.get(d): cbs[d] = None; continue
        X  = np.vstack(by_d[d])
        k  = min(VQ_K, len(X))
        km = MiniBatchKMeans(n_clusters=k, init="k-means++", n_init=10,
                             max_iter=300, batch_size=min(1024, len(X)),
                             random_state=VQ_RANDOM_STATE if 'VQ_RANDOM_STATE' in dir()
                             else 42)
        km.fit(X)
        cbs[d] = km.cluster_centers_
    return cbs


def vq_distortions(cbs, samples):
    out = np.full((len(samples), 10), np.inf)
    for i, s in enumerate(samples):
        F = s.vq_feats
        for d, cb in cbs.items():
            if cb is None: continue
            d2 = np.sum((F[:, None, :] - cb[None, :, :]) ** 2, axis=2)
            out[i, d] = float(np.mean(np.min(d2, axis=1)))
    return out

# ── Fusión ────────────────────────────────────────────────────────────────────

def _norm_softmax(sc):
    mu = sc.mean(1, keepdims=True)
    sd = sc.std(1,  keepdims=True) + 1e-12
    s  = (sc - mu) / sd - np.max((sc - mu) / sd, 1, keepdims=True)
    e  = np.exp(s)
    return e / e.sum(1, keepdims=True)


def fusion_scores(hmm_lls, vq_dists):
    hmm_p = _norm_softmax(hmm_lls)
    vq_p  = _norm_softmax(-vq_dists)

    conf_hmm   = hmm_p.max(1); conf_vq = vq_p.max(1)
    margin_hmm = np.sort(hmm_p, 1)[:, -1] - np.sort(hmm_p, 1)[:, -2]
    margin_vq  = np.sort(vq_p,  1)[:, -1] - np.sort(vq_p,  1)[:, -2]

    soft  = 0.5 * hmm_p + 0.5 * vq_p
    wh    = conf_hmm[:, None]; wv = conf_vq[:, None]
    cw    = (wh * hmm_p + wv * vq_p) / (wh + wv + 1e-12)
    mw    = (margin_hmm[:, None] * hmm_p + margin_vq[:, None] * vq_p)
    mw   /= (margin_hmm[:, None] + margin_vq[:, None] + 1e-12)

    pred_hmm = np.argmax(hmm_lls,  1); pred_vq = np.argmin(vq_dists, 1)
    agree    = pred_hmm == pred_vq
    pick_hmm = (conf_hmm >= conf_vq) | agree
    agr_p    = np.where(pick_hmm[:, None], hmm_p, vq_p)

    return {
        "hmm": hmm_p, "vq": vq_p, "agreement": agr_p,
        "soft": soft, "conf_weighted": cw, "margin_weighted": mw,
    }, {
        "conf_hmm": conf_hmm, "conf_vq": conf_vq,
        "margin_hmm": margin_hmm, "margin_vq": margin_vq,
    }

# ── Evaluación de un fold ─────────────────────────────────────────────────────

def evaluate_fold(fold_idx, train_uids, test_uids, by_user):
    train_s = [s for u in train_uids for s in by_user[u]]
    test_s  = [s for u in test_uids  for s in by_user[u]]

    print(f"\n[Fold {fold_idx+1}/{K_FOLDS}] "
          f"train={len(train_uids)} usuarios ({len(train_s)} muestras)  "
          f"test={len(test_uids)} usuarios ({len(test_s)} muestras)", flush=True)

    print(f"  Entrenando GMMHMM "
          f"(n_iter={HMM_FIT['n_iter']}, n_restarts={HMM_FIT['n_restarts']})...",
          flush=True)
    t0 = time.time()
    hmm_m, hmm_norm = train_hmm(train_s)
    print(f"  HMM listo ({time.time()-t0:.0f}s)", flush=True)

    print(f"  Entrenando VQ (K={VQ_K}, {VQ_FEATURE_NAME})...", flush=True)
    t0 = time.time()
    vq_cbs = train_vq(train_s)
    print(f"  VQ listo ({time.time()-t0:.0f}s)", flush=True)

    print("  Inferencia...", flush=True)
    lls   = hmm_log_likelihoods(hmm_m, hmm_norm, test_s)
    dists = vq_distortions(vq_cbs, test_s)
    scores, meta = fusion_scores(lls, dists)
    preds = {k: np.argmax(v, 1) for k, v in scores.items()}

    y = np.array([s.digit for s in test_s])
    for k in ("hmm", "vq", "margin_weighted"):
        print(f"    {k:18s}: {accuracy_score(y, preds[k])*100:.2f}%", flush=True)

    rows = []
    for i, s in enumerate(test_s):
        rows.append({
            "user_id": s.user_id, "digit": s.digit,
            "session": s.session, "sample_id": s.sample_id,
            "y_true": s.digit,
            "hmm_lls":  [float(x) for x in lls[i]],
            "vq_dists": [float(x) for x in dists[i]],
            "conf_hmm":   float(meta["conf_hmm"][i]),
            "conf_vq":    float(meta["conf_vq"][i]),
            "margin_hmm": float(meta["margin_hmm"][i]),
            "margin_vq":  float(meta["margin_vq"][i]),
            **{f"pred_{k}":   int(preds[k][i])          for k in preds},
            **{f"scores_{k}": [float(x) for x in scores[k][i]] for k in scores},
        })
    return rows

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()

    print("=" * 70)
    print(f"  K-FOLD HMM+VQ  (K={K_FOLDS}, n_iter={HMM_FIT['n_iter']}, "
          f"n_restarts={HMM_FIT['n_restarts']})")
    print("=" * 70, flush=True)

    if os.path.exists(OUT_PATH):
        print(f"\nResultado ya existe: {OUT_PATH}")
        print("Borrarlo manualmente si quieres reejecutar. Saliendo.")
        return

    by_user, user_ids = build_dataset()
    folds = make_folds(user_ids, k=K_FOLDS, seed=RANDOM_SEED)
    print(f"\nPliegues ({K_FOLDS} folds):")
    for i, f in enumerate(folds):
        print(f"  Fold {i+1}: {len(f)} usuarios — {f}", flush=True)

    # ── Cargar checkpoint ─────────────────────────────────────────────────────
    completed = {}
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            completed = {int(k): v for k, v in json.load(f).items()}
        print(f"\nCheckpoint: {len(completed)}/{K_FOLDS} folds completados",
              flush=True)

    # ── Ejecutar folds pendientes ─────────────────────────────────────────────
    for fold_idx in range(K_FOLDS):
        if fold_idx in completed:
            print(f"\n[Fold {fold_idx+1}/{K_FOLDS}] Ya completado, saltando.",
                  flush=True)
            continue

        test_uids  = set(folds[fold_idx])
        train_uids = set(u for i, f in enumerate(folds)
                         if i != fold_idx for u in f)

        fold_rows = evaluate_fold(fold_idx, train_uids, test_uids, by_user)
        completed[fold_idx] = fold_rows

        with open(CKPT_PATH, "w") as f:
            json.dump(completed, f)
        print(f"  Checkpoint guardado ({len(completed)}/{K_FOLDS} folds)",
              flush=True)

    # ── Consolidar resultado final ────────────────────────────────────────────
    all_rows = []
    for fold_idx in range(K_FOLDS):
        all_rows.extend(completed[fold_idx])

    y_all = np.array([r["y_true"] for r in all_rows])
    methods = ("hmm", "vq", "agreement", "soft", "conf_weighted", "margin_weighted")
    preds_all = {m: np.array([r[f"pred_{m}"] for r in all_rows]) for m in methods}
    accs = {m: float(accuracy_score(y_all, preds_all[m])) for m in methods}
    accs["oracle"] = float(np.mean(
        (preds_all["hmm"] == y_all) | (preds_all["vq"] == y_all)))

    print("\n── Accuracies globales (K5-fold) ───────────────────────────────────")
    for m in methods + ("oracle",):
        print(f"  {m:22s}: {accs[m]*100:.2f}%")

    result = {
        "tag": "K5fold",
        "k_folds": K_FOLDS,
        "hmm_config": HMM_CFG,
        "hmm_fit": HMM_FIT,
        "vq_features": VQ_FEATURE_NAME,
        "vq_features_idx": VQ_FEATURES_IDX,
        "vq_k": VQ_K,
        "fold_user_ids": [list(f) for f in folds],
        "n_total_samples": len(all_rows),
        "accuracies": accs,
        "rows": all_rows,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResultados guardados en: {OUT_PATH}")
    print(f"Total: {(time.time()-t_total)/3600:.2f} horas")


if __name__ == "__main__":
    main()
