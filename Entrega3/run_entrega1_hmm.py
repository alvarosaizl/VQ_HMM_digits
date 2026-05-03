"""
=============================================================================
RE-EJECUCION HMM E1 (Entrega 1) GUARDANDO SCORES POR MUESTRA
=============================================================================
Entrega 1 reporta accuracies pero no guarda los scores 10D necesarios para
construir curvas DET. Este script re-entrena el modelo final de Entrega 1
(GaussianHMM, n_mix=1, 7 estados, p_autolazo=0.6, features=`med` 12D,
covarianza diagonal) sobre N=74, N=47 y LOO, y persiste por cada muestra
de test el vector 10D de log-verosimilitudes.

Configuracion HMM aligerada respecto a Entrega 1 original (n_iter=100,
n_restarts=5) para hacer viable el LOO en una sola noche:

  - HMM_FIT (N=74, N=47): n_iter=40, n_restarts=2
  - HMM_FIT_LOO:          n_iter=15, n_restarts=1 (93 folds)

La accuracy puede caer respecto a la reportada en Entrega 1 pero el
ranking de scores (que es lo que importa para DET y EER) preserva la
forma cualitativa.
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
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")
import logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
ENTREGA1_DIR = os.path.join(REPO_ROOT, "Entrega1")
ENTREGA2_DIR = os.path.join(REPO_ROOT, "Entrega2")
HMM_DIR = os.path.join(ENTREGA2_DIR, "HMM")
EXTRACTOR_DIR = os.path.join(
    REPO_ROOT, "Extractores_adaptados", "Extractores", "Extractor Local"
)
DB_PATH = os.path.join(ENTREGA2_DIR, "e-BioDigit_DB", "e-BioDigit_DB")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "Entrega1_rerun")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, EXTRACTOR_DIR)
sys.path.insert(0, ENTREGA1_DIR)
sys.path.insert(0, HMM_DIR)

from clasificador_digitos import (  # noqa: E402
    cargar_base_datos, preprocesar, extraer_features,
    NormalizadorZScore, FEATURE_SUBSETS,
)
from clasificador_digitos_v3 import entrenar_gmmhmm_digito  # noqa: E402

# Configuracion E1 (Entrega 1 final)
HMM_CFG = dict(
    n_mix=1,             # GaussianHMM
    n_estados=7,
    cov="diag",
    prob_autolazo=0.6,
    features="med",
)
HMM_FIT = dict(n_iter=40, n_restarts=2)
HMM_FIT_LOO = dict(n_iter=15, n_restarts=1)

N_TRAIN_74 = 74
N_TRAIN_47 = 47

_FNAME_RE = re.compile(r"u(\d+)_digit_(\d)_(\d+)\.txt")


# =============================================================================
# Dataset
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
# HMM helpers (n_mix=1 -> GaussianHMM)
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
            n_mix=HMM_CFG["n_mix"],            # 1 -> GaussianHMM
            tipo_covarianza=HMM_CFG["cov"],
            n_iter=fit_opts["n_iter"],
            n_restarts=fit_opts["n_restarts"],
            prob_autolazo=HMM_CFG["prob_autolazo"],
            semilla_base=d * 100,
        )
        modelos[d] = m
    return modelos, norm


def hmm_log_likelihoods(modelos, norm, samples):
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


def softmax_norm(scores):
    """Z-score per sample then softmax. Devuelve probabilidades 10D."""
    mu = np.mean(scores, axis=1, keepdims=True)
    sd = np.std(scores, axis=1, keepdims=True) + 1e-12
    z = (scores - mu) / sd
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


# =============================================================================
# Splits
# =============================================================================

def evaluate_split(by_user, user_ids, n_train, tag):
    out_path = os.path.join(RESULTS_DIR, f"e1_{tag}.json")
    train_uids = list(user_ids[:n_train])
    test_uids = list(user_ids[n_train:])
    train_samples = [s for u in train_uids for s in by_user[u]]
    test_samples = [s for u in test_uids for s in by_user[u]]
    print(f"\n[{tag}] train={len(train_samples)} test={len(test_samples)}",
          flush=True)

    t0 = time.time()
    print(f"  Entrenando GaussianHMM (n_iter={HMM_FIT['n_iter']}, "
          f"n_restarts={HMM_FIT['n_restarts']})...", flush=True)
    modelos, norm = train_hmm(train_samples, HMM_FIT)
    print(f"  HMM listo ({time.time()-t0:.0f}s)", flush=True)

    lls = hmm_log_likelihoods(modelos, norm, test_samples)
    probs = softmax_norm(lls)
    preds = np.argmax(lls, axis=1)
    y_true = np.array([s.digit for s in test_samples])
    acc = float(accuracy_score(y_true, preds))
    print(f"  Accuracy: {acc*100:.2f}%", flush=True)

    rows = []
    for i, s in enumerate(test_samples):
        rows.append({
            "user_id": s.user_id,
            "digit": s.digit,
            "session": s.session,
            "sample_id": s.sample_id,
            "y_true": s.digit,
            "hmm_lls": [float(x) for x in lls[i]],
            "scores_e1_hmm": [float(x) for x in probs[i]],
            "pred_e1_hmm": int(preds[i]),
        })
    with open(out_path, "w") as f:
        json.dump({
            "tag": tag, "model": "E1_GaussianHMM",
            "hmm_config": HMM_CFG, "hmm_fit": HMM_FIT,
            "accuracy": acc, "rows": rows,
        }, f, indent=2)
    print(f"  Guardado: {out_path}", flush=True)
    return acc


def evaluate_loo(by_user, user_ids):
    tag = "LOO"
    out_path = os.path.join(RESULTS_DIR, f"e1_{tag}.json")
    ckpt_path = os.path.join(RESULTS_DIR, f"checkpoint_{tag}.json")

    rows_by_uid = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            rows_by_uid = {int(k): v for k, v in json.load(f).items()}
        print(f"  [LOO] checkpoint: {len(rows_by_uid)}/{len(user_ids)}",
              flush=True)

    for i, test_uid in enumerate(user_ids, 1):
        if test_uid in rows_by_uid:
            continue
        t0 = time.time()
        train_samples = [s for u in user_ids if u != test_uid
                         for s in by_user[u]]
        test_samples = by_user[test_uid]
        modelos, norm = train_hmm(train_samples, HMM_FIT_LOO)
        lls = hmm_log_likelihoods(modelos, norm, test_samples)
        probs = softmax_norm(lls)
        preds = np.argmax(lls, axis=1)
        y_test = np.array([s.digit for s in test_samples])
        user_rows = []
        for j, s in enumerate(test_samples):
            user_rows.append({
                "user_id": s.user_id,
                "digit": s.digit,
                "session": s.session,
                "sample_id": s.sample_id,
                "y_true": s.digit,
                "hmm_lls": [float(x) for x in lls[j]],
                "scores_e1_hmm": [float(x) for x in probs[j]],
                "pred_e1_hmm": int(preds[j]),
            })
        rows_by_uid[test_uid] = user_rows
        with open(ckpt_path, "w") as f:
            json.dump(rows_by_uid, f)
        acc = float(np.mean(preds == y_test))
        print(f"  [LOO {len(rows_by_uid):2d}/{len(user_ids)}] "
              f"uid={test_uid} acc={acc*100:5.1f}% "
              f"({time.time()-t0:.0f}s)", flush=True)

    all_rows = []
    for uid in user_ids:
        all_rows.extend(rows_by_uid[uid])
    y_true = np.array([r["y_true"] for r in all_rows])
    preds = np.array([r["pred_e1_hmm"] for r in all_rows])
    acc = float(accuracy_score(y_true, preds))
    print(f"\n[LOO] accuracy global = {acc*100:.2f}%")
    with open(out_path, "w") as f:
        json.dump({
            "tag": tag, "model": "E1_GaussianHMM",
            "hmm_config": HMM_CFG, "hmm_fit": HMM_FIT_LOO,
            "n_folds": len(user_ids),
            "accuracy": acc, "rows": all_rows,
        }, f, indent=2)
    return acc


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  RE-EJECUCION HMM E1 (GaussianHMM, Entrega 1) - guardando scores")
    print("=" * 70, flush=True)

    by_user, user_ids = build_dataset()
    scenarios = os.environ.get("SCENARIOS", "N74,N47,LOO").split(",")

    summary = {}
    if "N74" in scenarios:
        summary["N74"] = evaluate_split(by_user, user_ids, N_TRAIN_74, "N74")
    if "N47" in scenarios:
        summary["N47"] = evaluate_split(by_user, user_ids, N_TRAIN_47, "N47")
    if "LOO" in scenarios:
        summary["LOO"] = evaluate_loo(by_user, user_ids)

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
