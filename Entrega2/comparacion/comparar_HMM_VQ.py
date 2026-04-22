"""
=============================================================================
COMPARACION HMM vs VQ - agreement / ensemble viability
=============================================================================
Entrena HMM (GMMHMM n_mix=2, n_est=7, diag, p=0.6, features=med) y VQ
(1 codebook KMeans 32 centroides por digito, 7D) sobre los MISMOS splits
(N=74, N=47, LOO) y guarda las predicciones por muestra para poder comparar
fallos, acuerdos y cota de oraculo (ensemble maximo).

Escribe resultados en predicciones_<escenario>.json con filas:
    user_id, digit, session, sample_id, y_true, y_pred_hmm, y_pred_vq
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

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")
import logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# -- Paths --
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ENTREGA2_DIR = os.path.dirname(SCRIPT_DIR)
ENTREGA1_DIR = os.path.join(os.path.dirname(ENTREGA2_DIR), "Entrega1")
HMM_DIR = os.path.join(ENTREGA2_DIR, "HMM")
VQ_DIR = os.path.join(ENTREGA2_DIR, "VQ")
EXTRACTOR_DIR = os.path.join(
    REPO_ROOT, "Extractores_adaptados", "Extractores", "Extractor Local"
)
DB_PATH = os.path.join(ENTREGA2_DIR, "e-BioDigit_DB", "e-BioDigit_DB")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "resultados")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, EXTRACTOR_DIR)
sys.path.insert(0, ENTREGA1_DIR)
sys.path.insert(0, HMM_DIR)
sys.path.insert(0, VQ_DIR)

from clasificador_digitos import (
    cargar_base_datos, preprocesar, extraer_features,
    NormalizadorZScore, clasificar_lote, FEATURE_SUBSETS,
)
from clasificador_digitos_v3 import entrenar_gmmhmm_digito
from implementacion_VQ import (
    VQDigitClassifier, preprocess_trace as vq_preprocess_trace,
    compute_local_features as vq_compute_features,
)

_FNAME_RE = re.compile(r"u(\d+)_digit_(\d)_(\d+)\.txt")

# =============================================================================
# CONFIG
# =============================================================================

HMM_CONFIG = {
    "n_mix": 2,
    "n_estados": 7,
    "cov": "diag",
    "prob_autolazo": 0.6,
    "features": "med",
}

# Config para todos los escenarios (compromiso tiempo/calidad)
HMM_FULL = {"n_iter": 60, "n_restarts": 4}
# LOO: aun mas reducido para que 93 folds sean viables
HMM_LOO = {"n_iter": 40, "n_restarts": 2}

VQ_N_CENTROIDS = 32

N_TRAIN_74 = 74
N_TRAIN_47 = 47


# =============================================================================
# UNIFIED SAMPLE LOADING
# =============================================================================

class UnifiedSample:
    """Holds both HMM and VQ feature tensors for a single trace."""
    __slots__ = ("user_id", "digit", "session", "sample_id",
                 "hmm_feats", "vq_feats")

    def __init__(self, user_id, digit, session, sample_id,
                 hmm_feats, vq_feats):
        self.user_id = user_id
        self.digit = digit
        self.session = session
        self.sample_id = sample_id
        self.hmm_feats = hmm_feats
        self.vq_feats = vq_feats


def _parse_sample_id(filepath):
    name = os.path.basename(filepath)
    m = _FNAME_RE.match(name)
    if not m:
        return None
    return int(m.group(3))


def build_unified_dataset():
    """Load DB once and compute HMM + VQ features per sample.

    Returns dict {user_id_int: [UnifiedSample, ...]} and sorted user_ids.
    """
    print("  Cargando base de datos...", flush=True)
    db, user_ids_str = cargar_base_datos(DB_PATH)

    samples_by_user = defaultdict(list)
    indices_med = FEATURE_SUBSETS[HMM_CONFIG["features"]]

    for uid_str in user_ids_str:
        uid = int(uid_str)
        for digit in range(10):
            if digit not in db[uid_str]:
                continue
            for session in [1, 2]:
                if session not in db[uid_str][digit]:
                    continue
                for muestra in db[uid_str][digit][session]:
                    fp = muestra["filepath"]
                    sample_id = _parse_sample_id(fp)
                    if sample_id is None:
                        continue

                    # HMM path
                    x_h, y_h, _ = preprocesar(
                        muestra["x"], muestra["y"], muestra["timestamp"],
                        n_resample=80, suavizar=True)
                    hmm_feats = extraer_features(
                        x_h, y_h, muestra["presion"],
                        indices_features=indices_med)

                    # VQ path (same raw data, different pipeline)
                    x_v, y_v, _ = vq_preprocess_trace(
                        np.array(muestra["x"], dtype=float),
                        np.array(muestra["y"], dtype=float),
                        np.array(muestra["timestamp"], dtype=float))
                    vq_feats = vq_compute_features(
                        x_v, y_v, muestra["presion"])

                    samples_by_user[uid].append(UnifiedSample(
                        uid, digit, session, sample_id,
                        hmm_feats, vq_feats))

    user_ids = sorted(samples_by_user.keys())
    total = sum(len(v) for v in samples_by_user.values())
    print(f"  Usuarios: {len(user_ids)} - Muestras: {total}", flush=True)
    return samples_by_user, user_ids


# =============================================================================
# TRAINING + PREDICTION
# =============================================================================

def _gather_hmm_seqs(samples):
    seqs = [s.hmm_feats for s in samples]
    labels = [s.digit for s in samples]
    return seqs, labels


def train_hmm(train_samples, fit_opts, verbose=False):
    train_seqs, train_labels = _gather_hmm_seqs(train_samples)

    norm = NormalizadorZScore()
    train_norm = norm.ajustar_y_transformar(train_seqs)

    train_por_digito = defaultdict(list)
    for seq, label in zip(train_norm, train_labels):
        train_por_digito[label].append(seq)

    modelos = {}
    for digit in range(10):
        seqs = train_por_digito.get(digit, [])
        if not seqs:
            modelos[digit] = None
            continue
        t0 = time.time()
        model, score = entrenar_gmmhmm_digito(
            seqs,
            n_estados=HMM_CONFIG["n_estados"],
            n_mix=HMM_CONFIG["n_mix"],
            tipo_covarianza=HMM_CONFIG["cov"],
            n_iter=fit_opts["n_iter"],
            n_restarts=fit_opts["n_restarts"],
            prob_autolazo=HMM_CONFIG["prob_autolazo"],
            semilla_base=digit * 100,
        )
        modelos[digit] = model
        if verbose:
            print(f"    digito {digit}: {len(seqs)} seqs, "
                  f"LL={score:.0f} ({time.time()-t0:.0f}s)", flush=True)
    return modelos, norm


def predict_hmm(modelos, norm, test_samples):
    test_seqs = [s.hmm_feats for s in test_samples]
    test_labels = [s.digit for s in test_samples]
    test_norm = norm.transformar(test_seqs)
    preds, _, _ = clasificar_lote(modelos, test_norm, test_labels)
    return list(preds)


def train_vq(train_samples):
    class _S: pass
    proxies = []
    for s in train_samples:
        p = _S()
        p.digit = s.digit
        p.features = s.vq_feats
        proxies.append(p)
    model = VQDigitClassifier(n_centroids=VQ_N_CENTROIDS, random_state=42)
    model.fit_quiet(proxies)
    return model


def predict_vq(model, test_samples):
    preds = []
    for s in test_samples:
        pred, _ = model.predict_one(s.vq_feats)
        preds.append(int(pred))
    return preds


# =============================================================================
# SCENARIO RUNNERS
# =============================================================================

def run_split(samples_by_user, user_ids, n_train, tag, fit_opts):
    out_path = os.path.join(RESULTS_DIR, f"predicciones_{tag}.json")
    if os.path.exists(out_path):
        print(f"  [{tag}] ya existe {out_path}, saltando.", flush=True)
        with open(out_path) as f:
            return json.load(f)

    train_uids = set(user_ids[:n_train])
    test_uids = set(user_ids[n_train:])

    train_samples = [s for uid in train_uids for s in samples_by_user[uid]]
    test_samples = [s for uid in test_uids for s in samples_by_user[uid]]

    print(f"\n  [{tag}] train usuarios={len(train_uids)} "
          f"({len(train_samples)} muestras) "
          f"| test usuarios={len(test_uids)} ({len(test_samples)} muestras)",
          flush=True)

    t0 = time.time()
    print(f"  [{tag}] entrenando HMM "
          f"(n_iter={fit_opts['n_iter']}, n_restarts={fit_opts['n_restarts']})...",
          flush=True)
    hmm_models, hmm_norm = train_hmm(train_samples, fit_opts, verbose=True)
    print(f"  [{tag}] HMM OK ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    print(f"  [{tag}] entrenando VQ...", flush=True)
    vq_model = train_vq(train_samples)
    print(f"  [{tag}] VQ OK ({time.time()-t0:.0f}s)", flush=True)

    hmm_preds = predict_hmm(hmm_models, hmm_norm, test_samples)
    vq_preds = predict_vq(vq_model, test_samples)

    rows = []
    for s, yh, yv in zip(test_samples, hmm_preds, vq_preds):
        rows.append({
            "user_id": s.user_id,
            "digit": s.digit,
            "session": s.session,
            "sample_id": s.sample_id,
            "y_true": s.digit,
            "y_pred_hmm": int(yh),
            "y_pred_vq": int(yv),
        })

    data = {
        "tag": tag,
        "hmm_config": HMM_CONFIG,
        "hmm_fit": fit_opts,
        "vq_n_centroids": VQ_N_CENTROIDS,
        "n_train_users": len(train_uids),
        "n_test_users": len(test_uids),
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [{tag}] guardado: {out_path}", flush=True)
    return data


def run_loo(samples_by_user, user_ids, fit_opts):
    tag = "LOO"
    out_path = os.path.join(RESULTS_DIR, f"predicciones_{tag}.json")
    ckpt_path = os.path.join(RESULTS_DIR, f"checkpoint_{tag}.json")

    rows_by_uid = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            rows_by_uid = {int(k): v for k, v in json.load(f).items()}
        print(f"  [LOO] checkpoint: {len(rows_by_uid)}/{len(user_ids)} "
              f"usuarios ya completados", flush=True)

    for i, test_uid in enumerate(user_ids, 1):
        if test_uid in rows_by_uid:
            continue

        t0 = time.time()
        train_samples = [s for uid in user_ids if uid != test_uid
                         for s in samples_by_user[uid]]
        test_samples = samples_by_user[test_uid]

        hmm_models, hmm_norm = train_hmm(train_samples, fit_opts)
        vq_model = train_vq(train_samples)

        hmm_preds = predict_hmm(hmm_models, hmm_norm, test_samples)
        vq_preds = predict_vq(vq_model, test_samples)

        user_rows = []
        for s, yh, yv in zip(test_samples, hmm_preds, vq_preds):
            user_rows.append({
                "user_id": s.user_id,
                "digit": s.digit,
                "session": s.session,
                "sample_id": s.sample_id,
                "y_true": s.digit,
                "y_pred_hmm": int(yh),
                "y_pred_vq": int(yv),
            })
        rows_by_uid[test_uid] = user_rows

        with open(ckpt_path, "w") as f:
            json.dump(rows_by_uid, f)

        acc_hmm = np.mean([r["y_pred_hmm"] == r["y_true"] for r in user_rows])
        acc_vq = np.mean([r["y_pred_vq"] == r["y_true"] for r in user_rows])
        done = len(rows_by_uid)
        elapsed = time.time() - t0
        print(f"  [LOO {done:2d}/{len(user_ids)}] uid={test_uid} "
              f"HMM={acc_hmm*100:5.1f}% VQ={acc_vq*100:5.1f}% "
              f"({elapsed:.0f}s)", flush=True)

    # Flatten
    all_rows = []
    for uid in user_ids:
        all_rows.extend(rows_by_uid[uid])

    data = {
        "tag": tag,
        "hmm_config": HMM_CONFIG,
        "hmm_fit": fit_opts,
        "vq_n_centroids": VQ_N_CENTROIDS,
        "n_folds": len(user_ids),
        "rows": all_rows,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [LOO] guardado: {out_path}", flush=True)
    return data


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  COMPARACION HMM vs VQ - predicciones por muestra")
    print("=" * 70, flush=True)

    samples_by_user, user_ids = build_unified_dataset()

    scenarios = os.environ.get("SCENARIOS", "N74,N47,LOO").split(",")

    if "N74" in scenarios:
        print("\n[N=74]")
        run_split(samples_by_user, user_ids, N_TRAIN_74, "N74", HMM_FULL)

    if "N47" in scenarios:
        print("\n[N=47]")
        run_split(samples_by_user, user_ids, N_TRAIN_47, "N47", HMM_FULL)

    if "LOO" in scenarios:
        print("\n[LOO]")
        run_loo(samples_by_user, user_ids, HMM_LOO)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} minutos")


if __name__ == "__main__":
    main()
