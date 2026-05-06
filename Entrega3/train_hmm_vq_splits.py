#!/usr/bin/env python3
"""
Entrena GMMHMM (n_iter=100, n_restarts=10) + VQ K=128 para los escenarios
N=74 y N=47. Guarda en ensemble_N74_full.json / ensemble_N47_full.json
con el mismo formato que ensemble_paralelo.py para ser usados en las DET.

Checkpoint por escenario — se puede interrumpir y retomar.
"""

import os, re, sys, json, time, warnings
from collections import defaultdict

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
import logging; logging.getLogger("hmmlearn").setLevel(logging.ERROR)

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT     = os.path.dirname(SCRIPT_DIR)
ENTREGA1_DIR  = os.path.join(REPO_ROOT, "Entrega1")
ENTREGA2_DIR  = os.path.join(REPO_ROOT, "Entrega2")
HMM_DIR       = os.path.join(ENTREGA2_DIR, "HMM")
VQ_DIR        = os.path.join(ENTREGA2_DIR, "VQ")
EXTRACTOR_DIR = os.path.join(REPO_ROOT, "Extractores_adaptados",
                              "Extractores", "Extractor Local")
DB_PATH       = os.path.join(ENTREGA2_DIR, "e-BioDigit_DB", "e-BioDigit_DB")
RESULTS_DIR   = os.path.join(SCRIPT_DIR, "Paralel", "resultados")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, EXTRACTOR_DIR)
sys.path.insert(0, ENTREGA1_DIR)
sys.path.insert(0, HMM_DIR)
sys.path.insert(0, VQ_DIR)

from clasificador_digitos import (
    cargar_base_datos, preprocesar, extraer_features,
    NormalizadorZScore, FEATURE_SUBSETS,
)
from clasificador_digitos_v3 import entrenar_gmmhmm_digito
from busqueda_VQ import preprocess_trace as vq_preprocess, compute_full_features

HMM_CFG = dict(n_mix=2, n_estados=7, cov="diag", prob_autolazo=0.6, features="med")
HMM_FIT = dict(n_iter=100, n_restarts=10)
VQ_FEATURES_IDX = [0, 1, 19, 20, 10]
VQ_K = 128
_FNAME_RE = re.compile(r"u(\d+)_digit_(\d)_(\d+)\.txt")

SCENARIOS = {
    "N74": {"n_train": 74, "out": os.path.join(RESULTS_DIR, "ensemble_N74_full.json")},
    "N47": {"n_train": 47, "out": os.path.join(RESULTS_DIR, "ensemble_N47_full.json")},
}


class Sample:
    __slots__ = ("user_id","digit","session","sample_id","hmm_feats","vq_feats")
    def __init__(self, uid, d, s, sid, hf, vf):
        self.user_id=uid; self.digit=d; self.session=s
        self.sample_id=sid; self.hmm_feats=hf; self.vq_feats=vf


def build_dataset():
    print("Cargando base de datos...", flush=True)
    db, uids = cargar_base_datos(DB_PATH)
    idx_hmm  = FEATURE_SUBSETS[HMM_CFG["features"]]
    by_user  = defaultdict(list)
    for uid_str in uids:
        uid = int(uid_str)
        for digit in range(10):
            if digit not in db[uid_str]: continue
            for session in (1, 2):
                if session not in db[uid_str][digit]: continue
                for m in db[uid_str][digit][session]:
                    mid = _FNAME_RE.match(os.path.basename(m["filepath"]))
                    sid = int(mid.group(3)) if mid else None
                    if sid is None: continue
                    x_h, y_h, _ = preprocesar(m["x"], m["y"], m["timestamp"],
                                               n_resample=80, suavizar=True)
                    hf = extraer_features(x_h, y_h, m["presion"],
                                          indices_features=idx_hmm)
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


def train_hmm(train_samples):
    seqs = [s.hmm_feats for s in train_samples]
    norm = NormalizadorZScore()
    seqs_n = norm.ajustar_y_transformar(seqs)
    by_d = defaultdict(list)
    for sq, s in zip(seqs_n, train_samples):
        by_d[s.digit].append(sq)
    modelos = {}
    t0 = time.time()
    for d in range(10):
        if not by_d.get(d): modelos[d] = None; continue
        model, score = entrenar_gmmhmm_digito(
            by_d[d], n_estados=HMM_CFG["n_estados"], n_mix=HMM_CFG["n_mix"],
            tipo_covarianza=HMM_CFG["cov"], n_iter=HMM_FIT["n_iter"],
            n_restarts=HMM_FIT["n_restarts"], prob_autolazo=HMM_CFG["prob_autolazo"],
            semilla_base=d * 100)
        modelos[d] = model
        print(f"    dígito {d}: LL={score:.0f}  ({time.time()-t0:.0f}s acum.)",
              flush=True)
    return modelos, norm


def hmm_lls(modelos, norm, samples):
    seqs_n = norm.transformar([s.hmm_feats for s in samples])
    out = np.full((len(samples), 10), -np.inf)
    for i, sq in enumerate(seqs_n):
        for d in range(10):
            mdl = modelos.get(d)
            if mdl is None: continue
            try:
                sc = mdl.score(sq)
                if np.isfinite(sc) and abs(sc) < 1e6:
                    out[i, d] = sc
            except Exception: pass
    finite = np.isfinite(out)
    out[~finite] = (float(np.min(out[finite])) - 1e3) if finite.any() else -1e6
    return out


def train_vq(train_samples):
    by_d = defaultdict(list)
    for s in train_samples: by_d[s.digit].append(s.vq_feats)
    cbs = {}
    for d in range(10):
        if not by_d.get(d): cbs[d] = None; continue
        X  = np.vstack(by_d[d])
        km = MiniBatchKMeans(n_clusters=min(VQ_K, len(X)), init="k-means++",
                             n_init=10, max_iter=300,
                             batch_size=min(1024, len(X)), random_state=42)
        km.fit(X); cbs[d] = km.cluster_centers_
    return cbs


def vq_dists(cbs, samples):
    out = np.full((len(samples), 10), np.inf)
    for i, s in enumerate(samples):
        for d, cb in cbs.items():
            if cb is None: continue
            d2 = np.sum((s.vq_feats[:, None, :] - cb[None, :, :]) ** 2, axis=2)
            out[i, d] = float(np.mean(np.min(d2, axis=1)))
    return out


def _norm_softmax(sc):
    mu = sc.mean(1, keepdims=True); sd = sc.std(1, keepdims=True) + 1e-12
    s  = (sc - mu) / sd - np.max((sc - mu) / sd, 1, keepdims=True)
    e  = np.exp(s); return e / e.sum(1, keepdims=True)


def fusion_scores(lls, dists):
    hp = _norm_softmax(lls); vp = _norm_softmax(-dists)
    ch = hp.max(1); cv = vp.max(1)
    mh = np.sort(hp, 1)[:, -1] - np.sort(hp, 1)[:, -2]
    mv = np.sort(vp, 1)[:, -1] - np.sort(vp, 1)[:, -2]
    soft = 0.5 * hp + 0.5 * vp
    cw   = (ch[:, None]*hp + cv[:, None]*vp) / (ch[:, None]+cv[:, None]+1e-12)
    mw   = (mh[:, None]*hp + mv[:, None]*vp) / (mh[:, None]+mv[:, None]+1e-12)
    agree = np.where(((hp.argmax(1) == vp.argmax(1)) | (ch >= cv))[:, None], hp, vp)
    return ({"hmm":hp,"vq":vp,"agreement":agree,"soft":soft,
             "conf_weighted":cw,"margin_weighted":mw},
            {"conf_hmm":ch,"conf_vq":cv,"margin_hmm":mh,"margin_vq":mv})


def evaluate_split(tag, n_train, by_user, user_ids):
    out_path  = SCENARIOS[tag]["out"]
    ckpt_path = os.path.join(SCRIPT_DIR, f"splits_ckpt_{tag}.json")

    if os.path.exists(out_path):
        print(f"\n[{tag}] Ya existe {out_path}, saltando.", flush=True)
        return

    print(f"\n{'='*60}\n[{tag}] n_iter={HMM_FIT['n_iter']} "
          f"n_restarts={HMM_FIT['n_restarts']}\n{'='*60}", flush=True)

    train_uids = set(user_ids[:n_train])
    test_uids  = set(user_ids[n_train:])
    train_s    = [s for u in train_uids for s in by_user[u]]
    test_s     = [s for u in test_uids  for s in by_user[u]]
    print(f"  train={len(train_uids)} usuarios ({len(train_s)} muestras)  "
          f"test={len(test_uids)} usuarios ({len(test_s)} muestras)", flush=True)

    # Checkpoint: si ya hay resultados parciales de HMM, cargar
    ckpt = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f: ckpt = json.load(f)

    if "hmm_lls" not in ckpt:
        print(f"  Entrenando GMMHMM...", flush=True)
        t0 = time.time()
        modelos, norm = train_hmm(train_s)
        lls_arr = hmm_lls(modelos, norm, test_s)
        ckpt["hmm_lls"] = lls_arr.tolist()
        with open(ckpt_path, "w") as f: json.dump(ckpt, f)
        print(f"  HMM listo ({time.time()-t0:.0f}s)", flush=True)
    else:
        print("  HMM cargado desde checkpoint.", flush=True)
        lls_arr = np.array(ckpt["hmm_lls"])

    print(f"  Entrenando VQ...", flush=True)
    vq_cbs   = train_vq(train_s)
    dists_arr = vq_dists(vq_cbs, test_s)

    scores, meta = fusion_scores(lls_arr, dists_arr)
    preds = {k: np.argmax(v, 1) for k, v in scores.items()}
    y = np.array([s.digit for s in test_s])
    accs = {k: float(accuracy_score(y, preds[k])) for k in preds}
    accs["oracle"] = float(np.mean(
        (preds["hmm"] == y) | (preds["vq"] == y)))

    print(f"\n  Accuracies [{tag}]:")
    for k in ("hmm","vq","margin_weighted","oracle"):
        print(f"    {k:18s}: {accs[k]*100:.2f}%")

    rows = []
    for i, s in enumerate(test_s):
        rows.append({
            "user_id": s.user_id, "digit": s.digit,
            "session": s.session, "sample_id": s.sample_id,
            "y_true": s.digit,
            "hmm_lls":  [float(x) for x in lls_arr[i]],
            "vq_dists": [float(x) for x in dists_arr[i]],
            "conf_hmm":   float(meta["conf_hmm"][i]),
            "conf_vq":    float(meta["conf_vq"][i]),
            "margin_hmm": float(meta["margin_hmm"][i]),
            "margin_vq":  float(meta["margin_vq"][i]),
            **{f"pred_{k}":   int(preds[k][i])              for k in preds},
            **{f"scores_{k}": [float(x) for x in scores[k][i]] for k in scores},
        })

    with open(out_path, "w") as f:
        json.dump({"tag": tag, "hmm_config": HMM_CFG, "hmm_fit": HMM_FIT,
                   "vq_k": VQ_K, "accuracies": accs, "rows": rows}, f, indent=2)
    print(f"  Guardado: {out_path}", flush=True)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def main():
    t0 = time.time()
    print("="*60)
    print(f"  TRAIN HMM+VQ  N74+N47  "
          f"(n_iter={HMM_FIT['n_iter']}, n_restarts={HMM_FIT['n_restarts']})")
    print("="*60, flush=True)

    by_user, user_ids = build_dataset()

    for tag, cfg in SCENARIOS.items():
        evaluate_split(tag, cfg["n_train"], by_user, user_ids)

    print(f"\nTotal: {(time.time()-t0)/3600:.2f} horas")


if __name__ == "__main__":
    main()
