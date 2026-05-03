"""
=============================================================================
ENSEMBLE PARALELO HMM + VQ
=============================================================================
Ambos modelos clasifican la misma muestra de forma independiente. Se fusionan
las decisiones combinando prediccion y confianza por clase:

  - HMM (GMMHMM, n_mix=2, 7 estados, p_autolazo=0.6, features=med [12D])
        produce un vector de 10 log-verosimilitudes (mayor = mas probable).
  - VQ  (MiniBatchKMeans, k=128, features=pos_ang_curv [5D])
        produce un vector de 10 distorsiones medias (menor = mas probable).

Reglas de fusion evaluadas:

  1. agreement     : si HMM y VQ coinciden, esa es la prediccion. Si no,
                     se elige la del modelo con mayor confianza top-1.
  2. soft          : promedio simple de probabilidades softmax.
  3. conf_weighted : combinacion ponderada por confianza top-1 de cada modelo.
  4. margin_weighted: combinacion ponderada por el margen top1-top2.
  5. oracle        : cota superior del ensemble (si CUALQUIER modelo acierta).

Justificacion: cuando un modelo falla, el otro suele acertar.  Si VQ dice 4
(distorsion baja) y HMM dice 9 pero su LL para 4 es casi tan alta como para
9, el modelo HMM esta dudando -> conviene confiar en VQ.

Salida:
  resultados/ensemble_<TAG>.json  - predicciones y scores por muestra
  resultados/summary.json         - tabla de accuracies por escenario y regla
  plots/cm_<TAG>_<regla>.png      - confusiones de cada regla
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
VQ_DIR = os.path.join(ENTREGA2_DIR, "VQ")
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
sys.path.insert(0, VQ_DIR)

from clasificador_digitos import (  # noqa: E402
    cargar_base_datos, preprocesar, extraer_features,
    NormalizadorZScore, FEATURE_SUBSETS,
)
from clasificador_digitos_v3 import entrenar_gmmhmm_digito  # noqa: E402
from busqueda_VQ import (  # noqa: E402
    preprocess_trace as vq_preprocess,
    compute_full_features,
)


# =============================================================================
# Configuracion (mejores modelos segun informe E2)
# =============================================================================

HMM_CFG = dict(
    n_mix=2,
    n_estados=7,
    cov="diag",
    prob_autolazo=0.6,
    features="med",       # 12D: dx, dy, sin, cos, dtheta, v, rho, a, dv, lewi, x, y
)
HMM_FIT = dict(n_iter=40, n_restarts=2)       # N=74 y N=47
HMM_FIT_LOO = dict(n_iter=15, n_restarts=1)   # 93 folds: agresivo

VQ_FEATURE_NAME = "pos_ang_curv"
VQ_FEATURES_IDX = [0, 1, 19, 20, 10]  # x, y, sin, cos, dtheta
VQ_K = 128
VQ_RANDOM_STATE = 42

N_TRAIN_74 = 74
N_TRAIN_47 = 47

_FNAME_RE = re.compile(r"u(\d+)_digit_(\d)_(\d+)\.txt")


# =============================================================================
# Dataset unificado
# =============================================================================

class Sample:
    __slots__ = (
        "user_id", "digit", "session", "sample_id", "hmm_feats", "vq_feats",
    )

    def __init__(self, uid, d, s, sid, hf, vf):
        self.user_id = uid
        self.digit = d
        self.session = s
        self.sample_id = sid
        self.hmm_feats = hf
        self.vq_feats = vf


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

                    # Pipeline HMM (suavizado + remuestreo a 80 puntos)
                    x_h, y_h, _ = preprocesar(
                        muestra["x"], muestra["y"], muestra["timestamp"],
                        n_resample=80, suavizar=True,
                    )
                    hmm_feats = extraer_features(
                        x_h, y_h, muestra["presion"],
                        indices_features=indices_hmm,
                    )

                    # Pipeline VQ (sin suavizado/remuestreo)
                    x_v, y_v, _ = vq_preprocess(
                        np.array(muestra["x"], dtype=float),
                        np.array(muestra["y"], dtype=float),
                        np.array(muestra["timestamp"], dtype=float),
                    )
                    full = compute_full_features(x_v, y_v, muestra["presion"])
                    vq_feats = full[:, VQ_FEATURES_IDX]

                    by_user[uid].append(Sample(
                        uid, digit, session, sid, hmm_feats, vq_feats))

    user_ids = sorted(by_user.keys())
    total = sum(len(v) for v in by_user.values())
    print(f"  {len(user_ids)} usuarios, {total} muestras", flush=True)
    return by_user, user_ids


# =============================================================================
# Entrenamiento e inferencia
# =============================================================================

def train_hmm(train_samples, fit_opts=HMM_FIT, verbose=False):
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
        t0 = time.time()
        model, score = entrenar_gmmhmm_digito(
            by_d[d],
            n_estados=HMM_CFG["n_estados"],
            n_mix=HMM_CFG["n_mix"],
            tipo_covarianza=HMM_CFG["cov"],
            n_iter=fit_opts["n_iter"],
            n_restarts=fit_opts["n_restarts"],
            prob_autolazo=HMM_CFG["prob_autolazo"],
            semilla_base=d * 100,
        )
        modelos[d] = model
        if verbose:
            print(f"    digito {d}: {len(by_d[d])} seqs, "
                  f"LL={score:.0f} ({time.time()-t0:.0f}s)", flush=True)
    return modelos, norm


def hmm_log_likelihoods(modelos, norm, samples):
    """Devuelve matriz (N, 10) con log-verosimilitudes por digito.

    Si un modelo HMM fallo en convergir (None) o un score lanza error, la
    LL queda como -inf y se reemplaza por un suelo finito (mejor LL global
    -1e3) para no corromper normalizaciones por z-score / softmax.
    """
    seqs_n = norm.transformar([s.hmm_feats for s in samples])
    out = np.full((len(samples), 10), -np.inf, dtype=float)
    # Modelos con scores patologicos (|LL| > 1e6 para una sola secuencia)
    # son modelos colapsados: tratarlos como fallidos para evitar que
    # contaminen normalizaciones por z-score / softmax.
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


def train_vq(train_samples):
    by_d = defaultdict(list)
    for s in train_samples:
        by_d[s.digit].append(s.vq_feats)
    codebooks = {}
    for d in range(10):
        if not by_d.get(d):
            codebooks[d] = None
            continue
        X = np.vstack(by_d[d])
        k = min(VQ_K, len(X))
        km = MiniBatchKMeans(
            n_clusters=k, init="k-means++", n_init=10,
            max_iter=300, batch_size=min(1024, len(X)),
            random_state=VQ_RANDOM_STATE,
        )
        km.fit(X)
        codebooks[d] = km.cluster_centers_
    return codebooks


def vq_distortions(codebooks, samples):
    """Distorsion media (T x dist^2 al centroide mas cercano) por digito."""
    out = np.full((len(samples), 10), np.inf, dtype=float)
    for i, s in enumerate(samples):
        F = s.vq_feats
        for d in range(10):
            cb = codebooks.get(d)
            if cb is None:
                continue
            dists = np.sum((F[:, None, :] - cb[None, :, :]) ** 2, axis=2)
            out[i, d] = float(np.mean(np.min(dists, axis=1)))
    return out


# =============================================================================
# Reglas de fusion
# =============================================================================

def _softmax(scores, axis=-1):
    s = scores - np.max(scores, axis=axis, keepdims=True)
    e = np.exp(s)
    return e / np.sum(e, axis=axis, keepdims=True)


def _margin(probs):
    s = np.sort(probs, axis=-1)
    return s[..., -1] - s[..., -2]


def fusion_scores(hmm_lls, vq_dists):
    """Devuelve scores 10D por muestra para HMM, VQ y reglas de fusion.

    Returns:
        scores: dict {method: (N,10) array de "scores" donde mayor=mejor}.
                Para hmm y vq son las LL/(-dist) brutas (utiles para AUC).
                Para fusion son probabilidades suaves.
        meta:   dict con probs y confidences/margenes auxiliares por muestra.
    """
    # HMM y VQ: las escalas absolutas (LL en miles, dist^2 en O(1)) son muy
    # distintas y un softmax directo sobre LLs satura a one-hot, anulando la
    # ponderacion por confianza. Normalizamos AMBOS scores por desviacion
    # intra-muestra antes de softmax para tener probabilidades comparables.
    def _norm_softmax(scores):
        mu = np.mean(scores, axis=1, keepdims=True)
        sd = np.std(scores, axis=1, keepdims=True) + 1e-12
        return _softmax((scores - mu) / sd, axis=1)

    hmm_probs = _norm_softmax(hmm_lls)
    vq_probs = _norm_softmax(-vq_dists)

    conf_hmm = np.max(hmm_probs, axis=1)
    conf_vq = np.max(vq_probs, axis=1)
    margin_hmm = _margin(hmm_probs)
    margin_vq = _margin(vq_probs)

    # Soft (promedio simple)
    soft_probs = 0.5 * hmm_probs + 0.5 * vq_probs

    # Conf-weighted (ponderado por confianza top-1)
    w_h = conf_hmm[:, None]
    w_v = conf_vq[:, None]
    cw = w_h * hmm_probs + w_v * vq_probs
    cw = cw / (np.sum(cw, axis=1, keepdims=True) + 1e-12)

    # Margin-weighted
    w_h_m = margin_hmm[:, None]
    w_v_m = margin_vq[:, None]
    mw = w_h_m * hmm_probs + w_v_m * vq_probs
    mw = mw / (np.sum(mw, axis=1, keepdims=True) + 1e-12)

    # Agreement: si coinciden, usa hmm_probs; si no, usa el modelo con mayor
    # confianza top-1. Score 10D = probs del modelo elegido (per-sample).
    preds_hmm = np.argmax(hmm_lls, axis=1)
    preds_vq = np.argmin(vq_dists, axis=1)
    agree_mask = preds_hmm == preds_vq
    pick_hmm = (conf_hmm >= conf_vq) | agree_mask
    agree_probs = np.where(pick_hmm[:, None], hmm_probs, vq_probs)

    scores = {
        "hmm": hmm_probs,
        "vq": vq_probs,
        "agreement": agree_probs,
        "soft": soft_probs,
        "conf_weighted": cw,
        "margin_weighted": mw,
    }
    meta = {
        "conf_hmm": conf_hmm,
        "conf_vq": conf_vq,
        "margin_hmm": margin_hmm,
        "margin_vq": margin_vq,
    }
    return scores, meta


def fusion_predictions(hmm_lls, vq_dists):
    scores, meta = fusion_scores(hmm_lls, vq_dists)
    preds = {k: np.argmax(v, axis=1) for k, v in scores.items()}
    return preds, scores, meta


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


# =============================================================================
# Escenarios
# =============================================================================

def evaluate_split(by_user, user_ids, n_train, tag):
    train_uids = set(user_ids[:n_train])
    test_uids = set(user_ids[n_train:])
    train_samples = [s for u in train_uids for s in by_user[u]]
    test_samples = [s for u in test_uids for s in by_user[u]]
    print(f"\n[{tag}] train usuarios={len(train_uids)} ({len(train_samples)} muestras) "
          f"| test usuarios={len(test_uids)} ({len(test_samples)} muestras)",
          flush=True)

    print(f"  Entrenando HMM (n_iter={HMM_FIT['n_iter']}, "
          f"n_restarts={HMM_FIT['n_restarts']})...", flush=True)
    t0 = time.time()
    hmm_models, hmm_norm = train_hmm(train_samples, verbose=True)
    print(f"  HMM listo ({time.time()-t0:.0f}s)", flush=True)

    print(f"  Entrenando VQ (MBKMeans k={VQ_K}, features={VQ_FEATURE_NAME})...",
          flush=True)
    t0 = time.time()
    codebooks = train_vq(train_samples)
    print(f"  VQ listo ({time.time()-t0:.0f}s)", flush=True)

    print("  Inferencia sobre test...", flush=True)
    t0 = time.time()
    hmm_lls = hmm_log_likelihoods(hmm_models, hmm_norm, test_samples)
    vq_dists = vq_distortions(codebooks, test_samples)
    print(f"  Inferencia lista ({time.time()-t0:.0f}s)", flush=True)

    y_true = np.array([s.digit for s in test_samples])
    preds, scores, meta = fusion_predictions(hmm_lls, vq_dists)

    accuracies = {k: float(accuracy_score(y_true, p)) for k, p in preds.items()}
    oracle = (preds["hmm"] == y_true) | (preds["vq"] == y_true)
    accuracies["oracle"] = float(np.mean(oracle))
    accuracies["both_wrong"] = float(np.mean(
        (preds["hmm"] != y_true) & (preds["vq"] != y_true)))
    accuracies["only_hmm_right"] = float(np.mean(
        (preds["hmm"] == y_true) & (preds["vq"] != y_true)))
    accuracies["only_vq_right"] = float(np.mean(
        (preds["vq"] == y_true) & (preds["hmm"] != y_true)))

    print(f"\n[{tag}] Accuracies:")
    for k in ("hmm", "vq", "agreement", "soft", "conf_weighted",
              "margin_weighted", "oracle"):
        print(f"    {k:18s}: {accuracies[k]*100:6.2f}%")
    print(f"    fallos solapados (both wrong): "
          f"{accuracies['both_wrong']*100:.2f}%")

    # Confusiones
    for method in ("hmm", "vq", "agreement", "soft",
                   "conf_weighted", "margin_weighted"):
        cm = confusion_matrix(y_true, preds[method], labels=list(range(10)))
        plot_confusion(
            cm, f"{tag} - {method} ({accuracies[method]*100:.2f}%)",
            os.path.join(PLOTS_DIR, f"cm_{tag}_{method}.png"),
        )

    # Persistencia por muestra (incluye scores 10D por metodo para AUC/EER)
    rows = []
    for i, s in enumerate(test_samples):
        rows.append({
            "user_id": s.user_id,
            "digit": s.digit,
            "session": s.session,
            "sample_id": s.sample_id,
            "y_true": s.digit,
            "hmm_lls": [float(x) for x in hmm_lls[i]],
            "vq_dists": [float(x) for x in vq_dists[i]],
            "conf_hmm": float(meta["conf_hmm"][i]),
            "conf_vq": float(meta["conf_vq"][i]),
            "margin_hmm": float(meta["margin_hmm"][i]),
            "margin_vq": float(meta["margin_vq"][i]),
            "pred_hmm": int(preds["hmm"][i]),
            "pred_vq": int(preds["vq"][i]),
            "pred_agreement": int(preds["agreement"][i]),
            "pred_soft": int(preds["soft"][i]),
            "pred_conf_weighted": int(preds["conf_weighted"][i]),
            "pred_margin_weighted": int(preds["margin_weighted"][i]),
            # Scores 10D por metodo (probabilidades) para AUC/EER one-vs-rest
            "scores_hmm": [float(x) for x in scores["hmm"][i]],
            "scores_vq": [float(x) for x in scores["vq"][i]],
            "scores_agreement": [float(x) for x in scores["agreement"][i]],
            "scores_soft": [float(x) for x in scores["soft"][i]],
            "scores_conf_weighted": [float(x) for x in scores["conf_weighted"][i]],
            "scores_margin_weighted": [float(x) for x in scores["margin_weighted"][i]],
        })

    out_path = os.path.join(RESULTS_DIR, f"ensemble_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({
            "tag": tag,
            "hmm_config": HMM_CFG,
            "hmm_fit": HMM_FIT,
            "vq_features": VQ_FEATURE_NAME,
            "vq_features_idx": VQ_FEATURES_IDX,
            "vq_k": VQ_K,
            "n_train_users": len(train_uids),
            "n_test_users": len(test_uids),
            "accuracies": accuracies,
            "rows": rows,
        }, f, indent=2)
    print(f"  Guardado: {out_path}", flush=True)
    return accuracies


# =============================================================================
# Leave-One-User-Out
# =============================================================================

def evaluate_loo(by_user, user_ids):
    tag = "LOO"
    out_path = os.path.join(RESULTS_DIR, f"ensemble_{tag}.json")
    ckpt_path = os.path.join(RESULTS_DIR, f"checkpoint_{tag}.json")

    rows_by_uid = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            rows_by_uid = {int(k): v for k, v in json.load(f).items()}
        print(f"  [LOO] checkpoint: {len(rows_by_uid)}/{len(user_ids)} "
              f"usuarios completados", flush=True)

    for i, test_uid in enumerate(user_ids, 1):
        if test_uid in rows_by_uid:
            continue
        t0 = time.time()
        train_samples = [s for u in user_ids if u != test_uid
                         for s in by_user[u]]
        test_samples = by_user[test_uid]

        hmm_models, hmm_norm = train_hmm(train_samples, fit_opts=HMM_FIT_LOO)
        codebooks = train_vq(train_samples)

        hmm_lls = hmm_log_likelihoods(hmm_models, hmm_norm, test_samples)
        vq_dists = vq_distortions(codebooks, test_samples)

        preds, scores, meta = fusion_predictions(hmm_lls, vq_dists)

        user_rows = []
        for j, s in enumerate(test_samples):
            user_rows.append({
                "user_id": s.user_id,
                "digit": s.digit,
                "session": s.session,
                "sample_id": s.sample_id,
                "y_true": s.digit,
                "hmm_lls": [float(x) for x in hmm_lls[j]],
                "vq_dists": [float(x) for x in vq_dists[j]],
                "conf_hmm": float(meta["conf_hmm"][j]),
                "conf_vq": float(meta["conf_vq"][j]),
                "margin_hmm": float(meta["margin_hmm"][j]),
                "margin_vq": float(meta["margin_vq"][j]),
                "pred_hmm": int(preds["hmm"][j]),
                "pred_vq": int(preds["vq"][j]),
                "pred_agreement": int(preds["agreement"][j]),
                "pred_soft": int(preds["soft"][j]),
                "pred_conf_weighted": int(preds["conf_weighted"][j]),
                "pred_margin_weighted": int(preds["margin_weighted"][j]),
                "scores_hmm": [float(x) for x in scores["hmm"][j]],
                "scores_vq": [float(x) for x in scores["vq"][j]],
                "scores_agreement": [float(x) for x in scores["agreement"][j]],
                "scores_soft": [float(x) for x in scores["soft"][j]],
                "scores_conf_weighted": [float(x) for x in scores["conf_weighted"][j]],
                "scores_margin_weighted": [float(x) for x in scores["margin_weighted"][j]],
            })
        rows_by_uid[test_uid] = user_rows

        with open(ckpt_path, "w") as f:
            json.dump(rows_by_uid, f)

        y_true = np.array([s.digit for s in test_samples])
        accs = {k: float(accuracy_score(y_true, preds[k]))
                for k in ("hmm", "vq", "agreement", "conf_weighted")}
        elapsed = time.time() - t0
        print(f"  [LOO {len(rows_by_uid):2d}/{len(user_ids)}] uid={test_uid} "
              f"hmm={accs['hmm']*100:5.1f}% vq={accs['vq']*100:5.1f}% "
              f"agree={accs['agreement']*100:5.1f}% "
              f"cw={accs['conf_weighted']*100:5.1f}% "
              f"({elapsed:.0f}s)", flush=True)

    # Flatten + global accuracies
    all_rows = []
    for uid in user_ids:
        all_rows.extend(rows_by_uid[uid])
    y_true = np.array([r["y_true"] for r in all_rows])
    methods = ("hmm", "vq", "agreement", "soft",
               "conf_weighted", "margin_weighted")
    preds_all = {m: np.array([r[f"pred_{m}"] for r in all_rows])
                 for m in methods}
    accuracies = {m: float(accuracy_score(y_true, p))
                  for m, p in preds_all.items()}
    accuracies["oracle"] = float(np.mean(
        (preds_all["hmm"] == y_true) | (preds_all["vq"] == y_true)))
    accuracies["both_wrong"] = float(np.mean(
        (preds_all["hmm"] != y_true) & (preds_all["vq"] != y_true)))

    print(f"\n[LOO] Accuracies globales:")
    for m in methods + ("oracle",):
        print(f"    {m:18s}: {accuracies[m]*100:6.2f}%")
    print(f"    fallos solapados: {accuracies['both_wrong']*100:.2f}%")

    for m in methods:
        cm = confusion_matrix(y_true, preds_all[m], labels=list(range(10)))
        plot_confusion(cm, f"LOO - {m} ({accuracies[m]*100:.2f}%)",
                       os.path.join(PLOTS_DIR, f"cm_LOO_{m}.png"))

    with open(out_path, "w") as f:
        json.dump({
            "tag": tag,
            "hmm_config": HMM_CFG,
            "hmm_fit": HMM_FIT_LOO,
            "vq_features": VQ_FEATURE_NAME,
            "vq_features_idx": VQ_FEATURES_IDX,
            "vq_k": VQ_K,
            "n_folds": len(user_ids),
            "accuracies": accuracies,
            "rows": all_rows,
        }, f, indent=2)
    print(f"  Guardado: {out_path}", flush=True)
    return accuracies


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  ENSEMBLE PARALELO HMM (GMMHMM med) + VQ (MBKMeans pos_ang_curv k=128)")
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
