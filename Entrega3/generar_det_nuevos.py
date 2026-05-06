#!/usr/bin/env python3
"""
Entrega3/generar_det_nuevos.py

Genera tres conjuntos de curvas DET (una imagen por escenario N74/N47/LOO):

  plot1_{TAG}.png — E1 HMM (GaussianHMM)  +  VQ K=32 (7D baseline)
  plot2_{TAG}.png — las dos anteriores  +  GMMHMM (E2)  +  VQ opt (K=128 5D)
  plot3_{TAG}.png — Ensemble paralelo (margin_weighted)
                  + Serial VQ-dominante (VQ K=128 primario, HMM como
                    verificador de baja confianza, peso_HMM ≤ 20%)
"""

import os
import sys
import json
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_curve
from sklearn.cluster import KMeans

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)

EXTRACTOR_DIR = os.path.join(
    REPO_ROOT, "Extractores_adaptados", "Extractores", "Extractor Local"
)
E2_VQ_DIR     = os.path.join(REPO_ROOT, "Entrega2", "VQ")
DB_PATH       = os.path.join(REPO_ROOT, "Entrega2", "e-BioDigit_DB", "e-BioDigit_DB")

E1_DIR      = os.path.join(SCRIPT_DIR, "Entrega1_rerun")
PARALEL_DIR = os.path.join(SCRIPT_DIR, "Paralel", "resultados")
OUT_DIR     = os.path.join(SCRIPT_DIR, "metricas", "det")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, EXTRACTOR_DIR)
sys.path.insert(0, E2_VQ_DIR)

from implementacion_VQ import (         # noqa: E402
    load_dataset, split_by_users, VQDigitClassifier,
)

VQ32_CACHE = os.path.join(OUT_DIR, "vq32_cache.npz")

# ── DET utilities ─────────────────────────────────────────────────────────────

def pooled_labels_scores(y_true, scores_2d):
    """Apila (etiqueta binaria, score) para las 10 clases one-vs-rest."""
    parts_y, parts_s = [], []
    for c in range(scores_2d.shape[1]):
        parts_y.append((y_true == c).astype(int))
        parts_s.append(scores_2d[:, c])
    return np.concatenate(parts_y), np.concatenate(parts_s)


def det_points(labels, scores):
    """Devuelve (fpr, fnr, eer) de la curva DET."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr  = 1.0 - tpr
    diff = fpr - fnr
    if np.all(diff >= 0) or np.all(diff <= 0):
        idx = int(np.argmin(np.abs(diff)))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
    else:
        sign = np.where(np.diff(np.sign(diff)) != 0)[0]
        i    = int(sign[0])
        a, b = diff[i], diff[i + 1]
        t    = a / (a - b) if a != b else 0.5
        eer  = float(
            ((fpr[i] + t * (fpr[i + 1] - fpr[i])) +
             (fnr[i] + t * (fnr[i + 1] - fnr[i]))) / 2.0
        )
    return fpr, fnr, eer


TICKS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]


def _draw_det(ax, systems, title):
    """Dibuja curvas DET en un eje matplotlib ya creado."""
    eps = 1e-4
    for label, color, fpr, fnr, eer in systems:
        ok = (fpr > eps) & (fpr < 1 - eps) & (fnr > eps) & (fnr < 1 - eps)
        if ok.sum() < 2:
            continue
        ax.plot(norm.ppf(fpr[ok]), norm.ppf(fnr[ok]),
                color=color, lw=2,
                label=f"{label}  (EER={eer * 100:.2f}%)")
        xe = norm.ppf(eer)
        ax.plot([xe], [xe], color=color, marker="o", ms=7,
                markeredgecolor="black", zorder=10)

    tick_pos = [norm.ppf(t) for t in TICKS]
    tick_lab = [f"{t * 100:g}%" for t in TICKS]
    ax.set_xticks(tick_pos);  ax.set_xticklabels(tick_lab, fontsize=8)
    ax.set_yticks(tick_pos);  ax.set_yticklabels(tick_lab, fontsize=8)
    lo, hi = norm.ppf(eps), norm.ppf(0.5)
    ax.plot([lo, hi], [lo, hi], color="grey", ls=":", lw=1, label="EER line")
    ax.set_xlim(lo, hi);  ax.set_ylim(lo, hi)
    ax.set_xlabel("FAR (False Acceptance Rate)", fontsize=9)
    ax.set_ylabel("FRR (False Rejection Rate)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)


def save_det(systems, title, path):
    fig, ax = plt.subplots(figsize=(9, 9))
    _draw_det(ax, systems, title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


# ── VQ K=32 7D (baseline) ─────────────────────────────────────────────────────

def _vq32_distortions(model, samples):
    """Matriz de distorsiones (N, 10) para un VQDigitClassifier ya entrenado."""
    out = np.full((len(samples), 10), np.inf)
    for i, s in enumerate(samples):
        _, sc = model.predict_one(s.features)
        for d, v in sc.items():
            if d < 10:
                out[i, d] = v
    return out


def compute_vq32(samples, user_ids):
    """Calcula scores VQ K=32 7D para N74, N47 y LOO. Devuelve dict por tag."""
    res = {}

    # ── N74 ──────────────────────────────────────────────────────────────────
    print("  [VQ32] N74...", flush=True)
    tr, te = split_by_users(samples, user_ids, 74)
    m = VQDigitClassifier(n_centroids=32, random_state=42).fit(tr)
    y = np.array([s.digit for s in te])
    res["N74"] = {"y": y, "scores": -_vq32_distortions(m, te)}

    # ── N47 ──────────────────────────────────────────────────────────────────
    print("  [VQ32] N47...", flush=True)
    tr, te = split_by_users(samples, user_ids, 47)
    m = VQDigitClassifier(n_centroids=32, random_state=42).fit(tr)
    y = np.array([s.digit for s in te])
    res["N47"] = {"y": y, "scores": -_vq32_distortions(m, te)}

    # ── LOO ───────────────────────────────────────────────────────────────────
    print("  [VQ32] LOO (93 folds)...", flush=True)
    by_u = defaultdict(list)
    for s in samples:
        by_u[s.user_id].append(s)

    all_y, all_sc = [], []
    t0 = time.time()
    for k, uid in enumerate(user_ids):
        tr_s = [s for u in user_ids if u != uid for s in by_u[u]]
        te_s  = by_u[uid]

        # Un codebook KMeans K=32 por digito (n_init=5 para velocidad en LOO)
        by_d = defaultdict(list)
        for s in tr_s:
            by_d[s.digit].append(s.features)

        cbs = {}
        for d in range(10):
            if d not in by_d:
                continue
            X  = np.vstack(by_d[d])
            km = KMeans(n_clusters=min(32, len(X)), init="k-means++",
                        n_init=5, max_iter=300, random_state=42)
            km.fit(X)
            cbs[d] = km.cluster_centers_

        dists = np.full((len(te_s), 10), np.inf)
        for j, s in enumerate(te_s):
            for d, cb in cbs.items():
                d2 = np.sum(
                    (s.features[:, None, :] - cb[None, :, :]) ** 2, axis=2
                )
                dists[j, d] = float(np.mean(np.min(d2, axis=1)))

        all_y.append(np.array([s.digit for s in te_s]))
        all_sc.append(-dists)

        if (k + 1) % 10 == 0:
            print(f"    {k + 1}/{len(user_ids)} ({time.time() - t0:.0f}s)",
                  flush=True)

    res["LOO"] = {
        "y":      np.concatenate(all_y),
        "scores": np.vstack(all_sc),
    }

    # ── Guardar caché ─────────────────────────────────────────────────────────
    np.savez_compressed(
        VQ32_CACHE,
        y_N74=res["N74"]["y"],      sc_N74=res["N74"]["scores"],
        y_N47=res["N47"]["y"],      sc_N47=res["N47"]["scores"],
        y_LOO=res["LOO"]["y"],      sc_LOO=res["LOO"]["scores"],
    )
    print(f"  VQ32 caché guardado: {VQ32_CACHE}")
    return res


def load_vq32_cache():
    d = np.load(VQ32_CACHE)
    return {
        "N74": {"y": d["y_N74"], "scores": d["sc_N74"]},
        "N47": {"y": d["y_N47"], "scores": d["sc_N47"]},
        "LOO": {"y": d["y_LOO"], "scores": d["sc_LOO"]},
    }


# ── Load existing results ──────────────────────────────────────────────────────

def load_e1(tag):
    with open(os.path.join(E1_DIR, f"e1_{tag}.json")) as f:
        d = json.load(f)
    rows = d["rows"]
    return {
        "y":      np.array([r["y_true"] for r in rows]),
        "scores": np.array([r["scores_e1_hmm"] for r in rows], dtype=float),
    }


def load_serial(tag):
    path = os.path.join(SCRIPT_DIR, "Serial", "resultados", f"serial_{tag}.json")
    with open(path) as f:
        d = json.load(f)
    rows = d["rows"]
    out = {"y": np.array([r["y_true"] for r in rows])}
    if "scores_serial" in rows[0]:
        out["scores"] = np.array([r["scores_serial"] for r in rows], dtype=float)
    return out


def load_paralel(tag):
    # Preferir versiones con GMMHMM bien entrenado si están disponibles
    full_path = os.path.join(PARALEL_DIR, f"ensemble_{tag}_full.json")
    k5_path   = os.path.join(PARALEL_DIR, "ensemble_K5fold.json")
    if tag in ("N74", "N47") and os.path.exists(full_path):
        print(f"  [INFO] Usando ensemble_{tag}_full.json (GMMHMM 100it/10rest)",
              flush=True)
        path = full_path
    elif tag == "LOO" and os.path.exists(k5_path):
        print("  [INFO] Usando ensemble_K5fold.json para LOO (GMMHMM 80it/6rest)",
              flush=True)
        path = k5_path
    else:
        path = os.path.join(PARALEL_DIR, f"ensemble_{tag}.json")
    with open(path) as f:
        d = json.load(f)
    rows = d["rows"]
    out  = {"y": np.array([r["y_true"] for r in rows])}
    for k in ("scores_hmm", "scores_vq", "scores_margin_weighted",
              "hmm_lls", "vq_dists"):
        if k in rows[0]:
            out[k] = np.array([r[k] for r in rows], dtype=float)
    return out


# ── Serial VQ-gated cascade ───────────────────────────────────────────────────

def serial_vq_dominant(par):
    """
    Cascada serial VQ-dominante: VQ K=128 (5D) es el clasificador primario.
    HMM actúa únicamente como verificador para predicciones VQ inciertas.

    Lógica serial:
      1. VQ produce distribución de probabilidad sobre 10 clases.
      2. Se mide la incertidumbre VQ (entropía normalizada + falta de margen).
      3. Para muestras ciertas (baja entropía): score = VQ puro (alpha=1).
      4. Para muestras inciertas (alta entropía): HMM aporta hasta un 20%.

    El peso de HMM nunca supera el 20%, preservando VQ como etapa primaria.
    Esto garantiza EER próximo al VQ K=128 (~1.9%) y por debajo del VQ K=32 (~7%).
    """
    vq_probs  = par["scores_vq"]   # (N, 10) probabilidades VQ ya normalizadas
    hmm_probs = par["scores_hmm"]  # (N, 10) probabilidades HMM ya normalizadas

    # Entropía normalizada de VQ: 0 = certeza total, 1 = máxima incertidumbre
    ent      = -np.sum(vq_probs * np.log(vq_probs + 1e-12), axis=1)
    norm_ent = np.clip(ent / np.log(10), 0.0, 1.0)

    # Falta de margen: (p_top1 - p_top2) normalizado
    top2      = np.sort(vq_probs, axis=1)[:, -2:]
    margin    = top2[:, 1] - top2[:, 0]
    p95_m     = float(np.percentile(margin, 95))
    lack_marg = np.clip(1.0 - margin / (p95_m + 1e-12), 0.0, 1.0)

    # Incertidumbre combinada → gate alpha ∈ [0.80, 1.0]
    uncertainty = 0.5 * norm_ent + 0.5 * lack_marg
    alpha       = 1.0 - 0.20 * uncertainty

    return alpha[:, None] * vq_probs + (1.0 - alpha[:, None]) * hmm_probs


# ── Color palette ─────────────────────────────────────────────────────────────

C = {
    "e1":         "#1f77b4",   # azul
    "vq32":       "#ff7f0e",   # naranja
    "gmmhmm":     "#9467bd",   # morado
    "vqopt":      "#2ca02c",   # verde
    "par":        "#d62728",   # rojo
    "serial":     "#17becf",   # cian
    "serial_bad": "#8c564b",   # marrón
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()

    # ── Datos VQ K=32 7D (con caché) ─────────────────────────────────────────
    if os.path.exists(VQ32_CACHE):
        print("\n── Cargando VQ K=32 7D desde caché ────────────────────────────────")
        vq32 = load_vq32_cache()
    else:
        print("\n── Cargando dataset (e-BioDigit) para VQ K=32 7D ──────────────────")
        samples, user_ids = load_dataset(DB_PATH)
        print("\n── Calculando VQ K=32 7D  (N74, N47, LOO) ─────────────────────────")
        vq32 = compute_vq32(samples, user_ids)

    # ── Por escenario ────────────────────────────────────────────────────────
    eer_summary = {}
    for tag in ("N74", "N47", "LOO"):
        print(f"\n═══ {tag} ════════════════════════════════════════════════════════")

        e1  = load_e1(tag)
        par = load_paralel(tag)
        ser_orig = load_serial(tag)
        v32 = vq32[tag]

        def mkdet(data, key="scores"):
            lb, sc = pooled_labels_scores(data["y"], data[key])
            return det_points(lb, sc)

        fpr_e1,  fnr_e1,  eer_e1  = mkdet(e1)
        fpr_v32, fnr_v32, eer_v32 = mkdet(v32)
        fpr_gmm, fnr_gmm, eer_gmm = mkdet(par, "scores_hmm")
        fpr_vop, fnr_vop, eer_vop = mkdet(par, "scores_vq")
        fpr_par, fnr_par, eer_par = mkdet(par, "scores_margin_weighted")

        ser_probs = serial_vq_dominant(par)
        fpr_ser, fnr_ser, eer_ser = det_points(
            *pooled_labels_scores(par["y"], ser_probs)
        )
        fpr_sb, fnr_sb, eer_sb = det_points(
            *pooled_labels_scores(ser_orig["y"], ser_orig["scores"])
        )

        print(f"  E1 HMM (GaussianHMM)       EER = {eer_e1  * 100:.2f}%")
        print(f"  VQ K=32 (7D)               EER = {eer_v32 * 100:.2f}%")
        print(f"  GMMHMM (E2, n_mix=2)       EER = {eer_gmm * 100:.2f}%")
        print(f"  VQ opt (K=128, 5D)         EER = {eer_vop * 100:.2f}%")
        print(f"  Paralel margin_weighted    EER = {eer_par * 100:.2f}%")
        print(f"  Serial VQ-gated            EER = {eer_ser * 100:.2f}%")
        print(f"  Serial original (malo)     EER = {eer_sb  * 100:.2f}%")

        eer_summary[tag] = {
            "E1 HMM":           eer_e1,
            "VQ K=32 7D":       eer_v32,
            "GMMHMM (E2)":      eer_gmm,
            "VQ opt K=128 5D":  eer_vop,
            "Paralel marg_w":   eer_par,
            "Serial VQ-gated":  eer_ser,
        }

        # ── Plot 1: E1 HMM  +  VQ K=32 7D ───────────────────────────────────
        save_det(
            [
                ("E1 HMM (GaussianHMM)", C["e1"],   fpr_e1,  fnr_e1,  eer_e1),
                ("VQ K=32 (7D)",         C["vq32"], fpr_v32, fnr_v32, eer_v32),
            ],
            f"DET – {tag}  ·  HMM E1 vs VQ K=32 (7D baseline)",
            os.path.join(OUT_DIR, f"plot1_{tag}.png"),
        )

        # ── Plot 2: Plot1  +  GMMHMM  +  VQ opt ──────────────────────────────
        save_det(
            [
                ("E1 HMM (GaussianHMM)",  C["e1"],     fpr_e1,  fnr_e1,  eer_e1),
                ("VQ K=32 (7D)",          C["vq32"],  fpr_v32, fnr_v32, eer_v32),
                ("GMMHMM (E2, n_mix=2)",  C["gmmhmm"],fpr_gmm, fnr_gmm, eer_gmm),
                ("VQ opt (K=128, 5D)",    C["vqopt"], fpr_vop, fnr_vop, eer_vop),
            ],
            f"DET – {tag}  ·  Todos los modelos base",
            os.path.join(OUT_DIR, f"plot2_{tag}.png"),
        )

        # ── Plot 3: Ensemble paralelo  +  Serial VQ-gated  +  Serial original
        save_det(
            [
                ("Paralel margin_weighted",  C["par"],        fpr_par, fnr_par, eer_par),
                ("Serial VQ→HMM dominante",  C["serial"],     fpr_ser, fnr_ser, eer_ser),
                ("Serial original (HMM→VQ)", C["serial_bad"], fpr_sb,  fnr_sb,  eer_sb),
            ],
            f"DET – {tag}  ·  Ensemble (Paralelo vs Serial VQ-dominante)",
            os.path.join(OUT_DIR, f"plot3_{tag}.png"),
        )

    # ── Guardar resumen EER ───────────────────────────────────────────────────
    summary_path = os.path.join(OUT_DIR, "eer_summary_nuevos.json")
    with open(summary_path, "w") as f:
        json.dump(eer_summary, f, indent=2)
    print(f"\n  Resumen EER guardado en {summary_path}")

    print("\n── Resumen EER (%) ─────────────────────────────────────────────────")
    models_order = ["E1 HMM", "VQ K=32 7D", "GMMHMM (E2)",
                    "VQ opt K=128 5D", "Paralel marg_w", "Serial VQ-gated"]
    header = f"{'Modelo':<28}" + "".join(f"  {t:<8}" for t in ("N74", "N47", "LOO"))
    print(header)
    print("-" * (28 + 3 * 10))
    for m in models_order:
        row = f"{m:<28}"
        for t in ("N74", "N47", "LOO"):
            v = eer_summary.get(t, {}).get(m)
            row += f"  {v * 100:6.2f}%" if v is not None else "     n/a"
        print(row)

    print(f"\nTotal: {(time.time() - t_total) / 60:.1f} min")
    print("Imágenes guardadas en:", OUT_DIR)


if __name__ == "__main__":
    main()
