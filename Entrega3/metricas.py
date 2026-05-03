"""
=============================================================================
METRICAS BIOMETRICAS (AUC + EER) - ENTREGA 3
=============================================================================
Lee los JSON de salida de los dos sistemas (Paralel/ y Serial/) y calcula:

  - AUC por clase (one-vs-rest) y AUC macro-promediada.
  - EER por clase y EER macro-promediada.
  - Accuracy global por metodo (la que reporta el JSON de entrada).

Resultados:
  Entrega3/metricas/<sistema>_<TAG>.json   - desglose por metodo y clase.
  Entrega3/metricas/summary.json           - tabla compacta.
  Entrega3/metricas/plots/auc_eer_<TAG>.png - barras AUC/EER por metodo.
=============================================================================
"""

import os
import json
import glob
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARALEL_DIR = os.path.join(SCRIPT_DIR, "Paralel", "resultados")
SERIAL_DIR = os.path.join(SCRIPT_DIR, "Serial", "resultados")
OUT_DIR = os.path.join(SCRIPT_DIR, "metricas")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

PARALEL_METHODS = ["hmm", "vq", "agreement", "soft",
                   "conf_weighted", "margin_weighted"]


def compute_eer(y_true_bin, scores):
    """Equal Error Rate: punto donde FPR == FNR.

    y_true_bin: (N,) binario {0,1}
    scores:     (N,)  mayor = mas probable que sea positivo
    """
    if np.sum(y_true_bin) == 0 or np.sum(y_true_bin) == len(y_true_bin):
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true_bin, scores)
    fnr = 1 - tpr
    # Buscamos el cruce FPR = FNR
    diff = fpr - fnr
    if np.all(diff <= 0) or np.all(diff >= 0):
        # No hay cruce -> aproximamos con el minimo |diff|
        idx = int(np.argmin(np.abs(diff)))
        return float((fpr[idx] + fnr[idx]) / 2.0)
    # Interpolacion lineal en el punto de cruce
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        idx = int(np.argmin(np.abs(diff)))
        return float((fpr[idx] + fnr[idx]) / 2.0)
    i = int(sign_change[0])
    a = diff[i]
    b = diff[i + 1]
    t = a / (a - b) if (a - b) != 0 else 0.5
    fpr_eer = fpr[i] + t * (fpr[i + 1] - fpr[i])
    fnr_eer = fnr[i] + t * (fnr[i + 1] - fnr[i])
    return float((fpr_eer + fnr_eer) / 2.0)


def per_class_auc_eer(y_true, scores_2d, n_classes=10):
    """Returns dict per class + macro averages.

    scores_2d: (N, n_classes), mayor = mas probable.
    """
    auc_per = []
    eer_per = []
    for c in range(n_classes):
        bin_true = (y_true == c).astype(int)
        if bin_true.sum() == 0:
            auc_per.append(float("nan"))
            eer_per.append(float("nan"))
            continue
        s = scores_2d[:, c]
        # AUC
        try:
            fpr, tpr, _ = roc_curve(bin_true, s)
            auc_per.append(float(auc(fpr, tpr)))
        except ValueError:
            auc_per.append(float("nan"))
        eer_per.append(compute_eer(bin_true, s))
    auc_arr = np.array(auc_per, dtype=float)
    eer_arr = np.array(eer_per, dtype=float)
    return {
        "per_class_auc": [float(x) for x in auc_per],
        "per_class_eer": [float(x) for x in eer_per],
        "macro_auc": float(np.nanmean(auc_arr)),
        "macro_eer": float(np.nanmean(eer_arr)),
    }


# =============================================================================
# Lectores
# =============================================================================

def _stack_rows(rows, score_key):
    """Devuelve (y_true, scores 2D)."""
    y = np.array([r["y_true"] for r in rows], dtype=int)
    S = np.array([r[score_key] for r in rows], dtype=float)
    return y, S


def metrics_paralel(json_path):
    with open(json_path) as f:
        data = json.load(f)
    rows = data["rows"]
    tag = data["tag"]
    out = {"tag": tag, "n_samples": len(rows),
           "accuracies": data.get("accuracies", {}),
           "methods": {}}
    for m in PARALEL_METHODS:
        key = f"scores_{m}"
        if not rows or key not in rows[0]:
            continue
        y, S = _stack_rows(rows, key)
        m_metrics = per_class_auc_eer(y, S)
        m_metrics["accuracy"] = data.get("accuracies", {}).get(m, None)
        out["methods"][m] = m_metrics
    return out


def metrics_serial(json_path):
    with open(json_path) as f:
        data = json.load(f)
    rows = data["rows"]
    tag = data["tag"]
    out = {"tag": tag, "n_samples": len(rows),
           "acc_hmm_baseline": data.get("acc_hmm_baseline"),
           "acc_serial": data.get("acc_serial"),
           "methods": {}}
    # HMM baseline scores = log-likelihoods (mayor = mejor)
    if rows and "hmm_lls" in rows[0]:
        y, S = _stack_rows(rows, "hmm_lls")
        m_h = per_class_auc_eer(y, S)
        m_h["accuracy"] = data.get("acc_hmm_baseline")
        out["methods"]["hmm_baseline"] = m_h
    # Serial scores = -min_dist (mayor = mejor)
    if rows and "scores_serial" in rows[0]:
        y, S = _stack_rows(rows, "scores_serial")
        m_s = per_class_auc_eer(y, S)
        m_s["accuracy"] = data.get("acc_serial")
        out["methods"]["serial"] = m_s
    return out


# =============================================================================
# Plots
# =============================================================================

def plot_auc_eer(metrics_by_method, title, filepath):
    methods = list(metrics_by_method.keys())
    aucs = [metrics_by_method[m]["macro_auc"] * 100 for m in methods]
    eers = [metrics_by_method[m]["macro_eer"] * 100 for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(methods, aucs, color="steelblue")
    axes[0].set_ylabel("AUC macro (%)")
    axes[0].set_title(f"{title} - AUC")
    axes[0].set_ylim(min(aucs) - 2, 100.5)
    for i, a in enumerate(aucs):
        axes[0].text(i, a + 0.1, f"{a:.2f}", ha="center", fontsize=8)
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(methods, eers, color="indianred")
    axes[1].set_ylabel("EER macro (%)")
    axes[1].set_title(f"{title} - EER")
    axes[1].set_ylim(0, max(eers) * 1.25 + 0.5)
    for i, a in enumerate(eers):
        axes[1].text(i, a + 0.05, f"{a:.2f}", ha="center", fontsize=8)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    summary = {"paralel": {}, "serial": {}}

    # Paralel
    for path in sorted(glob.glob(os.path.join(PARALEL_DIR, "ensemble_*.json"))):
        tag = os.path.splitext(os.path.basename(path))[0].replace("ensemble_", "")
        if tag in ("summary",):
            continue
        print(f"[Paralel/{tag}] computando AUC/EER...", flush=True)
        m = metrics_paralel(path)
        with open(os.path.join(OUT_DIR, f"paralel_{tag}.json"), "w") as f:
            json.dump(m, f, indent=2)
        summary["paralel"][tag] = {
            method: {
                "macro_auc": v["macro_auc"],
                "macro_eer": v["macro_eer"],
                "accuracy": v.get("accuracy"),
            }
            for method, v in m["methods"].items()
        }
        plot_auc_eer(m["methods"], f"Paralel {tag}",
                     os.path.join(PLOTS_DIR, f"paralel_{tag}.png"))

    # Serial
    for path in sorted(glob.glob(os.path.join(SERIAL_DIR, "serial_*.json"))):
        tag = os.path.splitext(os.path.basename(path))[0].replace("serial_", "")
        if tag in ("summary",):
            continue
        print(f"[Serial/{tag}] computando AUC/EER...", flush=True)
        m = metrics_serial(path)
        with open(os.path.join(OUT_DIR, f"serial_{tag}.json"), "w") as f:
            json.dump(m, f, indent=2)
        summary["serial"][tag] = {
            method: {
                "macro_auc": v["macro_auc"],
                "macro_eer": v["macro_eer"],
                "accuracy": v.get("accuracy"),
            }
            for method, v in m["methods"].items()
        }
        plot_auc_eer(m["methods"], f"Serial {tag}",
                     os.path.join(PLOTS_DIR, f"serial_{tag}.png"))

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Imprimir tabla resumen
    print("\n" + "=" * 78)
    print("  RESUMEN AUC / EER / Accuracy (macro one-vs-rest)")
    print("=" * 78)
    for system in ("paralel", "serial"):
        for tag, methods in summary[system].items():
            print(f"\n[{system.upper()} {tag}]")
            print(f"  {'metodo':22s}  {'acc':>7s}  {'AUC':>7s}  {'EER':>7s}")
            for m, v in methods.items():
                acc = v["accuracy"]
                acc_s = f"{acc*100:6.2f}%" if acc is not None else "  -   "
                print(f"  {m:22s}  {acc_s}  "
                      f"{v['macro_auc']*100:6.2f}%  "
                      f"{v['macro_eer']*100:6.2f}%")


if __name__ == "__main__":
    main()
