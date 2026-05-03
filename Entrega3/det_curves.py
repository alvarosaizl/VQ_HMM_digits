"""
=============================================================================
DET CURVES + RESUMEN EER GLOBAL
=============================================================================
Lee los scores guardados por:
  - Entrega1_rerun/e1_<TAG>.json    (HMM E1, GaussianHMM)
  - Paralel/resultados/ensemble_<TAG>.json  (HMM E2B, VQ opt, fusiones)
  - Serial/resultados/serial_<TAG>.json     (cascada HMM->VQ)

Para cada escenario (N=74, N=47, LOO) y cada modelo, calcula la curva DET
en formato pooled one-vs-rest (concatenando scores de las 10 clases en un
unico problema binario "es la clase c vs no") y la dibuja en escala
probit (norm.ppf). Calcula la EER de cada curva y produce:

  metricas/det/<TAG>.png       - DET con todos los sistemas overlay
  metricas/det/eer_summary.json - tabla EER por sistema y escenario
  metricas/det/eer_summary.md   - tabla markdown lista para informe
=============================================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
E1_DIR = os.path.join(SCRIPT_DIR, "Entrega1_rerun")
PARALEL_DIR = os.path.join(SCRIPT_DIR, "Paralel", "resultados")
SERIAL_DIR = os.path.join(SCRIPT_DIR, "Serial", "resultados")
OUT_DIR = os.path.join(SCRIPT_DIR, "metricas", "det")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# EER y curvas DET (pooled one-vs-rest)
# =============================================================================

def pooled_labels_scores(y_true, scores_2d, n_classes=10):
    """Apila etiquetas binarias y scores para todas las clases.

    Para cada clase c, genera (y == c, scores[:, c]). Concatenando
    devuelve un problema binario unico utilizable para DET global.
    """
    N = len(y_true)
    parts_y = []
    parts_s = []
    for c in range(n_classes):
        if c >= scores_2d.shape[1]:
            continue
        parts_y.append((y_true == c).astype(int))
        parts_s.append(scores_2d[:, c])
    labels = np.concatenate(parts_y)
    scores = np.concatenate(parts_s)
    return labels, scores


def det_points(labels, scores):
    """FPR, FNR a varios umbrales + EER."""
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    # EER: cruce FPR == FNR
    diff = fpr - fnr
    if np.all(diff >= 0) or np.all(diff <= 0):
        idx = int(np.argmin(np.abs(diff)))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
    else:
        sign = np.where(np.diff(np.sign(diff)) != 0)[0]
        if len(sign) == 0:
            idx = int(np.argmin(np.abs(diff)))
            eer = float((fpr[idx] + fnr[idx]) / 2.0)
        else:
            i = int(sign[0])
            a = diff[i]
            b = diff[i + 1]
            t = a / (a - b) if (a - b) != 0 else 0.5
            fpr_eer = fpr[i] + t * (fpr[i + 1] - fpr[i])
            fnr_eer = fnr[i] + t * (fnr[i + 1] - fnr[i])
            eer = float((fpr_eer + fnr_eer) / 2.0)
    return fpr, fnr, eer


def plot_det(systems, title, out_path):
    """systems: list de (label, color, fpr, fnr, eer)."""
    fig, ax = plt.subplots(figsize=(9, 9))

    eps = 1e-4
    for label, color, fpr, fnr, eer in systems:
        ok = (fpr > eps) & (fpr < 1 - eps) & (fnr > eps) & (fnr < 1 - eps)
        if ok.sum() < 2:
            continue
        x = norm.ppf(fpr[ok])
        y = norm.ppf(fnr[ok])
        ax.plot(x, y, color=color, lw=2,
                label=f"{label}  (EER={eer*100:.2f}%)")
        # Marcar EER
        x_e = norm.ppf(eer)
        ax.plot([x_e], [x_e], color=color, marker="o", ms=7,
                markeredgecolor="black", zorder=10)

    # Diagonal de referencia (random + EER line)
    ticks = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    tick_pos = [norm.ppf(t) for t in ticks]
    tick_lab = [f"{t*100:g}%" for t in ticks]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab)

    lo = norm.ppf(eps)
    hi = norm.ppf(0.5)
    ax.plot([lo, hi], [lo, hi], color="grey", ls=":", lw=1, label="EER line")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("False Negative Rate (FRR)")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Lectores
# =============================================================================

def load_paralel(tag):
    path = os.path.join(PARALEL_DIR, f"ensemble_{tag}.json")
    with open(path) as f:
        d = json.load(f)
    rows = d["rows"]
    y = np.array([r["y_true"] for r in rows], dtype=int)
    out = {"y": y}
    for key in ("scores_hmm", "scores_vq", "scores_agreement", "scores_soft",
                "scores_conf_weighted", "scores_margin_weighted"):
        if rows and key in rows[0]:
            out[key] = np.array([r[key] for r in rows], dtype=float)
    return out


def load_serial(tag):
    path = os.path.join(SERIAL_DIR, f"serial_{tag}.json")
    with open(path) as f:
        d = json.load(f)
    rows = d["rows"]
    y = np.array([r["y_true"] for r in rows], dtype=int)
    out = {"y": y}
    if rows and "scores_serial" in rows[0]:
        out["scores_serial"] = np.array(
            [r["scores_serial"] for r in rows], dtype=float)
    if rows and "hmm_lls" in rows[0]:
        out["hmm_lls"] = np.array(
            [r["hmm_lls"] for r in rows], dtype=float)
    return out


def load_e1(tag):
    path = os.path.join(E1_DIR, f"e1_{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    rows = d["rows"]
    y = np.array([r["y_true"] for r in rows], dtype=int)
    return {
        "y": y,
        "scores_e1_hmm": np.array([r["scores_e1_hmm"] for r in rows],
                                  dtype=float),
    }


# =============================================================================
# Plot por escenario
# =============================================================================

# Estilo: nombre humano -> color
PLOT_STYLE = {
    "E1 HMM (GaussianHMM)":           "#1f77b4",
    "E2B HMM (GMMHMM)":               "#9467bd",
    "VQ optimizado":                  "#2ca02c",
    "Paralel agreement":              "#ff7f0e",
    "Paralel soft":                   "#d62728",
    "Paralel margin_weighted":        "#bcbd22",
    "Serial cascade (HMM->VQ)":       "#17becf",
}


def build_scenario(tag):
    """Devuelve lista de (label, color, fpr, fnr, eer) para un escenario."""
    P = load_paralel(tag)
    S = load_serial(tag)
    E1 = load_e1(tag)

    systems = []

    if E1 is not None:
        labels, scores = pooled_labels_scores(E1["y"], E1["scores_e1_hmm"])
        fpr, fnr, eer = det_points(labels, scores)
        systems.append(("E1 HMM (GaussianHMM)",
                        PLOT_STYLE["E1 HMM (GaussianHMM)"],
                        fpr, fnr, eer))

    if "scores_hmm" in P:
        labels, scores = pooled_labels_scores(P["y"], P["scores_hmm"])
        fpr, fnr, eer = det_points(labels, scores)
        systems.append(("E2B HMM (GMMHMM)",
                        PLOT_STYLE["E2B HMM (GMMHMM)"],
                        fpr, fnr, eer))

    if "scores_vq" in P:
        labels, scores = pooled_labels_scores(P["y"], P["scores_vq"])
        fpr, fnr, eer = det_points(labels, scores)
        systems.append(("VQ optimizado",
                        PLOT_STYLE["VQ optimizado"],
                        fpr, fnr, eer))

    for paralel_method, color_key in (
        ("scores_agreement",        "Paralel agreement"),
        ("scores_soft",             "Paralel soft"),
        ("scores_margin_weighted",  "Paralel margin_weighted"),
    ):
        if paralel_method in P:
            labels, scores = pooled_labels_scores(P["y"], P[paralel_method])
            fpr, fnr, eer = det_points(labels, scores)
            systems.append((color_key, PLOT_STYLE[color_key],
                            fpr, fnr, eer))

    if "scores_serial" in S:
        labels, scores = pooled_labels_scores(S["y"], S["scores_serial"])
        fpr, fnr, eer = det_points(labels, scores)
        systems.append(("Serial cascade (HMM->VQ)",
                        PLOT_STYLE["Serial cascade (HMM->VQ)"],
                        fpr, fnr, eer))

    return systems


# =============================================================================
# Main
# =============================================================================

def main():
    summary = {}
    eer_table = {}  # {tag: {model: eer}}
    for tag in ("N74", "N47", "LOO"):
        print(f"[{tag}] construyendo curvas DET...", flush=True)
        try:
            systems = build_scenario(tag)
        except FileNotFoundError as e:
            print(f"  Falta fichero: {e}", flush=True)
            continue
        if not systems:
            print(f"  No hay sistemas para {tag}", flush=True)
            continue
        plot_det(systems, f"DET - {tag} (pooled one-vs-rest)",
                 os.path.join(OUT_DIR, f"DET_{tag}.png"))
        eer_table[tag] = {label: eer for label, _, _, _, eer in systems}
        summary[tag] = [
            {"model": label, "eer": eer} for label, _, _, _, eer in systems
        ]
        for label, _, _, _, eer in systems:
            print(f"    {label:32s}  EER={eer*100:6.2f}%", flush=True)

    with open(os.path.join(OUT_DIR, "eer_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Tabla markdown
    md = ["# Resumen EER global (pooled one-vs-rest)\n",
          "Para cada modelo, EER es el punto de la curva DET "
          "donde FPR = FNR (calculada sobre los scores agregados de las 10 "
          "clases en formato uno-contra-resto).\n"]
    all_models = set()
    for tag, scores in eer_table.items():
        all_models.update(scores.keys())
    # Orden estable segun PLOT_STYLE
    ordered = [m for m in PLOT_STYLE if m in all_models]
    for m in all_models:
        if m not in ordered:
            ordered.append(m)
    md.append("| Modelo | N=74 EER | N=47 EER | LOO EER |")
    md.append("|--------|----------|----------|---------|")
    for m in ordered:
        row = f"| {m} |"
        for tag in ("N74", "N47", "LOO"):
            v = eer_table.get(tag, {}).get(m)
            row += f" {v*100:6.2f}% |" if v is not None else " - |"
        md.append(row)
    md_text = "\n".join(md) + "\n"
    with open(os.path.join(OUT_DIR, "eer_summary.md"), "w") as f:
        f.write(md_text)
    print("\n" + md_text)


if __name__ == "__main__":
    main()
