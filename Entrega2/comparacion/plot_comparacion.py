"""
=============================================================================
PLOTS DE COMPARACION HMM vs VQ - por usuario
=============================================================================
Lee predicciones_<escenario>.json (producido por comparar_HMM_VQ.py) y genera
scatter por SUJETO: un punto por usuario con x=accuracy VQ, y=accuracy HMM.
Configs de Entrega 2 (mejores):
  HMM  -> GMMHMM n_mix=2, n_estados=7, diag, p_autolazo=0.6, features=med
  VQ   -> KMeans 32 centroides por digito, features 7D (x,y,dx,dy,v,sin,cos)

Tambien anade matriz de acuerdo por-muestra y cota de oraculo para
contextualizar si el ensemble merece la pena.
=============================================================================
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "resultados")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

SCENARIOS = ["N74", "N47", "LOO"]
COLORS = {
    "both_ok":    "#27ae60",   # ambos aciertan
    "only_hmm":   "#2980b9",   # solo HMM acierta (VQ falla)
    "only_vq":    "#e67e22",   # solo VQ acierta (HMM falla)
    "both_fail":  "#c0392b",   # ambos fallan
}


def load_predictions(tag):
    path = os.path.join(RESULTS_DIR, f"predicciones_{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def categorize(rows):
    """Return arrays (y_true, h_ok, v_ok, cat_label)."""
    y_true = np.array([r["y_true"] for r in rows])
    y_h = np.array([r["y_pred_hmm"] for r in rows])
    y_v = np.array([r["y_pred_vq"] for r in rows])
    h_ok = (y_h == y_true).astype(int)
    v_ok = (y_v == y_true).astype(int)

    cats = np.empty(len(rows), dtype=object)
    cats[(h_ok == 1) & (v_ok == 1)] = "both_ok"
    cats[(h_ok == 1) & (v_ok == 0)] = "only_hmm"
    cats[(h_ok == 0) & (v_ok == 1)] = "only_vq"
    cats[(h_ok == 0) & (v_ok == 0)] = "both_fail"
    return y_true, y_h, y_v, h_ok, v_ok, cats


def agreement_stats(rows):
    _, _, _, h_ok, v_ok, cats = categorize(rows)
    n = len(rows)
    counts = {k: int(np.sum(cats == k)) for k in COLORS}
    fracs = {k: v / n for k, v in counts.items()}
    acc_hmm = h_ok.mean()
    acc_vq = v_ok.mean()
    acc_oracle = (h_ok | v_ok).mean()  # al menos uno acierta
    acc_majority = ((h_ok + v_ok) >= 1).mean()  # same as oracle for 2 models
    cohen = cohen_kappa(h_ok, v_ok)
    return {
        "n": n,
        "acc_hmm": acc_hmm,
        "acc_vq": acc_vq,
        "acc_oracle": acc_oracle,
        "counts": counts,
        "fracs": fracs,
        "cohen_kappa": cohen,
    }


def cohen_kappa(a, b):
    """Kappa sobre si-aciertan (no sobre clase predicha)."""
    n = len(a)
    po = np.mean(a == b)
    pa_1 = a.mean() * b.mean()
    pa_0 = (1 - a.mean()) * (1 - b.mean())
    pe = pa_1 + pa_0
    if 1 - pe < 1e-9:
        return 0.0
    return (po - pe) / (1 - pe)


# =============================================================================
# PER-USER ACCURACY (x=VQ, y=HMM)
# =============================================================================

def per_user_accuracies(rows):
    """Devuelve dict {user_id: (acc_hmm, acc_vq, n_samples)}."""
    by_user = {}
    for r in rows:
        u = r["user_id"]
        if u not in by_user:
            by_user[u] = {"n": 0, "h_ok": 0, "v_ok": 0}
        d = by_user[u]
        d["n"] += 1
        if r["y_pred_hmm"] == r["y_true"]:
            d["h_ok"] += 1
        if r["y_pred_vq"] == r["y_true"]:
            d["v_ok"] += 1
    out = {}
    for u, d in by_user.items():
        out[u] = (d["h_ok"] / d["n"], d["v_ok"] / d["n"], d["n"])
    return out


def plot_per_user_scatter(all_data):
    """Scatter por sujeto: x=VQ acc, y=HMM acc. Una subplot por escenario."""
    cols = [k for k in SCENARIOS if all_data.get(k)]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(5.8 * len(cols), 5.5),
                              squeeze=False)
    for i, tag in enumerate(cols):
        ax = axes[0, i]
        pu = per_user_accuracies(all_data[tag]["rows"])
        ids = sorted(pu.keys())
        xs = np.array([pu[u][1] * 100 for u in ids])  # VQ
        ys = np.array([pu[u][0] * 100 for u in ids])  # HMM
        ns = np.array([pu[u][2] for u in ids])

        # Jitter muy pequeno para evitar puntos exactamente solapados
        rng = np.random.RandomState(42)
        xj = xs + rng.uniform(-0.25, 0.25, size=len(xs))
        yj = ys + rng.uniform(-0.25, 0.25, size=len(ys))

        # Colorear por lado de la diagonal
        color = np.where(ys > xs, "#2980b9",
                         np.where(ys < xs, "#e67e22", "#7f8c8d"))

        ax.scatter(xj, yj, s=40, c=color, alpha=0.75, edgecolors="white",
                   linewidths=0.5)

        # Diagonal y=x
        lo = min(xs.min(), ys.min()) - 3
        hi = 101
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray",
                lw=1, alpha=0.7, label="y = x")

        # Anotar usuarios en las esquinas (peores casos)
        worst = np.argsort(np.minimum(xs, ys))[:3]
        for idx in worst:
            ax.annotate(str(ids[idx]), (xj[idx], yj[idx]),
                        fontsize=7, alpha=0.7,
                        xytext=(4, -2), textcoords="offset points")

        # Medias
        ax.axhline(ys.mean(), color="#2980b9", lw=0.6, alpha=0.3)
        ax.axvline(xs.mean(), color="#e67e22", lw=0.6, alpha=0.3)

        # Correlacion y cuentas
        if len(xs) > 1 and np.std(xs) > 0 and np.std(ys) > 0:
            r = np.corrcoef(xs, ys)[0, 1]
        else:
            r = float("nan")
        n_hmm_mejor = int((ys > xs).sum())
        n_vq_mejor = int((xs > ys).sum())
        n_igual = int((xs == ys).sum())

        ax.set_xlabel("Accuracy VQ por usuario (%)")
        ax.set_ylabel("Accuracy HMM por usuario (%)")
        ax.set_title(
            f"{tag}  |  N={len(ids)} usuarios\n"
            f"HMM media={ys.mean():.1f}%   VQ media={xs.mean():.1f}%   "
            f"r={r:.2f}\n"
            f"HMM>VQ: {n_hmm_mejor}   VQ>HMM: {n_vq_mejor}   "
            f"iguales: {n_igual}",
            fontsize=9,
        )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "HMM vs VQ  -  accuracy por sujeto (mejor config Entrega 2)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "scatter_por_usuario.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out}")


def plot_per_user_delta(all_data):
    """Histograma de (HMM - VQ) por usuario: cuanto mejora HMM sobre VQ."""
    cols = [k for k in SCENARIOS if all_data.get(k)]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(5.8 * len(cols), 4.2),
                              squeeze=False)
    for i, tag in enumerate(cols):
        ax = axes[0, i]
        pu = per_user_accuracies(all_data[tag]["rows"])
        delta = np.array([(pu[u][0] - pu[u][1]) * 100 for u in pu])
        ax.hist(delta, bins=20, color="#34495e", edgecolor="white")
        ax.axvline(0, color="red", lw=1, ls="--", alpha=0.6)
        ax.axvline(delta.mean(), color="#27ae60", lw=1.5,
                   label=f"media = {delta.mean():+.2f} pp")
        ax.set_xlabel("HMM - VQ  (pp por usuario)")
        ax.set_ylabel("# usuarios")
        ax.set_title(f"{tag}  (N={len(delta)})", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Distribucion de la diferencia HMM - VQ por sujeto",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "delta_por_usuario.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out}")


# =============================================================================
# PLOT 1: scatter jittered de correcciones
# =============================================================================

def plot_scatter(rows, tag, ax=None):
    _, _, _, h_ok, v_ok, cats = categorize(rows)
    n = len(rows)

    # Jitter grande para que se vean las densidades
    rng = np.random.RandomState(0)
    x = h_ok + rng.uniform(-0.32, 0.32, size=n)
    y = v_ok + rng.uniform(-0.32, 0.32, size=n)

    stats = agreement_stats(rows)

    solo = ax is None
    if solo:
        fig, ax = plt.subplots(figsize=(6.5, 6))

    for k, c in COLORS.items():
        mask = cats == k
        if mask.sum() == 0:
            continue
        ax.scatter(x[mask], y[mask], s=18, alpha=0.55, c=c,
                   edgecolors="none", label=_label_for(k, stats))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["HMM falla", "HMM acierta"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["VQ falla", "VQ acierta"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.6, 1.6)
    ax.axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.axvline(0.5, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.set_title(
        f"{tag}  |  HMM={stats['acc_hmm']*100:.1f}%  "
        f"VQ={stats['acc_vq']*100:.1f}%  "
        f"oraculo={stats['acc_oracle']*100:.1f}%  "
        f"κ={stats['cohen_kappa']:.2f}",
        fontsize=10,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
              frameon=False)

    if solo:
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"scatter_{tag}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Guardado: {out}")


def _label_for(key, stats):
    names = {
        "both_ok":   "ambos aciertan",
        "only_hmm":  "solo HMM",
        "only_vq":   "solo VQ",
        "both_fail": "ambos fallan",
    }
    c = stats["counts"][key]
    f = stats["fracs"][key] * 100
    return f"{names[key]}  ({c}, {f:.1f}%)"


# =============================================================================
# PLOT 2: scatter combinado para los 3 escenarios
# =============================================================================

def plot_scatter_combined(all_data):
    cols = [k for k in SCENARIOS if all_data.get(k)]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(7 * len(cols), 6),
                              squeeze=False)
    for i, tag in enumerate(cols):
        plot_scatter(all_data[tag]["rows"], tag, ax=axes[0, i])

    fig.suptitle("HMM vs VQ  -  scatter de aciertos por muestra "
                 "(jitter en ambos ejes)", fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "scatter_comparacion.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out}")


# =============================================================================
# PLOT 3: matriz de acuerdo 2x2 por escenario
# =============================================================================

def plot_agreement_matrix(all_data):
    cols = [k for k in SCENARIOS if all_data.get(k)]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4.5),
                              squeeze=False)
    for i, tag in enumerate(cols):
        stats = agreement_stats(all_data[tag]["rows"])
        mat = np.array([
            [stats["counts"]["both_fail"], stats["counts"]["only_vq"]],
            [stats["counts"]["only_hmm"],  stats["counts"]["both_ok"]],
        ])
        ax = axes[0, i]
        sns.heatmap(mat, annot=True, fmt="d", cmap="Blues",
                    cbar=False, ax=ax,
                    xticklabels=["VQ falla", "VQ acierta"],
                    yticklabels=["HMM falla", "HMM acierta"])
        ax.set_title(
            f"{tag}\n"
            f"oraculo={stats['acc_oracle']*100:.1f}%  "
            f"(ganancia sobre HMM = "
            f"{(stats['acc_oracle']-stats['acc_hmm'])*100:+.1f}pp)",
            fontsize=10,
        )
    fig.suptitle("Matriz de acuerdo HMM x VQ", fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "matriz_acuerdo.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out}")


# =============================================================================
# PLOT 4: desacuerdo por digito
# =============================================================================

def plot_per_digit(all_data):
    cols = [k for k in SCENARIOS if all_data.get(k)]
    if not cols:
        return
    fig, axes = plt.subplots(len(cols), 1, figsize=(9, 3.5 * len(cols)),
                              squeeze=False)
    digits = list(range(10))

    for i, tag in enumerate(cols):
        rows = all_data[tag]["rows"]
        y_true, _, _, h_ok, v_ok, cats = categorize(rows)

        both_ok = np.zeros(10)
        only_h = np.zeros(10)
        only_v = np.zeros(10)
        both_f = np.zeros(10)
        totals = np.zeros(10)
        for d in digits:
            mask = y_true == d
            totals[d] = mask.sum()
            if totals[d] == 0:
                continue
            both_ok[d] = np.sum(cats[mask] == "both_ok") / totals[d] * 100
            only_h[d] = np.sum(cats[mask] == "only_hmm") / totals[d] * 100
            only_v[d] = np.sum(cats[mask] == "only_vq") / totals[d] * 100
            both_f[d] = np.sum(cats[mask] == "both_fail") / totals[d] * 100

        ax = axes[i, 0]
        x = np.arange(10)
        ax.bar(x, both_ok, color=COLORS["both_ok"], label="ambos aciertan")
        ax.bar(x, only_h, bottom=both_ok, color=COLORS["only_hmm"],
               label="solo HMM")
        ax.bar(x, only_v, bottom=both_ok + only_h, color=COLORS["only_vq"],
               label="solo VQ")
        ax.bar(x, both_f, bottom=both_ok + only_h + only_v,
               color=COLORS["both_fail"], label="ambos fallan")
        ax.set_xticks(x)
        ax.set_xlabel("digito")
        ax.set_ylabel("% de muestras")
        ax.set_title(f"{tag} - desglose por digito")
        ax.set_ylim(0, 100)
        if i == 0:
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                      fontsize=8, frameon=False)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "desglose_por_digito.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out}")


# =============================================================================
# PLOT 5: resumen numerico
# =============================================================================

def plot_summary(all_data):
    cols = [k for k in SCENARIOS if all_data.get(k)]
    if not cols:
        return
    labels = cols
    accs_h = [agreement_stats(all_data[k]["rows"])["acc_hmm"] * 100
              for k in cols]
    accs_v = [agreement_stats(all_data[k]["rows"])["acc_vq"] * 100
              for k in cols]
    accs_o = [agreement_stats(all_data[k]["rows"])["acc_oracle"] * 100
              for k in cols]

    x = np.arange(len(cols))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w, accs_h, w, label="HMM", color="#2980b9")
    ax.bar(x, accs_v, w, label="VQ", color="#e67e22")
    ax.bar(x + w, accs_o, w, label="oraculo (HMM OR VQ)", color="#27ae60")
    for xi, (a, b, c) in enumerate(zip(accs_h, accs_v, accs_o)):
        ax.text(xi - w, a + 0.2, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(xi, b + 0.2, f"{b:.1f}", ha="center", fontsize=8)
        ax.text(xi + w, c + 0.2, f"{c:.1f}", ha="center", fontsize=8,
                color="#1e7a3f", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(min(accs_h + accs_v) - 3, 100)
    ax.set_title("HMM vs VQ vs oraculo (cota superior del ensemble)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "resumen_oraculo.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    all_data = {tag: load_predictions(tag) for tag in SCENARIOS}
    disponibles = [k for k, v in all_data.items() if v is not None]
    print(f"Escenarios disponibles: {disponibles}")

    if not disponibles:
        print("No hay predicciones. Ejecuta comparar_HMM_VQ.py primero.")
        return

    # Tabla resumen
    print("\n" + "=" * 72)
    print(f"  {'esc':<5} {'N':>5} {'HMM':>7} {'VQ':>7} {'oracle':>8} "
          f"{'dif':>7} {'kappa':>7}")
    print("-" * 72)
    for tag in disponibles:
        s = agreement_stats(all_data[tag]["rows"])
        dif = s["acc_oracle"] - max(s["acc_hmm"], s["acc_vq"])
        print(f"  {tag:<5} {s['n']:>5d} "
              f"{s['acc_hmm']*100:>6.2f}% "
              f"{s['acc_vq']*100:>6.2f}% "
              f"{s['acc_oracle']*100:>7.2f}% "
              f"{dif*100:>+6.2f} "
              f"{s['cohen_kappa']:>6.2f}")
    print("=" * 72 + "\n")

    # Scatter por sujeto (lo que pidio el usuario)
    plot_per_user_scatter(all_data)
    plot_per_user_delta(all_data)

    # Plots complementarios para contextualizar el ensemble
    plot_scatter_combined(all_data)
    plot_agreement_matrix(all_data)
    plot_per_digit(all_data)
    plot_summary(all_data)


if __name__ == "__main__":
    main()
