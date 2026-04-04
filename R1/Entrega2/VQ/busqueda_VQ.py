"""
=============================================================================
BUSQUEDA DE HIPERPARAMETROS VQ
=============================================================================
Grid search sobre:
  - Algoritmo: KMeans, LBG (Linde-Buzo-Gray), MiniBatchKMeans
  - Features: multiples subconjuntos (3D a 11D)
  - N centroides: 8, 16, 32, 64, 128, 256

Validacion: 78 train / 15 val (misma particion que entregas HMM).
Evaluacion final: N=74, N=47, LOO CV con la mejor config.

Uso:
  python3 busqueda_VQ.py
=============================================================================
"""

import os
import re
import glob
import json
import time
import warnings

import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURACION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(REPO_ROOT, "e-BioDigit_DB", "e-BioDigit_DB")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "resultados_VQ")
os.makedirs(RESULTS_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "checkpoint_busqueda.json")

N_OPT_TRAIN = 78
N_OPT_VAL = 15
N_TRAIN_74 = 74
N_TRAIN_47 = 47

# Grid de busqueda
GRID_ALGORITHMS = ["kmeans", "lbg", "mbkmeans"]
GRID_N_CENTROIDS = [8, 16, 32, 64, 128, 256]

# Feature subsets: name -> list of column indices into the full feature array
# Full features (11D): x, y, vx, vy, v, sin_a, cos_a, ax, ay, |a|, dtheta
FEATURE_SETS = {
    "pos_vel_ang":  [0, 1, 2, 3, 4, 5, 6],          # 7D: x,y,vx,vy,v,sin,cos (baseline)
    "vel_ang":      [2, 3, 4, 5, 6],                  # 5D: vx,vy,v,sin,cos
    "full":         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 11D: todo
    "vel_acc":      [2, 3, 4, 7, 8, 9],               # 6D: vx,vy,v,ax,ay,|a|
    "ang_curv":     [4, 5, 6, 10],                     # 4D: v,sin,cos,dtheta
    "pos_ang_curv": [0, 1, 5, 6, 10],                 # 5D: x,y,sin,cos,dtheta
    "all_no_pos":   [2, 3, 4, 5, 6, 7, 8, 9, 10],    # 9D: todo sin x,y
    "vel_only":     [2, 3, 4],                         # 3D: vx,vy,v
}


# =============================================================================
# Lectura de ficheros
# =============================================================================

def read_ebiodigit_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    n = int(lines[0])
    data = []
    for line in lines[1:n + 1]:
        vals = line.split()
        data.append([float(v) for v in vals[:4]])
    arr = np.array(data, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def parse_label_from_filename(filepath):
    name = os.path.basename(filepath)
    match = re.search(r"u(\d+)_digit_(\d)_(\d+)\.txt", name)
    if not match:
        raise ValueError(f"Nombre no reconocido: {name}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


# =============================================================================
# Preprocesado y features
# =============================================================================

def preprocess_trace(x, y, t):
    t = t - t[0]
    x = x - np.mean(x)
    y = y - np.mean(y)
    scale = max(np.max(np.abs(x)), np.max(np.abs(y)), 1e-8)
    x = x / scale
    y = y / scale
    return x, y, t


def compute_full_features(x, y, t):
    """Calcula 11 features locales por punto.

    Columns:
      0: x, 1: y, 2: vx, 3: vy, 4: v,
      5: sin(angle), 6: cos(angle),
      7: ax, 8: ay, 9: |a|, 10: dtheta (curvatura)
    """
    dt = np.diff(t)
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0

    dx = np.diff(x)
    dy = np.diff(y)
    vx = dx / dt
    vy = dy / dt
    v = np.sqrt(vx**2 + vy**2)
    angle = np.arctan2(dy, dx)
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)

    # Aceleracion (derivada de velocidad)
    if len(vx) > 1:
        dt2 = dt[1:]
        dt2[dt2 <= 0] = np.median(dt2[dt2 > 0]) if np.any(dt2 > 0) else 1.0
        ax = np.diff(vx) / dt2
        ay = np.diff(vy) / dt2
        a_mag = np.sqrt(ax**2 + ay**2)
        # Pad to match length (prepend first value)
        ax = np.concatenate([[ax[0]], ax])
        ay = np.concatenate([[ay[0]], ay])
        a_mag = np.concatenate([[a_mag[0]], a_mag])
    else:
        ax = np.zeros_like(vx)
        ay = np.zeros_like(vy)
        a_mag = np.zeros_like(v)

    # Curvatura (cambio de angulo)
    dtheta = np.diff(angle)
    # Wrap to [-pi, pi]
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    dtheta = np.concatenate([[dtheta[0]], dtheta]) if len(dtheta) > 0 else np.zeros_like(v)

    x_mid = x[1:]
    y_mid = y[1:]

    feats = np.column_stack([x_mid, y_mid, vx, vy, v, sin_a, cos_a,
                              ax, ay, a_mag, dtheta])
    return feats


# =============================================================================
# Carga de datos
# =============================================================================

class RawSample:
    """Stores full features; subsets are extracted on-the-fly."""
    __slots__ = ("user_id", "digit", "sample_id", "full_features")

    def __init__(self, user_id, digit, sample_id, full_features):
        self.user_id = user_id
        self.digit = digit
        self.sample_id = sample_id
        self.full_features = full_features


class FeatureView:
    """Lightweight view into a RawSample with a specific feature subset."""
    __slots__ = ("user_id", "digit", "features")

    def __init__(self, raw, indices):
        self.user_id = raw.user_id
        self.digit = raw.digit
        self.features = raw.full_features[:, indices]


def load_raw_dataset(db_path):
    txt_files = glob.glob(os.path.join(db_path, "**", "*.txt"), recursive=True)
    print(f"  Ficheros encontrados: {len(txt_files)}")

    samples = []
    for fp in txt_files:
        try:
            uid, digit, sid = parse_label_from_filename(fp)
            x, y, t, _ = read_ebiodigit_file(fp)
            x, y, t = preprocess_trace(x, y, t)
            feats = compute_full_features(x, y, t)
            if len(feats) > 0:
                samples.append(RawSample(uid, digit, sid, feats))
        except Exception as e:
            print(f"  Saltando {fp}: {e}")

    user_ids = sorted(set(s.user_id for s in samples))
    print(f"  Muestras cargadas: {len(samples)}, Usuarios: {len(user_ids)}")
    return samples, user_ids


def make_views(raw_samples, feature_indices):
    """Create FeatureView list from raw samples for a given feature subset."""
    return [FeatureView(s, feature_indices) for s in raw_samples]


def split_views_by_users(views, user_ids, n_train):
    train_set = set(user_ids[:n_train])
    train = [v for v in views if v.user_id in train_set]
    test = [v for v in views if v.user_id not in train_set]
    return train, test


# =============================================================================
# Algoritmos VQ
# =============================================================================

def build_codebook_kmeans(X, k, random_state=42):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                max_iter=300, random_state=random_state)
    km.fit(X)
    return km.cluster_centers_


def build_codebook_mbkmeans(X, k, random_state=42):
    km = MiniBatchKMeans(n_clusters=k, init="k-means++", n_init=10,
                         max_iter=300, batch_size=min(1024, len(X)),
                         random_state=random_state)
    km.fit(X)
    return km.cluster_centers_


def build_codebook_lbg(X, k, epsilon=0.01, max_lloyd=50):
    """Linde-Buzo-Gray algorithm: iterative splitting."""
    centroids = np.mean(X, axis=0, keepdims=True)

    while len(centroids) < k:
        # Split each centroid by perturbation
        perturb = epsilon * (np.std(X, axis=0, keepdims=True) + 1e-8)
        new_c = np.vstack([centroids + perturb, centroids - perturb])
        if len(new_c) > k:
            new_c = new_c[:k]
        centroids = new_c

        # Lloyd iterations (KMeans-like refinement)
        for _ in range(max_lloyd):
            dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            assignments = np.argmin(dists, axis=1)

            updated = np.zeros_like(centroids)
            for i in range(len(centroids)):
                mask = assignments == i
                if np.any(mask):
                    updated[i] = np.mean(X[mask], axis=0)
                else:
                    # Dead centroid: reinitialize to random data point
                    updated[i] = X[np.random.randint(len(X))]

            if np.allclose(centroids, updated, atol=1e-6):
                break
            centroids = updated

    return centroids


ALGO_BUILDERS = {
    "kmeans": build_codebook_kmeans,
    "lbg": build_codebook_lbg,
    "mbkmeans": build_codebook_mbkmeans,
}


# =============================================================================
# Clasificador VQ generico
# =============================================================================

class VQClassifier:
    def __init__(self, algorithm="kmeans", n_centroids=32, random_state=42):
        self.algorithm = algorithm
        self.n_centroids = n_centroids
        self.random_state = random_state
        self.codebooks = {}

    def fit(self, train_views, verbose=False):
        by_digit = defaultdict(list)
        for v in train_views:
            by_digit[v.digit].append(v.features)

        builder = ALGO_BUILDERS[self.algorithm]

        for digit in range(10):
            if digit not in by_digit:
                continue
            X = np.vstack(by_digit[digit])
            k = min(self.n_centroids, len(X))
            if self.algorithm == "lbg":
                cb = builder(X, k)
            else:
                cb = builder(X, k, random_state=self.random_state)
            self.codebooks[digit] = cb
            if verbose:
                print(f"    Digito {digit}: {len(X)} vecs -> {k} centroides")
        return self

    def distortion(self, features, codebook):
        dists = np.sum((features[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
        return np.mean(np.min(dists, axis=1))

    def predict(self, views):
        preds = []
        for v in views:
            scores = {d: self.distortion(v.features, cb)
                      for d, cb in self.codebooks.items()}
            preds.append(min(scores, key=scores.get))
        return np.array(preds)


# =============================================================================
# Evaluacion helpers
# =============================================================================

def accuracy_on(model, views):
    y_true = np.array([v.digit for v in views])
    y_pred = model.predict(views)
    return accuracy_score(y_true, y_pred)


def graficar_confusion(cm, titulo, filepath):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10), ax=ax)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {filepath}")


def graficar_top_configs(resultados, filepath, top_n=25):
    sorted_res = sorted(resultados, key=lambda x: x["accuracy"], reverse=True)[:top_n]
    labels = []
    accs = []
    for r in sorted_res:
        label = f"{r['algorithm']:8s} {r['features']:13s} k={r['n_centroids']}"
        labels.append(label)
        accs.append(r["accuracy"] * 100)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(accs)), accs, color="steelblue", edgecolor="navy")
    ax.set_yticks(range(len(accs)))
    ax.set_yticklabels(labels, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(f"Top {top_n} configuraciones VQ (validacion)")
    ax.invert_yaxis()
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {filepath}")


# =============================================================================
# Checkpointing
# =============================================================================

def cargar_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return None


def guardar_checkpoint(resultados):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(resultados, f, indent=2)


# =============================================================================
# LOO CV
# =============================================================================

def loo_cv(raw_samples, user_ids, algorithm, n_centroids, feature_indices):
    samples_by_user = defaultdict(list)
    for s in raw_samples:
        samples_by_user[s.user_id].append(s)

    n_users = len(user_ids)
    accuracies = {}
    cm_acumulada = np.zeros((10, 10), dtype=int)

    for i, test_uid in enumerate(user_ids):
        train_raw = []
        for uid in user_ids:
            if uid != test_uid:
                train_raw.extend(samples_by_user[uid])
        test_raw = samples_by_user[test_uid]

        train_views = make_views(train_raw, feature_indices)
        test_views = make_views(test_raw, feature_indices)

        model = VQClassifier(algorithm=algorithm, n_centroids=n_centroids)
        model.fit(train_views)

        y_true = np.array([v.digit for v in test_views])
        y_pred = model.predict(test_views)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

        accuracies[test_uid] = acc
        cm_acumulada += cm

        acc_parcial = np.mean(list(accuracies.values()))
        print(f"    [{i+1:2d}/{n_users}] Usuario {test_uid}: "
              f"{acc*100:5.1f}%  (media: {acc_parcial*100:.1f}%)", flush=True)

    acc_media = np.mean(list(accuracies.values()))
    acc_std = np.std(list(accuracies.values()))
    return acc_media, acc_std, accuracies, cm_acumulada


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_inicio = time.time()

    print("=" * 70)
    print("  BUSQUEDA DE HIPERPARAMETROS VQ")
    print("=" * 70, flush=True)

    # ------------------------------------------------------------------
    # [1] Cargar datos (full features, 11D)
    # ------------------------------------------------------------------
    print("\n[1] Cargando base de datos (features completas 11D)...", flush=True)
    raw_samples, user_ids = load_raw_dataset(DB_PATH)

    # ------------------------------------------------------------------
    # [2] Grid search con validacion (78 train / 15 val)
    # ------------------------------------------------------------------
    print(f"\n[2] GRID SEARCH ({N_OPT_TRAIN} train / {N_OPT_VAL} val)")
    print("=" * 70, flush=True)

    opt_train_ids = user_ids[:N_OPT_TRAIN]
    opt_val_ids = user_ids[N_OPT_TRAIN:N_OPT_TRAIN + N_OPT_VAL]

    # Generate all configs
    configs = []
    for algo in GRID_ALGORITHMS:
        for feat_name, feat_idx in FEATURE_SETS.items():
            for k in GRID_N_CENTROIDS:
                configs.append({
                    "algorithm": algo,
                    "features": feat_name,
                    "feature_indices": feat_idx,
                    "n_centroids": k,
                })

    total = len(configs)
    print(f"  Total configuraciones: {total}")
    print(f"  Algoritmos: {GRID_ALGORITHMS}")
    print(f"  Features: {list(FEATURE_SETS.keys())}")
    print(f"  N centroides: {GRID_N_CENTROIDS}\n", flush=True)

    # Load checkpoint
    ckpt = cargar_checkpoint()
    resultados = ckpt if ckpt else []
    n_done = len(resultados)
    if n_done > 0:
        print(f"  Checkpoint: {n_done}/{total} evaluadas", flush=True)

    mejor_acc = max((r["accuracy"] for r in resultados), default=0.0)

    for i, cfg in enumerate(configs):
        if i < n_done:
            continue

        feat_idx = cfg["feature_indices"]
        train_views = make_views(
            [s for s in raw_samples if s.user_id in set(opt_train_ids)],
            feat_idx)
        val_views = make_views(
            [s for s in raw_samples if s.user_id in set(opt_val_ids)],
            feat_idx)

        t0 = time.time()
        model = VQClassifier(algorithm=cfg["algorithm"],
                              n_centroids=cfg["n_centroids"])
        model.fit(train_views)
        acc = accuracy_on(model, val_views)
        elapsed = time.time() - t0

        marker = " ***" if acc > mejor_acc else ""
        if acc > mejor_acc:
            mejor_acc = acc

        print(f"  [{i+1:3d}/{total}] {cfg['algorithm']:8s} "
              f"{cfg['features']:13s} k={cfg['n_centroids']:3d} "
              f"-> {acc*100:5.2f}% ({elapsed:.1f}s){marker}", flush=True)

        resultados.append({
            "algorithm": cfg["algorithm"],
            "features": cfg["features"],
            "feature_indices": cfg["feature_indices"],
            "n_centroids": cfg["n_centroids"],
            "accuracy": acc,
            "time": elapsed,
        })

        # Checkpoint every 10 configs
        if (i + 1) % 10 == 0 or i == total - 1:
            guardar_checkpoint(resultados)

    # ------------------------------------------------------------------
    # [3] Analisis de resultados
    # ------------------------------------------------------------------
    print(f"\n\n[3] ANALISIS DE RESULTADOS")
    print("=" * 70, flush=True)

    sorted_all = sorted(resultados, key=lambda x: x["accuracy"], reverse=True)
    best = sorted_all[0]

    print(f"\n  MEJOR CONFIG:")
    print(f"    Algoritmo:    {best['algorithm']}")
    print(f"    Features:     {best['features']} "
          f"({len(best['feature_indices'])}D)")
    print(f"    N centroides: {best['n_centroids']}")
    print(f"    Accuracy val: {best['accuracy']*100:.2f}%")

    # Top 10
    print(f"\n  TOP 10:")
    print(f"  {'Rank':<5} {'Algo':8s} {'Features':13s} {'k':>4} {'Acc':>8}")
    print("  " + "-" * 42)
    for j, r in enumerate(sorted_all[:10]):
        print(f"  {j+1:<5} {r['algorithm']:8s} {r['features']:13s} "
              f"{r['n_centroids']:>4} {r['accuracy']*100:>7.2f}%")

    # Best per algorithm
    print(f"\n  MEJOR POR ALGORITMO:")
    for algo in GRID_ALGORITHMS:
        best_algo = max((r for r in resultados if r["algorithm"] == algo),
                        key=lambda x: x["accuracy"])
        print(f"    {algo:8s}: {best_algo['accuracy']*100:.2f}% "
              f"(feat={best_algo['features']}, k={best_algo['n_centroids']})")

    # Best per feature set
    print(f"\n  MEJOR POR FEATURES:")
    for fname in FEATURE_SETS:
        best_feat = max((r for r in resultados if r["features"] == fname),
                        key=lambda x: x["accuracy"])
        print(f"    {fname:13s}: {best_feat['accuracy']*100:.2f}% "
              f"(algo={best_feat['algorithm']}, k={best_feat['n_centroids']})")

    graficar_top_configs(
        resultados,
        os.path.join(RESULTS_DIR, "top_configs_VQ.png"))

    # ------------------------------------------------------------------
    # [4] Evaluacion final con la mejor config
    # ------------------------------------------------------------------
    print(f"\n\n[4] EVALUACION FINAL")
    print("=" * 70, flush=True)

    best_feat_idx = best["feature_indices"]
    best_algo = best["algorithm"]
    best_k = best["n_centroids"]
    best_feat_name = best["features"]

    # N=74
    print(f"\n  [4a] ESCENARIO N=74", flush=True)
    train_74 = make_views(
        [s for s in raw_samples if s.user_id in set(user_ids[:N_TRAIN_74])],
        best_feat_idx)
    test_74 = make_views(
        [s for s in raw_samples if s.user_id in set(user_ids[N_TRAIN_74:])],
        best_feat_idx)
    print(f"  Train: {len(train_74)} muestras, Test: {len(test_74)} muestras")

    model_74 = VQClassifier(algorithm=best_algo, n_centroids=best_k)
    model_74.fit(train_74, verbose=True)

    y_true_74 = np.array([v.digit for v in test_74])
    y_pred_74 = model_74.predict(test_74)
    acc_74 = accuracy_score(y_true_74, y_pred_74)
    cm_74 = confusion_matrix(y_true_74, y_pred_74, labels=list(range(10)))
    print(f"  >>> Accuracy N=74: {acc_74*100:.2f}%")

    graficar_confusion(
        cm_74,
        f"VQ Optimo N=74 — {best_algo} k={best_k} {best_feat_name} "
        f"({acc_74*100:.1f}%)",
        os.path.join(RESULTS_DIR, "confusion_VQ_opt_N74.png"))

    # N=47
    print(f"\n  [4b] ESCENARIO N=47", flush=True)
    train_47 = make_views(
        [s for s in raw_samples if s.user_id in set(user_ids[:N_TRAIN_47])],
        best_feat_idx)
    test_47 = make_views(
        [s for s in raw_samples if s.user_id in set(user_ids[N_TRAIN_47:])],
        best_feat_idx)
    print(f"  Train: {len(train_47)} muestras, Test: {len(test_47)} muestras")

    model_47 = VQClassifier(algorithm=best_algo, n_centroids=best_k)
    model_47.fit(train_47, verbose=True)

    y_true_47 = np.array([v.digit for v in test_47])
    y_pred_47 = model_47.predict(test_47)
    acc_47 = accuracy_score(y_true_47, y_pred_47)
    cm_47 = confusion_matrix(y_true_47, y_pred_47, labels=list(range(10)))
    print(f"  >>> Accuracy N=47: {acc_47*100:.2f}%")

    graficar_confusion(
        cm_47,
        f"VQ Optimo N=47 — {best_algo} k={best_k} {best_feat_name} "
        f"({acc_47*100:.1f}%)",
        os.path.join(RESULTS_DIR, "confusion_VQ_opt_N47.png"))

    # LOO CV
    print(f"\n  [4c] LOO CROSS-VALIDATION (93 folds)", flush=True)
    t_loo = time.time()
    acc_loo, std_loo, accs_user, cm_loo = loo_cv(
        raw_samples, user_ids, best_algo, best_k, best_feat_idx)
    t_loo = time.time() - t_loo

    print(f"\n  >>> LOO CV: {acc_loo*100:.2f}% +/- {std_loo*100:.2f}% "
          f"({t_loo:.0f}s)")

    graficar_confusion(
        cm_loo,
        f"VQ Optimo LOO — {best_algo} k={best_k} {best_feat_name} "
        f"({acc_loo*100:.1f}% +/- {std_loo*100:.1f}%)",
        os.path.join(RESULTS_DIR, "confusion_VQ_opt_LOO.png"))

    # ------------------------------------------------------------------
    # [5] Guardar resultados
    # ------------------------------------------------------------------
    resumen = {
        "mejor_config": {
            "algorithm": best_algo,
            "features": best_feat_name,
            "n_dims": len(best_feat_idx),
            "n_centroids": best_k,
            "accuracy_val": best["accuracy"],
        },
        "evaluacion_final": {
            "N74": {"accuracy": acc_74},
            "N47": {"accuracy": acc_47},
            "LOO": {
                "accuracy_media": acc_loo,
                "accuracy_std": std_loo,
                "accuracies_por_usuario": {str(k): v
                                            for k, v in accs_user.items()},
            },
        },
        "grid_search": {
            "total_configs": total,
            "top_20": [{k: v for k, v in r.items() if k != "feature_indices"}
                       for r in sorted_all[:20]],
        },
    }
    with open(os.path.join(RESULTS_DIR, "resultados_busqueda_VQ.json"), "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # [6] Resumen final
    # ------------------------------------------------------------------
    elapsed = time.time() - t_inicio

    print("\n" + "=" * 70)
    print("  RESUMEN BUSQUEDA VQ")
    print("=" * 70)
    print(f"\n  Mejor config (de {total} evaluadas):")
    print(f"    Algoritmo:    {best_algo}")
    print(f"    Features:     {best_feat_name} ({len(best_feat_idx)}D)")
    print(f"    N centroides: {best_k}")
    print(f"    Accuracy val: {best['accuracy']*100:.2f}%")
    print(f"\n  Evaluacion final:")
    print(f"    {'N=74':<15} {acc_74*100:>9.2f}%")
    print(f"    {'N=47':<15} {acc_47*100:>9.2f}%")
    print(f"    {'LOO CV':<15} {acc_loo*100:>6.2f}% +/- {std_loo*100:.2f}%")
    print(f"\n  Tiempo total: {elapsed/60:.1f} minutos")
    print("=" * 70)


if __name__ == "__main__":
    main()
