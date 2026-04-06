"""
=============================================================================
IMPLEMENTACION VQ: Clasificador de digitos por Vector Quantization
=============================================================================
Un codebook (KMeans) por digito, clasificacion por minima distorsion media.
Baseline 7D del extractor local (mismo que busqueda_VQ.py y pipeline HMM).

Escenarios: N=74, N=47, y Leave-One-User-Out Cross-Validation.

Uso:
  python3 implementacion_VQ.py
=============================================================================
"""

import os
import sys
import re
import glob
import json
import time
import warnings

import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# -- Añadir el extractor local al path --
SCRIPT_DIR_SETUP = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT_SETUP = os.path.dirname(os.path.dirname(SCRIPT_DIR_SETUP))
EXTRACTOR_DIR = os.path.join(
    REPO_ROOT_SETUP, "Extractores_adaptados", "Extractores", "Extractor Local"
)
sys.path.insert(0, EXTRACTOR_DIR)
from extract_local_features import get_features  # noqa: E402

# =============================================================================
# CONFIGURACION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DB_PATH = os.path.join(REPO_ROOT, "e-BioDigit_DB", "e-BioDigit_DB")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "resultados_VQ")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_TRAIN_74 = 74
N_TRAIN_47 = 47

N_CENTROIDS = 32

# Subset 7D del extractor local (23 cols): x, y, dx, dy, v, sin(angle), cos(angle)
# Analogo al baseline original (pos_vel_ang) pero sourcing del extractor comun.
BASELINE_FEATURE_INDICES = [0, 1, 7, 8, 4, 19, 20]
BASELINE_FEATURE_NAMES = "x, y, dx, dy, v, sin(angle), cos(angle)"


# =============================================================================
# Lectura de ficheros e-BioDigit
# =============================================================================

def read_ebiodigit_file(filepath):
    """Lee un fichero e-BioDigit y devuelve x, y, t, presion."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    n = int(lines[0])
    data = []
    for line in lines[1:n + 1]:
        vals = line.split()
        x, y, t, p = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
        data.append([x, y, t, p])
    arr = np.array(data, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def parse_label_from_filename(filepath):
    """Extrae user_id, digit, sample_id del nombre de fichero."""
    name = os.path.basename(filepath)
    match = re.search(r"u(\d+)_digit_(\d)_(\d+)\.txt", name)
    if not match:
        raise ValueError(f"Nombre no reconocido: {name}")
    user_id = int(match.group(1))
    digit = int(match.group(2))
    sample_id = int(match.group(3))
    return user_id, digit, sample_id


# =============================================================================
# Preprocesado
# =============================================================================

def preprocess_trace(x, y, t):
    """Centra, escala y normaliza el trazado."""
    t = t - t[0]
    x = x - np.mean(x)
    y = y - np.mean(y)
    scale = max(np.max(np.abs(x)), np.max(np.abs(y)), 1e-8)
    x = x / scale
    y = y / scale
    return x, y, t


def compute_local_features(x, y, presion):
    """Calcula features locales (7D) usando el extractor local.

    Devuelve subset [x, y, dx, dy, v, sin(angle), cos(angle)] — el mismo que el
    baseline original pero sourcing del extractor comun con busqueda_VQ y HMM.
    """
    if presion is None:
        presion = np.full_like(x, 255.0)
    feats = get_features(x, y, presion, zscore=False)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats[:, BASELINE_FEATURE_INDICES]


# =============================================================================
# Carga de datos
# =============================================================================

class Sample:
    __slots__ = ("user_id", "digit", "sample_id", "features")

    def __init__(self, user_id, digit, sample_id, features):
        self.user_id = user_id
        self.digit = digit
        self.sample_id = sample_id
        self.features = features


def load_dataset(db_path):
    """Carga todas las muestras de e-BioDigit.

    Returns:
        samples: lista de Sample
        user_ids: lista de user_ids ordenados numericamente
    """
    txt_files = glob.glob(os.path.join(db_path, "**", "*.txt"), recursive=True)
    print(f"  Ficheros encontrados: {len(txt_files)}")

    samples = []
    for fp in txt_files:
        try:
            user_id, digit, sample_id = parse_label_from_filename(fp)
            x, y, t, p = read_ebiodigit_file(fp)
            x, y, t = preprocess_trace(x, y, t)
            feats = compute_local_features(x, y, p)
            if len(feats) > 0:
                samples.append(Sample(user_id, digit, sample_id, feats))
        except Exception as e:
            print(f"  Saltando {fp}: {e}")

    user_ids = sorted(set(s.user_id for s in samples))
    print(f"  Muestras cargadas: {len(samples)}")
    print(f"  Usuarios: {len(user_ids)}")
    return samples, user_ids


def split_by_users(samples, user_ids, n_train_users):
    """Divide muestras en train/test por usuario.

    Args:
        samples: lista de Sample
        user_ids: lista ordenada de user_ids
        n_train_users: cuantos usuarios van a train

    Returns:
        train_samples, test_samples
    """
    train_user_set = set(user_ids[:n_train_users])
    test_user_set = set(user_ids[n_train_users:])

    train = [s for s in samples if s.user_id in train_user_set]
    test = [s for s in samples if s.user_id in test_user_set]

    # Verificar que no hay solapamiento
    train_uids = set(s.user_id for s in train)
    test_uids = set(s.user_id for s in test)
    assert train_uids.isdisjoint(test_uids), "ERROR: solapamiento train/test"

    print(f"  Train: {len(train_user_set)} usuarios, {len(train)} muestras")
    print(f"  Test:  {len(test_user_set)} usuarios, {len(test)} muestras")
    return train, test


# =============================================================================
# Modelo VQ — 1 codebook por digito
# =============================================================================

class VQDigitClassifier:
    def __init__(self, n_centroids=32, random_state=42):
        self.n_centroids = n_centroids
        self.random_state = random_state
        self.codebooks = {}

    def _train(self, train_samples, verbose=True):
        """Entrena un codebook (KMeans) por digito usando SOLO datos de train."""
        by_digit = defaultdict(list)
        for s in train_samples:
            by_digit[s.digit].append(s.features)

        for digit in range(10):
            if digit not in by_digit:
                if verbose:
                    print(f"    AVISO: no hay datos de entrenamiento para digito {digit}")
                continue
            X = np.vstack(by_digit[digit])
            k = min(self.n_centroids, len(X))
            km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                        max_iter=300, random_state=self.random_state)
            km.fit(X)
            self.codebooks[digit] = km.cluster_centers_
            if verbose:
                print(f"    Digito {digit}: {len(X)} vectores -> {k} centroides")
        return self

    def fit(self, train_samples):
        return self._train(train_samples, verbose=True)

    def fit_quiet(self, train_samples):
        return self._train(train_samples, verbose=False)

    def distortion(self, features, codebook):
        """Distorsion media: distancia al centroide mas cercano, promediada."""
        dists = np.sum((features[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
        return np.mean(np.min(dists, axis=1))

    def predict_one(self, features):
        """Clasifica una muestra por minima distorsion."""
        scores = {d: self.distortion(features, cb)
                  for d, cb in self.codebooks.items()}
        pred = min(scores, key=scores.get)
        return pred, scores

    def predict(self, samples):
        """Clasifica un lote de muestras."""
        preds = []
        for s in samples:
            pred, _ = self.predict_one(s.features)
            preds.append(pred)
        return np.array(preds)


# =============================================================================
# Evaluacion
# =============================================================================

def evaluar(model, test_samples, nombre=""):
    """Evalua el modelo en test_samples (datos NO vistos en entrenamiento)."""
    y_true = np.array([s.digit for s in test_samples])
    y_pred = model.predict(test_samples)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    print(f"  Accuracy: {acc * 100:.2f}%")
    return acc, cm, y_true, y_pred


def graficar_confusion(cm, titulo, filepath):
    """Guarda la matriz de confusion como imagen."""
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


# =============================================================================
# LOO Cross-Validation
# =============================================================================

def loo_cv(samples, user_ids, n_centroids=32):
    """Leave-One-User-Out CV: entrena con 92 usuarios, testea con 1, x93.

    Returns:
        acc_media, acc_std, accuracies_por_usuario, cm_acumulada
    """
    n_users = len(user_ids)
    # Index samples by user for fast lookup
    samples_by_user = defaultdict(list)
    for s in samples:
        samples_by_user[s.user_id].append(s)

    accuracies = {}
    cm_acumulada = np.zeros((10, 10), dtype=int)

    for i, test_uid in enumerate(user_ids):
        # Train: all users except test_uid
        train_samples = []
        for uid in user_ids:
            if uid != test_uid:
                train_samples.extend(samples_by_user[uid])

        test_samples = samples_by_user[test_uid]

        # Train model
        model = VQDigitClassifier(n_centroids=n_centroids, random_state=42)
        model.fit_quiet(train_samples)

        # Evaluate
        y_true = np.array([s.digit for s in test_samples])
        y_pred = model.predict(test_samples)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

        accuracies[test_uid] = acc
        cm_acumulada += cm

        acc_parcial = np.mean(list(accuracies.values()))
        print(f"    [{i+1:2d}/{n_users}] Usuario {test_uid}: "
              f"{acc*100:5.1f}%  (media parcial: {acc_parcial*100:.1f}%)",
              flush=True)

    acc_media = np.mean(list(accuracies.values()))
    acc_std = np.std(list(accuracies.values()))
    return acc_media, acc_std, accuracies, cm_acumulada


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_inicio = time.time()

    print("=" * 60)
    print("  CLASIFICADOR VQ - e-BioDigit")
    print("=" * 60, flush=True)

    # 1. Cargar datos
    print("\n[1] Cargando base de datos...", flush=True)
    print(f"  Ruta: {DB_PATH}")
    samples, user_ids = load_dataset(DB_PATH)

    if len(samples) == 0:
        print("ERROR: No se cargaron muestras. Verifica la ruta.")
        return

    # 2. Escenario N=74
    print(f"\n[2] ESCENARIO N=74 (74 train / 19 test)")
    print("=" * 60, flush=True)

    train_74, test_74 = split_by_users(samples, user_ids, N_TRAIN_74)

    print("\n  Entrenando codebooks...", flush=True)
    model_74 = VQDigitClassifier(n_centroids=N_CENTROIDS, random_state=42)
    model_74.fit(train_74)

    print("\n  Evaluando en test (usuarios NO vistos en entrenamiento)...",
          flush=True)
    acc_74, cm_74, _, _ = evaluar(model_74, test_74, "N=74")

    graficar_confusion(
        cm_74,
        f"VQ N=74 — Accuracy: {acc_74 * 100:.2f}%",
        os.path.join(RESULTS_DIR, "confusion_VQ_N74.png"))

    # 3. Escenario N=47
    print(f"\n[3] ESCENARIO N=47 (47 train / 46 test)")
    print("=" * 60, flush=True)

    train_47, test_47 = split_by_users(samples, user_ids, N_TRAIN_47)

    print("\n  Entrenando codebooks...", flush=True)
    model_47 = VQDigitClassifier(n_centroids=N_CENTROIDS, random_state=42)
    model_47.fit(train_47)

    print("\n  Evaluando en test (usuarios NO vistos en entrenamiento)...",
          flush=True)
    acc_47, cm_47, _, _ = evaluar(model_47, test_47, "N=47")

    graficar_confusion(
        cm_47,
        f"VQ N=47 — Accuracy: {acc_47 * 100:.2f}%",
        os.path.join(RESULTS_DIR, "confusion_VQ_N47.png"))

    # 4. LOO Cross-Validation
    print(f"\n[4] LEAVE-ONE-USER-OUT CROSS-VALIDATION (93 folds)")
    print("=" * 60, flush=True)

    t_loo = time.time()
    acc_loo, std_loo, accs_por_user, cm_loo = loo_cv(
        samples, user_ids, n_centroids=N_CENTROIDS)
    t_loo = time.time() - t_loo

    print(f"\n  LOO CV: {acc_loo * 100:.2f}% +/- {std_loo * 100:.2f}%")
    print(f"  Tiempo LOO: {t_loo:.1f}s")

    graficar_confusion(
        cm_loo,
        f"VQ LOO CV — Accuracy: {acc_loo * 100:.2f}% +/- {std_loo * 100:.2f}%",
        os.path.join(RESULTS_DIR, "confusion_VQ_LOO.png"))

    # 5. Guardar resultados
    resumen = {
        "modelo": "VQ (1 codebook KMeans por digito)",
        "n_centroids": N_CENTROIDS,
        "extractor": "local (get_features 23 cols, subset 7D)",
        "features_indices": BASELINE_FEATURE_INDICES,
        "features": f"{BASELINE_FEATURE_NAMES} (7D)",
        "N74": {
            "accuracy": acc_74,
            "train_users": len(user_ids[:N_TRAIN_74]),
            "test_users": len(user_ids[N_TRAIN_74:]),
            "train_samples": len(train_74),
            "test_samples": len(test_74),
        },
        "N47": {
            "accuracy": acc_47,
            "train_users": len(user_ids[:N_TRAIN_47]),
            "test_users": len(user_ids[N_TRAIN_47:]),
            "train_samples": len(train_47),
            "test_samples": len(test_47),
        },
        "LOO": {
            "accuracy_media": acc_loo,
            "accuracy_std": std_loo,
            "n_folds": len(user_ids),
            "accuracies_por_usuario": {str(k): v for k, v in accs_por_user.items()},
        },
    }
    results_path = os.path.join(RESULTS_DIR, "resultados_VQ.json")
    with open(results_path, "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultados guardados en {results_path}")

    # 6. Resumen
    elapsed = time.time() - t_inicio

    print("\n" + "=" * 60)
    print("  RESUMEN VQ")
    print("=" * 60)
    print(f"\n  Modelo: 1 codebook KMeans ({N_CENTROIDS} centroides) por digito")
    print(f"  Features: 7D ({BASELINE_FEATURE_NAMES}) — extractor local")
    print(f"\n  {'Escenario':<15} {'Accuracy':>10}")
    print("  " + "-" * 27)
    print(f"  {'N=74':<15} {acc_74 * 100:>9.2f}%")
    print(f"  {'N=47':<15} {acc_47 * 100:>9.2f}%")
    print(f"  {'LOO CV':<15} {acc_loo * 100:>6.2f}% +/- {std_loo * 100:.2f}%")
    print(f"\n  Tiempo total: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
