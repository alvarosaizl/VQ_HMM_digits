"""
=============================================================================
CLASIFICADOR DE DIGITOS MANUSCRITOS (0-9) MEDIANTE HMMs
=============================================================================
Asignatura: ASMI - 4o Telecomunicaciones
Base de datos: e-BioDigit (93 usuarios, dígitos dibujados con el dedo)

Este script implementa un sistema completo de clasificación de dígitos
usando Modelos Ocultos de Markov (HMM). Cada dígito tiene su propio HMM
y la clasificación se hace por máxima verosimilitud.

Pipeline:
  1. Carga de la base de datos
  2. Preprocesado (centrado, escalado, remuestreo, suavizado)
  3. Extracción de características locales (cinemáticas)
  4. Entrenamiento de 10 HMMs (uno por dígito)
  5. Clasificación por argmax de log-verosimilitud
  6. Evaluación: accuracy, matrices de confusión, comparativas
  7. Visualización Viterbi: estados del HMM sobre el trazado

Uso:
  python clasificador_digitos.py
=============================================================================
"""

import sys
import os
import re
import time
import json
import warnings
import pickle
from collections import defaultdict

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Silenciar warnings de convergencia de hmmlearn
warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore")
import logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

# -- Añadir el extractor local al path --
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EXTRACTOR_DIR = os.path.join(
    REPO_ROOT, "Extractores_adaptados", "Extractores", "Extractor Local"
)
sys.path.insert(0, EXTRACTOR_DIR)
from extract_local_features import get_features  # noqa: E402


# =============================================================================
# CONFIGURACION
# =============================================================================

# Ruta a la base de datos e-BioDigit
DB_PATH = os.path.join(REPO_ROOT, "e-BioDigit_DB", "e-BioDigit_DB")

# Directorio donde se guardan los resultados (gráficas, modelos, JSON)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "resultados")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -- Subconjuntos de características --
# De las 23 características locales, seleccionamos subconjuntos de distinto
# tamaño para estudiar el efecto de la dimensionalidad.
#
# Indices de las 23 features del extractor local:
#   0=x, 1=y, 3=theta, 4=v, 5=rho, 6=a, 7=dx, 8=dy,
#   10=dtheta, 11=dv, 12=drho, 13=da, 14=ddx, 15=ddy,
#   16=rminmax_v, 17=angle, 18=dangle, 19=sin(angle),
#   20=cos(angle), 21=lewiratio5, 22=lewiratio7
FEATURE_SUBSETS = {
    "min":  [7, 8, 19, 20, 10, 4, 5],                          # 7 features
    "med":  [7, 8, 19, 20, 10, 4, 5, 6, 11, 21, 0, 1],        # 12 features
    "full": [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13,          # 21 features
             14, 15, 16, 17, 18, 19, 20, 21, 22],
}

# -- Escenarios de evaluación --
# Los usuarios se ordenan y se dividen en train/test
N_TRAIN_74 = 74  # 74 train, 19 test
N_TRAIN_47 = 47  # 47 train, 46 test


# =============================================================================
# PARTE 1: CARGA DE DATOS
# =============================================================================

# Regex para parsear los nombres de fichero: u{id}_digit_{d}_{sample}.txt
_FNAME_RE = re.compile(r"u(\d+)_digit_(\d+)_(\d+)\.txt")


def cargar_muestra(filepath):
    """Lee un fichero .txt de e-BioDigit.

    Formato del fichero:
      - Linea 1: N (numero de puntos)
      - Lineas 2..N+1: X Y timestamp presion

    Devuelve: (x, y, timestamp, presion) como arrays numpy
    """
    data = np.loadtxt(filepath, skiprows=1)
    x = data[:, 0].astype(np.float64)
    y = data[:, 1].astype(np.float64)
    timestamp = data[:, 2].astype(np.float64)
    presion = data[:, 3].astype(np.float64)
    return x, y, timestamp, presion


def cargar_base_datos(db_path):
    """Carga toda la base de datos e-BioDigit.

    Estructura de la BD:
      e-BioDigit_DB/
        {user_id}/
          session_1/
            u{id}_digit_{d}_{sample}.txt
          session_2/
            ...

    Devuelve:
      db: diccionario {user_id: {digito: {sesion: [muestras]}}}
      user_ids: lista ordenada de IDs de usuario
    """
    db = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    errores = 0

    # Listar las carpetas de usuarios
    user_dirs = sorted(
        d for d in os.listdir(db_path)
        if os.path.isdir(os.path.join(db_path, d))
    )

    for uid in user_dirs:
        for sess_idx, sess_name in enumerate(["session_1", "session_2"], start=1):
            sess_dir = os.path.join(db_path, uid, sess_name)
            if not os.path.isdir(sess_dir):
                continue

            for fname in sorted(os.listdir(sess_dir)):
                m = _FNAME_RE.match(fname)
                if not m:
                    continue
                digito = int(m.group(2))
                fpath = os.path.join(sess_dir, fname)
                try:
                    x, y, ts, p = cargar_muestra(fpath)
                    db[uid][digito][sess_idx].append({
                        "x": x, "y": y, "timestamp": ts,
                        "presion": p, "filepath": fpath,
                    })
                except Exception:
                    errores += 1

    if errores > 0:
        print(f"  [!] {errores} ficheros con errores al cargar")

    user_ids = sorted(db.keys(), key=lambda x: int(x))
    return dict(db), user_ids


def iterar_muestras(db, user_ids):
    """Generador que recorre todas las muestras de los usuarios indicados.

    Yield: (user_id, digito, sesion, muestra_dict)
    """
    for uid in user_ids:
        if uid not in db:
            continue
        for digito in range(10):
            if digito not in db[uid]:
                continue
            for sesion in [1, 2]:
                if sesion not in db[uid][digito]:
                    continue
                for muestra in db[uid][digito][sesion]:
                    yield uid, digito, sesion, muestra


# =============================================================================
# PARTE 2: PREPROCESADO
# =============================================================================

def preprocesar(x, y, timestamp, n_resample=80, suavizar=True,
                savgol_ventana=7, savgol_orden=3):
    """Pipeline de preprocesado para una muestra.

    Pasos:
      1. Resetear timestamp a t=0
      2. Suavizado Savitzky-Golay (opcional, reduce ruido del sensor)
      3. Remuestreo equidistante en longitud de arco
      4. Centrado (restar media de X e Y)
      5. Escalado a [-1, 1] manteniendo proporcion

    Args:
        x, y, timestamp: coordenadas y tiempos en bruto
        n_resample: numero de puntos tras remuestrear (0 = no remuestrear)
        suavizar: si True, aplica filtro Savitzky-Golay
        savgol_ventana: tamaño de ventana del filtro
        savgol_orden: orden del polinomio del filtro

    Returns:
        x, y, timestamp: arrays preprocesados
    """
    # 1. Resetear timestamp
    timestamp = timestamp - timestamp[0]

    # 2. Suavizado (antes de remuestrear, sobre coordenadas originales)
    if suavizar and len(x) >= savgol_ventana:
        x = savgol_filter(x, savgol_ventana, savgol_orden)
        y = savgol_filter(y, savgol_ventana, savgol_orden)

    # 3. Remuestreo equidistante en longitud de arco
    if n_resample > 0 and len(x) >= 3:
        # Calcular longitud de arco acumulada
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])

        if s[-1] > 0:  # Solo si el trazado tiene longitud > 0
            s_norm = s / s[-1]  # Normalizar a [0, 1]

            # Eliminar puntos duplicados para interpolar
            mask = np.concatenate([[True], np.diff(s_norm) > 0])
            s_unico = s_norm[mask]
            if len(s_unico) >= 2:
                s_objetivo = np.linspace(0, 1, n_resample)
                fx = interp1d(s_unico, x[mask], kind="linear",
                              fill_value="extrapolate")
                fy = interp1d(s_unico, y[mask], kind="linear",
                              fill_value="extrapolate")
                ft = interp1d(s_unico, timestamp[mask], kind="linear",
                              fill_value="extrapolate")
                x, y, timestamp = fx(s_objetivo), fy(s_objetivo), ft(s_objetivo)

    # 4. Centrado
    x = x - np.mean(x)
    y = y - np.mean(y)

    # 5. Escalado preservando aspect ratio
    factor_escala = max(np.ptp(x), np.ptp(y))
    if factor_escala > 0:
        x = x / factor_escala
        y = y / factor_escala

    return x, y, timestamp


# =============================================================================
# PARTE 3: EXTRACCION DE CARACTERISTICAS
# =============================================================================

def extraer_features(x, y, presion=None, indices_features=None):
    """Extrae características locales (cinemáticas) de una muestra preprocesada.

    Usa el extractor de 23 features locales (dx, dy, velocidad, curvatura,
    aceleracion, angulos, etc.) y selecciona un subconjunto.

    Args:
        x, y: coordenadas preprocesadas (longitud T)
        presion: array de presion (o None -> se usa 255)
        indices_features: lista de indices a seleccionar (None = todas las 23)

    Returns:
        array (T, D) donde D = numero de features seleccionadas
    """
    if presion is None:
        presion = np.full_like(x, 255.0)

    # get_features devuelve (T, 23) - zscore=False porque normalizamos despues
    features = get_features(x, y, presion, zscore=False)

    # Reemplazar NaN/Inf por 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if indices_features is not None:
        features = features[:, indices_features]

    return features


class NormalizadorZScore:
    """Normalización Z-Score ajustada sobre datos de entrenamiento.

    Calcula media y desviación típica sobre el conjunto de entrenamiento
    y las aplica tanto a train como a test, para evitar data leakage.
    """

    def __init__(self):
        self.media = None
        self.std = None

    def ajustar(self, secuencias):
        """Calcula media y std a partir de una lista de secuencias (T_i, D)."""
        todos = np.vstack(secuencias)
        self.media = np.mean(todos, axis=0)
        self.std = np.std(todos, axis=0, ddof=1)
        # Evitar division por cero
        self.std[self.std < 1e-10] = 1.0
        return self

    def transformar(self, secuencias):
        """Aplica normalización Z-Score: (x - media) / std."""
        return [(seq - self.media) / self.std for seq in secuencias]

    def ajustar_y_transformar(self, secuencias):
        """Ajusta y transforma en un solo paso."""
        self.ajustar(secuencias)
        return self.transformar(secuencias)


# =============================================================================
# PARTE 4: MODELO HMM
# =============================================================================

def _startprob_left_right(n_estados):
    """Probabilidad inicial para HMM left-right: siempre empezar en estado 0."""
    sp = np.zeros(n_estados)
    sp[0] = 1.0
    return sp


def _transmat_left_right(n_estados, prob_autolazo=0.6):
    """Matriz de transición left-right: autolazo + avance al siguiente estado.

    En un HMM left-right:
    - Desde cada estado solo se puede ir al mismo estado o al siguiente
    - El ultimo estado es absorbente (prob=1 de quedarse)
    - prob_autolazo controla cuanto tiempo se permanece en cada estado

    Ejemplo con 3 estados y prob_autolazo=0.6:
        [[0.6, 0.4, 0.0],
         [0.0, 0.6, 0.4],
         [0.0, 0.0, 1.0]]
    """
    A = np.zeros((n_estados, n_estados))
    for i in range(n_estados - 1):
        A[i, i] = prob_autolazo
        A[i, i + 1] = 1.0 - prob_autolazo
    A[n_estados - 1, n_estados - 1] = 1.0  # Estado final absorbente
    return A


def entrenar_hmm_digito(secuencias, n_estados=5, tipo_covarianza="diag",
                        n_iter=100, n_restarts=3, prob_autolazo=0.6,
                        semilla_base=0):
    """Entrena un GaussianHMM left-right para un dígito.

    Hace n_restarts intentos con semillas distintas y devuelve el mejor
    modelo (el que tiene mayor log-verosimilitud sobre los datos de
    entrenamiento).

    Args:
        secuencias: lista de arrays (T_i, D) para este dígito
        n_estados: numero de estados ocultos del HMM
        tipo_covarianza: 'spherical', 'diag', 'full' o 'tied'
        n_iter: iteraciones maximas del algoritmo EM
        n_restarts: cuantas veces reiniciar con semilla distinta
        prob_autolazo: probabilidad de permanecer en el mismo estado
        semilla_base: semilla inicial (para reproducibilidad)

    Returns:
        (mejor_modelo, mejor_score)
    """
    # Concatenar todas las secuencias y guardar sus longitudes
    longitudes = [len(s) for s in secuencias]
    X = np.concatenate(secuencias)

    mejor_score = -np.inf
    mejor_modelo = None

    for restart in range(n_restarts):
        semilla = semilla_base + restart
        modelo = hmm.GaussianHMM(
            n_components=n_estados,
            covariance_type=tipo_covarianza,
            n_iter=n_iter,
            random_state=semilla,
            init_params="mc",   # Inicializar medias y covarianzas automáticamente
            params="mct",       # Entrenar medias, covarianzas y transiciones
            verbose=False,
        )

        # Forzar topología left-right
        modelo.startprob_ = _startprob_left_right(n_estados)
        modelo.transmat_ = _transmat_left_right(n_estados, prob_autolazo)

        try:
            modelo.fit(X, longitudes)
            score = modelo.score(X, longitudes)
            if score > mejor_score:
                mejor_score = score
                mejor_modelo = modelo
        except (ValueError, np.linalg.LinAlgError):
            continue  # Algunos restarts pueden fallar

    return mejor_modelo, mejor_score


def entrenar_todos_los_digitos(secuencias_por_digito, n_estados=5,
                                tipo_covarianza="diag", n_iter=100,
                                n_restarts=3, prob_autolazo=0.6, verbose=True):
    """Entrena un HMM por cada dígito (0-9).

    Args:
        secuencias_por_digito: dict {digito: [secuencias (T_i, D)]}

    Returns:
        modelos: dict {digito: GaussianHMM entrenado}
        scores: dict {digito: log-verosimilitud del mejor modelo}
    """
    modelos = {}
    scores = {}

    for digito in sorted(secuencias_por_digito.keys()):
        seqs = secuencias_por_digito[digito]
        if verbose:
            print(f"    Digito {digito}: {len(seqs)} secuencias, "
                  f"{n_estados} estados, cov={tipo_covarianza}...", end=" ")
        modelo, score = entrenar_hmm_digito(
            seqs, n_estados=n_estados, tipo_covarianza=tipo_covarianza,
            n_iter=n_iter, n_restarts=n_restarts, prob_autolazo=prob_autolazo,
            semilla_base=digito * 100,
        )
        modelos[digito] = modelo
        scores[digito] = score
        if verbose:
            print(f"LL={score:.1f}")

    return modelos, scores


# =============================================================================
# PARTE 5: CLASIFICACION
# =============================================================================

def clasificar(modelos, secuencia_features):
    """Clasifica una secuencia de features.

    Calcula la log-verosimilitud de la secuencia bajo cada uno de los
    10 modelos HMM y devuelve el dígito con mayor puntuacion.

    Args:
        modelos: dict {digito: GaussianHMM entrenado}
        secuencia_features: array (T, D)

    Returns:
        (digito_predicho, dict_puntuaciones)
    """
    puntuaciones = {}
    for digito, modelo in modelos.items():
        if modelo is None:
            puntuaciones[digito] = -np.inf
            continue
        try:
            puntuaciones[digito] = modelo.score(secuencia_features)
        except (ValueError, np.linalg.LinAlgError):
            puntuaciones[digito] = -np.inf
    return max(puntuaciones, key=puntuaciones.get), puntuaciones


def clasificar_lote(modelos, secuencias, etiquetas_reales):
    """Clasifica un conjunto de secuencias y calcula la accuracy.

    Args:
        modelos: dict {digito: GaussianHMM}
        secuencias: lista de arrays (T_i, D)
        etiquetas_reales: lista de digitos verdaderos

    Returns:
        predicciones, todas_las_puntuaciones, accuracy
    """
    predicciones = []
    todas_puntuaciones = []
    correctas = 0

    for seq, etiqueta in zip(secuencias, etiquetas_reales):
        pred, puntuaciones = clasificar(modelos, seq)
        predicciones.append(pred)
        todas_puntuaciones.append(puntuaciones)
        if pred == etiqueta:
            correctas += 1

    accuracy = correctas / len(etiquetas_reales) if etiquetas_reales else 0.0
    return predicciones, todas_puntuaciones, accuracy


# =============================================================================
# PARTE 6: PREPARACION DE DATOS Y ESCENARIOS
# =============================================================================

def preparar_datos(db, user_ids, indices_features, n_resample=80,
                   suavizar=True):
    """Preprocesa y extrae features para un conjunto de usuarios.

    Returns:
        secuencias_por_digito: dict {digito: [arrays (T, D)]}
        todas_secuencias: lista plana de arrays
        todas_etiquetas: lista plana de digitos
    """
    secuencias_por_digito = defaultdict(list)
    todas_secuencias = []
    todas_etiquetas = []

    for uid, digito, sesion, muestra in iterar_muestras(db, user_ids):
        x, y, ts = preprocesar(
            muestra["x"], muestra["y"], muestra["timestamp"],
            n_resample=n_resample, suavizar=suavizar,
        )
        feats = extraer_features(x, y, muestra["presion"],
                                 indices_features=indices_features)
        secuencias_por_digito[digito].append(feats)
        todas_secuencias.append(feats)
        todas_etiquetas.append(digito)

    return secuencias_por_digito, todas_secuencias, todas_etiquetas


def ejecutar_escenario(db, usuarios_train, usuarios_test, config,
                       nombre="escenario", verbose=True):
    """Ejecuta un escenario completo de entrenamiento y evaluación.

    Args:
        db: base de datos cargada
        usuarios_train: lista de IDs para entrenamiento
        usuarios_test: lista de IDs para test
        config: diccionario con hiperparámetros:
            - n_estados, tipo_covarianza, n_iter, n_restarts
            - prob_autolazo, subset_features, n_resample, suavizar
        nombre: nombre del escenario (para gráficas)
        verbose: si True, imprime progreso

    Returns:
        dict con accuracy, modelos, normalizer, predicciones, etc.
    """
    indices_features = FEATURE_SUBSETS.get(
        config.get("subset_features", "med"), FEATURE_SUBSETS["med"])
    n_estados = config.get("n_estados", 5)
    tipo_cov = config.get("tipo_covarianza", "diag")
    n_iter = config.get("n_iter", 100)
    n_restarts = config.get("n_restarts", 3)
    prob_autolazo = config.get("prob_autolazo", 0.6)
    n_resample = config.get("n_resample", 80)
    suavizar = config.get("suavizar", True)

    if verbose:
        print(f"\n  Escenario: {nombre}")
        print(f"    Train: {len(usuarios_train)} usuarios, "
              f"Test: {len(usuarios_test)} usuarios")
        print(f"    Estados={n_estados}, Cov={tipo_cov}, "
              f"Features={config.get('subset_features', 'med')}")

    t0 = time.time()

    # 1. Preparar datos
    if verbose:
        print("    [1/5] Preparando datos de entrenamiento...")
    train_por_digito, train_seqs, train_labels = preparar_datos(
        db, usuarios_train, indices_features, n_resample, suavizar)

    if verbose:
        print("    [2/5] Preparando datos de test...")
    _, test_seqs, test_labels = preparar_datos(
        db, usuarios_test, indices_features, n_resample, suavizar)

    # 2. Normalización Z-Score (ajustar SOLO con train)
    if verbose:
        print("    [3/5] Normalizando features...")
    normalizador = NormalizadorZScore()
    train_norm = normalizador.ajustar_y_transformar(train_seqs)
    test_norm = normalizador.transformar(test_seqs)

    # Reorganizar train normalizado por dígito
    train_norm_por_digito = defaultdict(list)
    for seq, label in zip(train_norm, train_labels):
        train_norm_por_digito[label].append(seq)

    # 3. Entrenar HMMs
    if verbose:
        print("    [4/5] Entrenando HMMs...")
    modelos, scores_train = entrenar_todos_los_digitos(
        train_norm_por_digito, n_estados=n_estados,
        tipo_covarianza=tipo_cov, n_iter=n_iter,
        n_restarts=n_restarts, prob_autolazo=prob_autolazo,
        verbose=verbose,
    )

    # 4. Clasificar test
    if verbose:
        print("    [5/5] Clasificando...")
    predicciones, todas_puntuaciones, accuracy = clasificar_lote(
        modelos, test_norm, test_labels)

    # 5. Métricas
    cm = sk_confusion_matrix(test_labels, predicciones, labels=list(range(10)))

    elapsed = time.time() - t0
    if verbose:
        print(f"\n    >>> Accuracy: {accuracy * 100:.2f}% ({elapsed:.1f}s)")

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "modelos": modelos,
        "normalizador": normalizador,
        "predicciones": predicciones,
        "puntuaciones": todas_puntuaciones,
        "test_labels": test_labels,
        "tiempo": elapsed,
    }


# =============================================================================
# PARTE 7: GRAFICAS Y VISUALIZACIONES
# =============================================================================

def graficar_confusion(cm, titulo, filepath):
    """Genera y guarda una matriz de confusión como imagen."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10), ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"      Guardado: {filepath}")


def graficar_comparativa(resultados, parametro, filepath):
    """Genera gráfica de barras comparando accuracy para distintos valores
    de un hiperparámetro.

    Args:
        resultados: lista de dicts con 'valor' y 'accuracy'
        parametro: nombre del hiperparámetro
        filepath: ruta para guardar la imagen
    """
    valores = [str(r["valor"]) for r in resultados]
    accuracies = [r["accuracy"] * 100 for r in resultados]

    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(valores, accuracies, color="steelblue", edgecolor="navy")
    ax.set_xlabel(parametro)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy vs {parametro}")

    # Poner el valor encima de cada barra
    for barra, val in zip(barras, accuracies):
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, max(accuracies) + 10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"      Guardado: {filepath}")


def graficar_comparativa_global(todos_resultados, filepath):
    """Genera una tabla resumen visual con todos los experimentos.

    Args:
        todos_resultados: dict {nombre_experimento: accuracy}
    """
    nombres = list(todos_resultados.keys())
    accuracies = [v * 100 for v in todos_resultados.values()]

    fig, ax = plt.subplots(figsize=(12, max(5, len(nombres) * 0.4)))
    colores = plt.cm.RdYlGn(np.array(accuracies) / 100)
    barras = ax.barh(nombres, accuracies, color=colores, edgecolor="gray")

    for barra, val in zip(barras, accuracies):
        ax.text(barra.get_width() + 0.5, barra.get_y() + barra.get_height() / 2,
                f"{val:.1f}%", ha="left", va="center", fontsize=9)

    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Comparativa de todos los experimentos")
    ax.set_xlim(0, max(accuracies) + 10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"      Guardado: {filepath}")


def graficar_viterbi(db, modelos, normalizador, user_ids, indices_features,
                     n_resample=80, suavizar=True, directorio_salida=None):
    """Genera gráficas de Viterbi: colorea el trazado por estado HMM.

    Para cada dígito (0-9), toma una muestra del primer usuario de test,
    ejecuta el algoritmo de Viterbi para obtener la secuencia de estados
    más probable, y colorea cada punto del trazado según su estado.

    Esto permite ver cómo el HMM ha segmentado el trazado en fases.
    """
    if directorio_salida:
        os.makedirs(directorio_salida, exist_ok=True)

    uid = user_ids[0]
    digitos_hechos = set()

    for _, digito, sesion, muestra in iterar_muestras(db, [uid]):
        if sesion != 1 or digito in digitos_hechos:
            continue

        # Preprocesar
        x, y, ts = preprocesar(muestra["x"], muestra["y"],
                                muestra["timestamp"],
                                n_resample=n_resample, suavizar=suavizar)

        # Extraer y normalizar features
        feats = extraer_features(x, y, muestra["presion"],
                                 indices_features=indices_features)
        feats_norm = (feats - normalizador.media) / normalizador.std

        modelo = modelos.get(digito)
        if modelo is None:
            continue

        try:
            # Decodificación Viterbi: secuencia de estados más probable
            _, estados = modelo.decode(feats_norm)
        except Exception:
            continue

        # Crear gráfica
        n_estados_unicos = len(np.unique(estados))
        cmap = plt.cm.get_cmap("tab10")

        fig, ax = plt.subplots(figsize=(5, 5))

        # Dibujar cada punto coloreado por su estado
        for s in range(max(estados) + 1):
            mascara = estados == s
            ax.scatter(x[mascara], -y[mascara], c=[cmap(s)], s=30,
                       label=f"Estado {s}", zorder=3)

        # Línea del trazado
        ax.plot(x, -y, "k-", alpha=0.2, linewidth=0.5, zorder=1)

        ax.set_title(f"Digito {digito} - Decodificacion Viterbi (Usuario {uid})")
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        if directorio_salida:
            ruta = os.path.join(directorio_salida,
                                f"viterbi_digito{digito}.png")
            fig.savefig(ruta, dpi=150)
            print(f"      Guardado: {ruta}")
        plt.close(fig)

        digitos_hechos.add(digito)


def graficar_matrices_transicion(modelos, filepath):
    """Visualiza las matrices de transición aprendidas para los 10 dígitos."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for digito in range(10):
        ax = axes[digito // 5][digito % 5]
        modelo = modelos.get(digito)
        if modelo is None:
            ax.set_title(f"Digito {digito}\n(sin modelo)")
            continue

        A = modelo.transmat_
        sns.heatmap(A, annot=True, fmt=".2f", cmap="YlOrRd",
                    vmin=0, vmax=1, ax=ax,
                    xticklabels=range(A.shape[0]),
                    yticklabels=range(A.shape[0]))
        ax.set_title(f"Digito {digito}")
        ax.set_xlabel("Hacia")
        ax.set_ylabel("Desde")

    plt.suptitle("Matrices de Transicion Aprendidas", fontsize=14)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"      Guardado: {filepath}")


def graficar_duracion_estados(modelos, filepath):
    """Gráfica de la duración esperada en cada estado por dígito.

    La duración esperada en un estado i es 1 / (1 - A[i,i]),
    donde A[i,i] es la probabilidad de autolazo.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for digito in range(10):
        modelo = modelos.get(digito)
        if modelo is None:
            continue

        A = modelo.transmat_
        self_probs = np.diag(A)
        duraciones = 1.0 / (1.0 - self_probs + 1e-10)
        ax.plot(range(len(duraciones)), duraciones, "o-",
                label=f"Digito {digito}")

    ax.set_xlabel("Estado")
    ax.set_ylabel("Duracion esperada (frames)")
    ax.set_title("Duracion Esperada por Estado y Digito")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"      Guardado: {filepath}")


# =============================================================================
# PARTE 8: PIPELINE PRINCIPAL
# =============================================================================

def _entrenar_y_evaluar(train_norm_por_digito, test_norm, test_labels,
                        n_estados=5, tipo_covarianza="diag", n_iter=50,
                        n_restarts=2, prob_autolazo=0.6):
    """Entrena HMMs y evalua sobre datos ya preprocesados y normalizados.

    Esta funcion SOLO entrena y clasifica; no preprocesa ni normaliza.
    Esto permite reutilizar los datos preprocesados entre experimentos.
    """
    modelos, _ = entrenar_todos_los_digitos(
        train_norm_por_digito, n_estados=n_estados,
        tipo_covarianza=tipo_covarianza, n_iter=n_iter,
        n_restarts=n_restarts, prob_autolazo=prob_autolazo,
        verbose=False,
    )
    predicciones, puntuaciones, accuracy = clasificar_lote(
        modelos, test_norm, test_labels)
    cm = sk_confusion_matrix(test_labels, predicciones, labels=list(range(10)))
    return {
        "accuracy": accuracy, "confusion_matrix": cm, "modelos": modelos,
        "predicciones": predicciones, "puntuaciones": puntuaciones,
        "test_labels": test_labels,
    }


def main():
    t_inicio = time.time()

    print("=" * 70)
    print("  CLASIFICADOR DE DIGITOS MANUSCRITOS MEDIANTE HMMs")
    print("=" * 70, flush=True)

    # ------------------------------------------------------------------
    # PASO 1: Cargar base de datos
    # ------------------------------------------------------------------
    print("\n[PASO 1] Cargando base de datos e-BioDigit...", flush=True)
    db, user_ids = cargar_base_datos(DB_PATH)

    total = sum(1 for _ in iterar_muestras(db, user_ids))
    print(f"  Usuarios: {len(user_ids)}")
    print(f"  Muestras totales: {total}", flush=True)

    # Dividir usuarios en train/test
    train_74 = user_ids[:N_TRAIN_74]
    test_74 = user_ids[N_TRAIN_74:]
    train_47 = user_ids[:N_TRAIN_47]
    test_47 = user_ids[N_TRAIN_47:]

    # Diccionario para almacenar todos los resultados
    todos_resultados = {}
    resultados_busqueda = []

    # ------------------------------------------------------------------
    # PASO 2: Pre-computar datos para los 3 subconjuntos de features
    # ------------------------------------------------------------------
    # OPTIMIZACION CLAVE: preparar los datos UNA VEZ por subset de features
    # y reutilizarlos en todos los experimentos de hiperparametros.
    # Esto evita reprocesar ~6000 muestras en cada test.
    # ------------------------------------------------------------------
    print("\n[PASO 2] Preparando datos (una sola vez por subset)...", flush=True)

    datos_cache = {}  # {(subset, suavizar, n_resample): (train_norm_por_digito, test_norm, test_labels, normalizador)}

    def _preparar_cache(subset, suavizar, n_resample, train_users, test_users,
                        etiqueta=None):
        """Prepara y cachea datos para una configuracion de preprocesado."""
        clave = (subset, suavizar, n_resample, len(train_users))
        if clave in datos_cache:
            return datos_cache[clave]

        indices = FEATURE_SUBSETS[subset]
        if etiqueta:
            print(f"    Preparando: {etiqueta}...", end=" ", flush=True)

        _, train_seqs, train_labels = preparar_datos(
            db, train_users, indices, n_resample, suavizar)
        _, test_seqs, test_labels = preparar_datos(
            db, test_users, indices, n_resample, suavizar)

        normalizador = NormalizadorZScore()
        train_norm = normalizador.ajustar_y_transformar(train_seqs)
        test_norm = normalizador.transformar(test_seqs)

        train_norm_por_digito = defaultdict(list)
        for seq, label in zip(train_norm, train_labels):
            train_norm_por_digito[label].append(seq)

        result = (train_norm_por_digito, test_norm, test_labels, normalizador)
        datos_cache[clave] = result
        if etiqueta:
            print("OK", flush=True)
        return result

    # Pre-computar para el subset por defecto (med) con N=74
    print("  Pre-computando datos N=74 (subset med, con suavizado)...")
    med_train, med_test, med_labels, med_norm = _preparar_cache(
        "med", True, 80, train_74, test_74, "med(12D), suavizado, resample=80")

    # ------------------------------------------------------------------
    # PASO 3: Búsqueda de hiperparámetros (con datos pre-computados)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [PASO 3] BUSQUEDA DE HIPERPARAMETROS")
    print("=" * 70, flush=True)

    # --- 3.1: Número de estados ---
    print("\n  --- Numero de estados (n_estados) ---", flush=True)
    res_estados = []
    for n in [3, 5, 7, 10]:
        t0 = time.time()
        r = _entrenar_y_evaluar(med_train, med_test, med_labels,
                                n_estados=n, n_restarts=2, n_iter=50)
        dt = time.time() - t0
        res_estados.append({"valor": n, "accuracy": r["accuracy"]})
        todos_resultados[f"n_estados={n}"] = r["accuracy"]
        resultados_busqueda.append({"param": "n_estados", "valor": n,
                                    "accuracy": r["accuracy"]})
        print(f"    n_estados={n:>3d}: accuracy={r['accuracy']*100:5.1f}%  "
              f"({dt:.0f}s)", flush=True)

    graficar_comparativa(res_estados, "Numero de estados",
                         os.path.join(RESULTS_DIR, "comp_n_estados.png"))

    # --- 3.2: Tipo de covarianza ---
    print("\n  --- Tipo de covarianza ---", flush=True)
    res_cov = []
    for cov in ["spherical", "diag", "tied"]:
        t0 = time.time()
        r = _entrenar_y_evaluar(med_train, med_test, med_labels,
                                tipo_covarianza=cov, n_restarts=2, n_iter=50)
        dt = time.time() - t0
        res_cov.append({"valor": cov, "accuracy": r["accuracy"]})
        todos_resultados[f"covarianza={cov}"] = r["accuracy"]
        resultados_busqueda.append({"param": "covarianza", "valor": cov,
                                    "accuracy": r["accuracy"]})
        print(f"    covarianza={cov:>12s}: accuracy={r['accuracy']*100:5.1f}%  "
              f"({dt:.0f}s)", flush=True)

    graficar_comparativa(res_cov, "Tipo de covarianza",
                         os.path.join(RESULTS_DIR, "comp_covarianza.png"))

    # --- 3.3: Subconjunto de features ---
    print("\n  --- Subconjunto de features ---", flush=True)
    res_feats = []
    for subset in ["min", "med", "full"]:
        n_dim = len(FEATURE_SUBSETS[subset])
        t0 = time.time()
        tr, te, tl, _ = _preparar_cache(
            subset, True, 80, train_74, test_74, f"{subset}({n_dim}D)")
        r = _entrenar_y_evaluar(tr, te, tl, n_restarts=2, n_iter=50)
        dt = time.time() - t0
        etiq = f"{subset}({n_dim}D)"
        res_feats.append({"valor": etiq, "accuracy": r["accuracy"]})
        todos_resultados[f"features={etiq}"] = r["accuracy"]
        resultados_busqueda.append({"param": "features", "valor": etiq,
                                    "accuracy": r["accuracy"]})
        print(f"    features={etiq:>12s}: accuracy={r['accuracy']*100:5.1f}%  "
              f"({dt:.0f}s)", flush=True)

    graficar_comparativa(res_feats, "Subconjunto de features",
                         os.path.join(RESULTS_DIR, "comp_features.png"))

    # --- 3.4: Probabilidad de autolazo ---
    print("\n  --- Probabilidad de autolazo ---", flush=True)
    res_sp = []
    for sp in [0.4, 0.5, 0.6, 0.7, 0.8]:
        t0 = time.time()
        r = _entrenar_y_evaluar(med_train, med_test, med_labels,
                                prob_autolazo=sp, n_restarts=2, n_iter=50)
        dt = time.time() - t0
        res_sp.append({"valor": sp, "accuracy": r["accuracy"]})
        todos_resultados[f"prob_autolazo={sp}"] = r["accuracy"]
        resultados_busqueda.append({"param": "prob_autolazo", "valor": sp,
                                    "accuracy": r["accuracy"]})
        print(f"    prob_autolazo={sp:>5.1f}: accuracy={r['accuracy']*100:5.1f}%  "
              f"({dt:.0f}s)", flush=True)

    graficar_comparativa(res_sp, "Probabilidad de autolazo",
                         os.path.join(RESULTS_DIR, "comp_autolazo.png"))

    # --- 3.5: Efectos del preprocesado ---
    print("\n  --- Efectos del preprocesado ---", flush=True)
    preproc_configs = [
        ("con_todo",       True,  80),
        ("sin_suavizado",  False, 80),
        ("sin_resample",   True,  0),
        ("sin_nada",       False, 0),
    ]
    res_preproc = []
    for nombre, suav, nres in preproc_configs:
        t0 = time.time()
        tr, te, tl, _ = _preparar_cache(
            "med", suav, nres, train_74, test_74, nombre)
        r = _entrenar_y_evaluar(tr, te, tl, n_restarts=2, n_iter=50)
        dt = time.time() - t0
        res_preproc.append({"valor": nombre, "accuracy": r["accuracy"]})
        todos_resultados[f"preproc={nombre}"] = r["accuracy"]
        resultados_busqueda.append({"param": "preprocesado", "valor": nombre,
                                    "accuracy": r["accuracy"]})
        print(f"    preprocesado={nombre:>16s}: accuracy={r['accuracy']*100:5.1f}%  "
              f"({dt:.0f}s)", flush=True)

    graficar_comparativa(res_preproc, "Preprocesado",
                         os.path.join(RESULTS_DIR, "comp_preprocesado.png"))

    # Guardar resultados de búsqueda
    with open(os.path.join(RESULTS_DIR, "busqueda_hiperparametros.json"), "w") as f:
        json.dump(resultados_busqueda, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultados guardados en busqueda_hiperparametros.json", flush=True)

    # ------------------------------------------------------------------
    # PASO 4: Evaluación final con la mejor configuración
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [PASO 4] EVALUACION CON MEJOR CONFIGURACION")
    print("=" * 70, flush=True)

    # Usar los datos pre-computados de med + suavizado + resample=80
    # Mejor configuración: 7 estados, diag, 5 restarts, 100 iter EM
    print("\n  >>> Escenario N=74 (74 train, 19 test) <<<", flush=True)
    res_74 = _entrenar_y_evaluar(med_train, med_test, med_labels,
                                 n_estados=7, tipo_covarianza="diag",
                                 n_iter=100, n_restarts=5, prob_autolazo=0.6)
    res_74["normalizador"] = med_norm
    todos_resultados["MEJOR_N74"] = res_74["accuracy"]
    print(f"    Accuracy: {res_74['accuracy']*100:.2f}%", flush=True)

    graficar_confusion(
        res_74["confusion_matrix"], "Matriz de Confusion - N=74 (mejor config)",
        os.path.join(RESULTS_DIR, "confusion_N74.png"))

    print("\n  >>> Escenario N=47 (47 train, 46 test) <<<", flush=True)
    tr47, te47, tl47, norm47 = _preparar_cache(
        "med", True, 80, train_47, test_47, "N47 med(12D)")
    res_47 = _entrenar_y_evaluar(tr47, te47, tl47,
                                 n_estados=7, tipo_covarianza="diag",
                                 n_iter=100, n_restarts=5, prob_autolazo=0.6)
    todos_resultados["MEJOR_N47"] = res_47["accuracy"]
    print(f"    Accuracy: {res_47['accuracy']*100:.2f}%", flush=True)

    graficar_confusion(
        res_47["confusion_matrix"], "Matriz de Confusion - N=47 (mejor config)",
        os.path.join(RESULTS_DIR, "confusion_N47.png"))

    # Resumen
    print(f"\n  {'Escenario':<15} {'Accuracy':>10}")
    print("  " + "-" * 28)
    print(f"  {'N=74':<15} {res_74['accuracy'] * 100:>9.2f}%")
    print(f"  {'N=47':<15} {res_47['accuracy'] * 100:>9.2f}%", flush=True)

    # ------------------------------------------------------------------
    # PASO 5: Visualizaciones
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [PASO 5] VISUALIZACIONES")
    print("=" * 70, flush=True)

    indices_features = FEATURE_SUBSETS["med"]

    # 5.1 Gráficas de Viterbi
    print("\n  [5.1] Generando graficas de Viterbi...", flush=True)
    graficar_viterbi(
        db, res_74["modelos"], res_74["normalizador"],
        test_74, indices_features,
        n_resample=80, suavizar=True,
        directorio_salida=os.path.join(RESULTS_DIR, "viterbi"))

    # 5.2 Matrices de transición
    print("\n  [5.2] Matrices de transicion...", flush=True)
    graficar_matrices_transicion(
        res_74["modelos"],
        os.path.join(RESULTS_DIR, "matrices_transicion.png"))

    # 5.3 Duración de estados
    print("\n  [5.3] Duracion de estados...", flush=True)
    graficar_duracion_estados(
        res_74["modelos"],
        os.path.join(RESULTS_DIR, "duracion_estados.png"))

    # 5.4 Comparativa global
    print("\n  [5.4] Comparativa global de todos los experimentos...", flush=True)
    graficar_comparativa_global(
        todos_resultados,
        os.path.join(RESULTS_DIR, "comparativa_global.png"))

    # 5.5 Guardar modelos
    ruta_modelos = os.path.join(RESULTS_DIR, "modelos_mejor.pkl")
    with open(ruta_modelos, "wb") as f:
        pickle.dump({"modelos": res_74["modelos"],
                      "normalizador": res_74["normalizador"]}, f)
    print(f"\n  Modelos guardados en: {ruta_modelos}", flush=True)

    # ------------------------------------------------------------------
    # RESUMEN FINAL
    # ------------------------------------------------------------------
    elapsed = time.time() - t_inicio

    print("\n" + "=" * 70)
    print("  RESUMEN FINAL")
    print("=" * 70)
    print(f"\n  Tiempo total: {elapsed / 60:.1f} minutos")
    print(f"\n  Mejor accuracy N=74: {res_74['accuracy'] * 100:.2f}%")
    print(f"  Mejor accuracy N=47: {res_47['accuracy'] * 100:.2f}%")
    print(f"\n  Ficheros generados en: {RESULTS_DIR}/")

    for f in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, f)
        if os.path.isfile(fpath):
            print(f"    {f}")
        elif os.path.isdir(fpath):
            n_files = len(os.listdir(fpath))
            print(f"    {f}/ ({n_files} ficheros)")

    print("\n" + "=" * 70)
    print("  FIN")
    print("=" * 70)


if __name__ == "__main__":
    main()
