# Informe VQ: Clasificacion de Digitos Manuscritos con Vector Quantization

## Proyecto ASMI - e-BioDigit

---

## 1. Objetivo

Clasificar digitos manuscritos (0-9) a partir de trazados dinamicos capturados en una tableta digitalizadora, utilizando **Vector Quantization (VQ)** basada en codebooks. Se evaluan dos fases:

- **Baseline:** Un codebook KMeans (32 centroides) por digito con 7 features.
- **Optimizacion:** Busqueda exhaustiva sobre algoritmo (KMeans, LBG, MiniBatchKMeans), features (8 subconjuntos de 3D a 11D) y numero de centroides (8-256).

---

## 2. Base de datos

- **e-BioDigit:** 93 usuarios, 10 digitos, 2 sesiones x 4 repeticiones = 80 muestras por usuario.
- **Total:** 7440 muestras.
- **Escenarios de evaluacion:**
  - **N=74:** 74 usuarios train / 19 usuarios test.
  - **N=47:** 47 usuarios train / 46 usuarios test.
  - **LOO CV:** Leave-One-Out (93 folds).

---

## 3. Metodologia VQ

### 3.1 Principio

Se entrena un **codebook** (conjunto de centroides) por cada digito (0-9) a partir de los vectores de features locales de las muestras de entrenamiento. Para clasificar una muestra de test:

1. Se calculan las features locales del trazado.
2. Para cada codebook, se computa la **distorsion media**: distancia euclidea media entre cada vector de features y su centroide mas cercano.
3. Se asigna el digito cuyo codebook produce la **minima distorsion media**.

### 3.2 Algoritmos de cuantificacion

| Algoritmo | Descripcion | Inicializacion |
|-----------|-------------|----------------|
| **KMeans** | Algoritmo de Lloyd clasico | k-means++ (10 reintentos) |
| **LBG** | Linde-Buzo-Gray: particion iterativa | Centroide global + splitting |
| **MiniBatchKMeans** | KMeans con mini-batches (mas rapido) | k-means++ (10 reintentos) |

**LBG (Linde-Buzo-Gray):** Comienza con un unico centroide (media global), lo divide en dos por perturbacion, refina con Lloyd, y repite hasta alcanzar k centroides. Ventaja: construccion jerarquica del codebook. Desventaja: significativamente mas lento que KMeans para k grande.

---

## 4. Preprocesado

Pipeline aplicado a cada trazado:

1. **Centrado:** restar media de x e y.
2. **Escalado:** normalizar por el rango maximo (max |x|, max |y|).
3. **Normalizacion temporal:** t = t - t[0].

**Nota:** A diferencia del pipeline HMM, no se aplica suavizado ni remuestreo. El clasificador VQ opera directamente sobre los vectores de features punto a punto sin necesidad de secuencias de longitud fija.

---

## 5. Extraccion de features

Se computan **11 features locales** por punto del trazado (N-1 puntos, calculados sobre diferencias consecutivas):

| Indice | Feature | Descripcion |
|--------|---------|-------------|
| 0 | x | Posicion horizontal (centrada/escalada) |
| 1 | y | Posicion vertical (centrada/escalada) |
| 2 | vx | Velocidad horizontal |
| 3 | vy | Velocidad vertical |
| 4 | v | Modulo de velocidad |
| 5 | sin(a) | Seno del angulo de trazo |
| 6 | cos(a) | Coseno del angulo de trazo |
| 7 | ax | Aceleracion horizontal |
| 8 | ay | Aceleracion vertical |
| 9 | |a| | Modulo de aceleracion |
| 10 | dtheta | Curvatura (cambio de angulo) |

### 5.1 Subconjuntos evaluados

Se definieron 8 subconjuntos de features para la busqueda:

| Subconjunto | Dim | Features |
|-------------|-----|----------|
| `pos_vel_ang` | 7D | x, y, vx, vy, v, sin, cos |
| `vel_ang` | 5D | vx, vy, v, sin, cos |
| **`full`** | **11D** | **todas** |
| `vel_acc` | 6D | vx, vy, v, ax, ay, |a| |
| `ang_curv` | 4D | v, sin, cos, dtheta |
| `pos_ang_curv` | 5D | x, y, sin, cos, dtheta |
| `all_no_pos` | 9D | todas sin x, y |
| `vel_only` | 3D | vx, vy, v |

---

## 6. Baseline: KMeans con 32 centroides

### 6.1 Configuracion

| Parametro | Valor |
|-----------|-------|
| Algoritmo | KMeans (k-means++, 10 reintentos) |
| N centroides | 32 |
| Features | pos_vel_ang (7D): x, y, vx, vy, v, sin, cos |
| Clasificacion | argmin distorsion media |

### 6.2 Resultados baseline

| Escenario | Accuracy |
|-----------|----------|
| N=74 | **96.12%** |
| N=47 | **96.06%** |
| LOO CV | **95.69% +/- 5.27%** |

Resultados notablemente buenos para un baseline sin optimizacion. La diferencia entre N=74 y N=47 es minima (0.06pp), indicando alta robustez al tamanio de entrenamiento.

---

## 7. Busqueda de hiperparametros

### 7.1 Configuracion de la busqueda

- **Validacion:** 78 usuarios train / 15 usuarios validacion (misma particion que entregas HMM).
- **Grid:** 3 algoritmos x 8 feature sets x 6 centroides = **144 configuraciones**.
- **Criterio:** Accuracy de clasificacion sobre validacion.

### 7.2 Top 10 configuraciones

| Rank | Algoritmo | Features | k | Accuracy (val) | Tiempo |
|------|-----------|----------|---|----------------|--------|
| **1** | **KMeans** | **full (11D)** | **256** | **97.33%** | **58s** |
| 2 | LBG | full (11D) | 256 | 97.17% | 419s |
| 3 | LBG | pos_ang_curv (5D) | 64 | 97.17% | 38s |
| 4 | KMeans | full (11D) | 64 | 97.08% | 19s |
| 5 | KMeans | full (11D) | 128 | 97.08% | 33s |
| 6 | LBG | full (11D) | 64 | 97.08% | 92s |
| 7 | KMeans | pos_ang_curv (5D) | 256 | 97.00% | 47s |
| 8 | LBG | pos_ang_curv (5D) | 128 | 97.00% | 91s |
| 9 | LBG | pos_ang_curv (5D) | 256 | 97.00% | 204s |
| 10 | KMeans | full (11D) | 32 | 96.92% | 13s |

### 7.3 Analisis por algoritmo

| Algoritmo | Mejor accuracy | Config | Tiempo |
|-----------|----------------|--------|--------|
| **KMeans** | **97.33%** | full, k=256 | 58s |
| LBG | 97.17% | full, k=256 | 419s |
| MiniBatchKMeans | 96.75% | full, k=256 | 10s |

- **KMeans** obtiene la mejor accuracy con un coste computacional razonable.
- **LBG** alcanza resultados comparables pero es ~7x mas lento para k=256.
- **MiniBatchKMeans** es el mas rapido (~6x mas que KMeans) a cambio de una ligera perdida (-0.58pp). Buena opcion para prototipado rapido.

### 7.4 Analisis por subconjunto de features

| Subconjunto | Dim | Mejor accuracy | Config |
|-------------|-----|----------------|--------|
| **full** | **11D** | **97.33%** | KMeans, k=256 |
| **pos_ang_curv** | **5D** | **97.17%** | LBG, k=64 |
| pos_vel_ang | 7D | 96.25% | KMeans, k=128 |
| all_no_pos | 9D | 95.83% | KMeans, k=256 |
| vel_acc | 6D | 88.33% | KMeans, k=256 |
| ang_curv | 4D | 84.08% | KMeans, k=256 |
| vel_ang | 5D | 48.92% | KMeans, k=128 |
| vel_only | 3D | 47.08% | KMeans, k=256 |

**Observaciones:**

1. **`full` (11D)** es el mejor subconjunto global: la informacion de aceleracion y curvatura (indices 7-10) aporta +1.08pp respecto al baseline `pos_vel_ang` (7D).

2. **`pos_ang_curv` (5D)** es la mejor opcion compacta: solo 5 features (x, y, sin, cos, dtheta) alcanzan 97.17%, practicamente igualando `full` con menos de la mitad de dimensiones.

3. **La posicion (x, y) es critica:** `all_no_pos` (9D sin posicion) pierde 1.5pp respecto a `full` (11D con posicion). Los features de velocidad solos (`vel_ang`, `vel_only`) son insuficientes (~48%).

4. **La curvatura (dtheta) es muy discriminativa:** `pos_ang_curv` (con dtheta) supera a `pos_vel_ang` (sin dtheta) en +0.92pp a pesar de tener menos features (5D vs 7D).

### 7.5 Efecto del numero de centroides

Para la configuracion ganadora (KMeans, full 11D):

| k | Accuracy (val) |
|---|----------------|
| 8 | 89.50% |
| 16 | 94.17% |
| 32 | 96.92% |
| 64 | 97.08% |
| 128 | 97.08% |
| **256** | **97.33%** |

La accuracy crece logaritmicamente con k: el mayor salto ocurre de 8 a 32 (+7.42pp), y a partir de 64 los rendimientos son decrecientes (+0.25pp de 64 a 256). Esto sugiere que ~32-64 centroides capturan la estructura esencial, y centroides adicionales refinan detalles menores.

---

## 8. Configuracion final optimizada

| Parametro | Valor |
|-----------|-------|
| Algoritmo | KMeans (k-means++, 10 reintentos) |
| N centroides | 256 |
| Features | full (11D): x, y, vx, vy, v, sin, cos, ax, ay, |a|, dtheta |
| Clasificacion | argmin distorsion media |

---

## 9. Resultados finales

### 9.1 Comparativa baseline vs optimizado

| Escenario | Baseline (7D, k=32) | Optimizado (11D, k=256) | Mejora |
|-----------|----------------------|--------------------------|--------|
| **N=74** | 96.12% | **97.24%** | **+1.12pp** |
| **N=47** | 96.06% | **97.28%** | **+1.22pp** |
| **LOO CV** | 95.69% +/- 5.27% | **96.91% +/- 4.23%** | **+1.22pp** |

### 9.2 Analisis de mejoras

- **Mejora consistente** en los tres escenarios (~1.2pp), indicando que la ganancia es real y no artefacto de una particion concreta.
- **Reduccion de variabilidad:** El LOO CV pasa de std=5.27% a std=4.23% (-1.04pp), indicando mayor robustez entre usuarios.
- **Robustez al tamanio de entrenamiento:** La configuracion optimizada pierde solo 0.04pp entre N=74 y N=47 (vs 0.06pp del baseline). Ambas configuraciones son muy estables.
- **Coste:** El modelo optimizado tiene 8x mas centroides y 1.6x mas features, pero la evaluacion sigue siendo rapida (~58s para grid completa, ~20 min LOO).

---

## 10. Coste computacional

| Fase | Configuraciones | Tiempo |
|------|-----------------|--------|
| Baseline (3 escenarios + LOO) | 3 + 93 folds | ~20 min |
| Grid search (validacion) | 144 configs | ~166 min |
| Evaluacion final (N=74, N=47, LOO) | 2 + 93 folds | ~25 min |

**Comparacion con HMM:** El entrenamiento VQ es ordenes de magnitud mas rapido que HMM. El grid search completo de 144 configs de VQ (~166 min) es comparable al tiempo de evaluar un unico escenario HMM (~4 horas).

---

## 11. Descripcion del codigo

| Fichero | Descripcion |
|---------|-------------|
| `implementacion_VQ.py` | Baseline VQ: carga, preprocesado, features (7D), KMeans k=32, 3 escenarios |
| `busqueda_VQ.py` | Grid search: 3 algos, 8 feature sets, 6 centroides, LBG, evaluacion final |

### Resultados generados

| Fichero | Contenido |
|---------|-----------|
| `resultados_VQ/resultados_VQ.json` | Resultados del baseline (N=74, N=47, LOO) |
| `resultados_VQ/resultados_busqueda_VQ.json` | Busqueda: mejor config, top 20, evaluacion final |
| `resultados_VQ/checkpoint_busqueda.json` | Checkpoint del grid search (144 configs) |
| `resultados_VQ/confusion_VQ_N74.png` | Confusion baseline N=74 |
| `resultados_VQ/confusion_VQ_N47.png` | Confusion baseline N=47 |
| `resultados_VQ/confusion_VQ_LOO.png` | Confusion baseline LOO |
| `resultados_VQ/confusion_VQ_opt_N74.png` | Confusion optimizado N=74 |
| `resultados_VQ/confusion_VQ_opt_N47.png` | Confusion optimizado N=47 |
| `resultados_VQ/confusion_VQ_opt_LOO.png` | Confusion optimizado LOO |
| `resultados_VQ/top_configs_VQ.png` | Top 25 configs del grid search |

---

## 12. Comparativa VQ vs HMM

| Escenario | HMM E1 (GaussianHMM) | HMM E2B (GMMHMM) | VQ Baseline | VQ Optimizado |
|-----------|-----------------------|-------------------|-------------|---------------|
| **N=74** | 92.89% | 93.03% | 96.12% | **97.24%** |
| **N=47** | 89.78% | 92.61% | 96.06% | **97.28%** |
| **LOO CV** | 78.40% +/- 8.79% | — | 95.69% +/- 5.27% | **96.91% +/- 4.23%** |

VQ supera a HMM por un amplio margen (+4.21pp en N=74, +4.67pp en N=47 respecto al mejor HMM). Las razones probables:

1. **No requiere modelado temporal explicito:** VQ trata cada punto como un vector independiente, evitando errores de estimacion de transiciones de estado del HMM.
2. **Mas datos efectivos por modelo:** Cada codebook se entrena con todos los vectores de features de un digito (miles), mientras que el HMM entrena sobre secuencias completas (cientos).
3. **Menor numero de parametros:** Un codebook de 256 centroides en 11D = 2816 parametros, frente a un GMMHMM con 7 estados y 2 mezclas en 12D que tiene ~350 parametros por estado x 7 = ~2450 parametros, pero con restricciones mucho mas fuertes (estimacion EM, topologia left-right).

---

## 13. Conclusiones

1. **VQ es un clasificador sorprendentemente eficaz** para digitos manuscritos, superando significativamente a todos los modelos HMM evaluados (+4pp en N=74).

2. **El baseline ya es competitivo:** KMeans con solo 32 centroides y 7 features alcanza 96.12% (N=74), superando al mejor HMM (93.03%).

3. **La optimizacion aporta una mejora modesta pero consistente** (+1.2pp) al pasar a 11D features con 256 centroides. Los rendimientos decrecientes sugieren que el baseline ya captura la mayor parte de la varianza discriminativa.

4. **Las features de posicion (x, y) y curvatura (dtheta) son las mas discriminativas.** El subconjunto `pos_ang_curv` (5D) iguala practicamente al `full` (11D).

5. **KMeans es el mejor algoritmo** por equilibrio entre calidad y velocidad. LBG alcanza resultados similares pero es ~7x mas lento para k grande. MiniBatchKMeans es la opcion rapida con minima perdida.

6. **VQ es ordenes de magnitud mas rapido que HMM**, tanto en entrenamiento como en busqueda de hiperparametros, lo que permite explorar espacios de busqueda mucho mayores.


