# Informe VQ: Clasificacion de Digitos Manuscritos con Vector Quantization

## Proyecto ASMI - e-BioDigit

---

## 1. Objetivo

Clasificar digitos manuscritos (0-9) a partir de trazados dinamicos capturados en una tableta digitalizadora, utilizando **Vector Quantization (VQ)** basada en codebooks. Se evaluan dos fases:

- **Baseline:** Un codebook KMeans (32 centroides) por digito con un subconjunto 7D del extractor local.
- **Optimizacion:** Busqueda exhaustiva sobre algoritmo (KMeans, LBG, MiniBatchKMeans), features (8 subconjuntos de 5D a 21D del extractor local) y numero de centroides (8-256).

Ambas fases usan el **mismo extractor de features locales** que el pipeline HMM (`extract_local_features.get_features`), lo que garantiza una comparativa homogenea entre clasificadores.

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

Se usa el **extractor local comun** (`Extractores_adaptados/.../extract_local_features.get_features`), compartido con el pipeline HMM. Produce una matriz `(T, 23)` por trazado; las columnas 2 y 9 (presion y su derivada) permanecen a cero — e-BioDigit no contiene presion util para la tarea — asi que quedan **21 features utiles**:

| Indice | Feature | Descripcion |
|--------|---------|-------------|
| 0 | x | Posicion horizontal |
| 1 | y | Posicion vertical |
| 3 | theta | Angulo atan(dy/dx) |
| 4 | v | Modulo de velocidad |
| 5 | rho | log(v / |dtheta|) |
| 6 | a | Modulo de aceleracion |
| 7 | dx | Derivada suavizada de x |
| 8 | dy | Derivada suavizada de y |
| 10 | dtheta | Curvatura |
| 11 | dv | Derivada de v |
| 12 | drho | Derivada de rho |
| 13 | da | Derivada de a |
| 14 | ddx | Segunda derivada de x |
| 15 | ddy | Segunda derivada de y |
| 16 | rminmax_v | min(v)/max(v) en ventana 5 |
| 17 | angle | Angulo de trayectoria |
| 18 | dangle | Derivada de angle |
| 19 | sin(angle) | Seno del angulo |
| 20 | cos(angle) | Coseno del angulo |
| 21 | lewiratio5 | Ratio longitud/anchura ventana 5 |
| 22 | lewiratio7 | Ratio longitud/anchura ventana 7 |

Las derivadas usan un esquema de diferencias centrales suavizadas sobre 5 puntos. Todos los NaN/Inf se sustituyen por 0.

### 5.1 Subconjuntos evaluados

Se definieron 8 subconjuntos de features para la busqueda (indices sobre las 23 columnas del extractor):

| Subconjunto | Dim | Indices | Features |
|-------------|-----|---------|----------|
| `pos_ang_curv` | **5D** | `0,1,19,20,10` | x, y, sin, cos, dtheta |
| `min` | 7D | `7,8,19,20,10,4,5` | dx, dy, sin, cos, dtheta, v, rho |
| `kin_deriv` | 8D | `7,8,4,6,11,14,15,13` | dx, dy, v, a, dv, ddx, ddy, da |
| `geom` | 9D | `0,1,3,10,5,19,20,17,21` | x, y, theta, dtheta, rho, sin, cos, angle, lewiratio5 |
| `med_noxy` | 10D | `7,8,19,20,10,4,5,6,11,21` | `min` + a, dv, lewiratio5 |
| `med` | 12D | `7,8,19,20,10,4,5,6,11,21,0,1` | `med_noxy` + x, y |
| `med_plus` | 14D | `7,8,19,20,10,4,5,6,11,21,0,1,12,13` | `med` + drho, da |
| **`full`** | **21D** | `0,1,3-8,10-22` | Las 21 features utiles |

Los subsets `min`, `med` y `full` son los mismos que emplea el pipeline HMM, para facilitar la comparacion directa.

---

## 6. Baseline: KMeans con 32 centroides

### 6.1 Configuracion

| Parametro | Valor |
|-----------|-------|
| Algoritmo | KMeans (k-means++, 10 reintentos) |
| N centroides | 32 |
| Features | Subset 7D del extractor local: `[0, 1, 7, 8, 4, 19, 20]` → x, y, dx, dy, v, sin(angle), cos(angle) |
| Clasificacion | argmin distorsion media |

Se elige un subset analogo al clasico `pos_vel_ang` pero sourcing del extractor comun, para que el baseline sea directamente comparable con la fase de optimizacion.

### 6.2 Resultados baseline

| Escenario | Accuracy |
|-----------|----------|
| N=74 | **95.59%** |
| N=47 | **96.06%** |
| LOO CV | **95.75% +/- 4.72%** |

La diferencia entre N=74 y N=47 es minima (0.47pp a favor de N=47), indicando robustez al tamanio de entrenamiento. La media LOO (95.75%) cae entre ambos, confirmando que no hay particion especialmente afortunada.

---

## 7. Busqueda de hiperparametros

### 7.1 Configuracion de la busqueda

- **Validacion:** 78 usuarios train / 15 usuarios validacion (misma particion que entregas HMM).
- **Grid:** 3 algoritmos x 8 feature sets x 6 centroides = **144 configuraciones**.
- **Criterio:** Accuracy de clasificacion sobre validacion.

### 7.2 Top 10 configuraciones

| Rank | Algoritmo | Features | k | Accuracy (val) | Tiempo |
|------|-----------|----------|---|----------------|--------|
| **1** | **MiniBatchKMeans** | **pos_ang_curv (5D)** | **128** | **96.92%** | **6s** |
| 2 | LBG | pos_ang_curv (5D) | 64 | 96.58% | 50s |
| 3 | LBG | pos_ang_curv (5D) | 256 | 96.58% | 262s |
| 4 | LBG | pos_ang_curv (5D) | 128 | 96.50% | 115s |
| 5 | MiniBatchKMeans | pos_ang_curv (5D) | 64 | 96.50% | 3s |
| 6 | KMeans | pos_ang_curv (5D) | 256 | 96.42% | 59s |
| 7 | KMeans | pos_ang_curv (5D) | 128 | 96.33% | 34s |
| 8 | KMeans | pos_ang_curv (5D) | 64 | 96.17% | 17s |
| 9 | MiniBatchKMeans | pos_ang_curv (5D) | 256 | 96.08% | 10s |
| 10 | KMeans | pos_ang_curv (5D) | 32 | 96.00% | 12s |

El top 10 esta completamente dominado por el subset **`pos_ang_curv` (5D)**, independientemente del algoritmo.

### 7.3 Analisis por algoritmo

| Algoritmo | Mejor accuracy | Config | Tiempo |
|-----------|----------------|--------|--------|
| **MiniBatchKMeans** | **96.92%** | pos_ang_curv, k=128 | 6s |
| LBG | 96.58% | pos_ang_curv, k=64 | 50s |
| KMeans | 96.42% | pos_ang_curv, k=256 | 59s |

- **MiniBatchKMeans** obtiene la mejor accuracy global *y* es el mas rapido (~10x mas que KMeans y ~20x mas que LBG). En 5D las actualizaciones en mini-batch convergen a codebooks ligeramente distintos que parecen generalizar mejor en validacion — el efecto estocastico actua como regularizacion.
- **LBG** alcanza resultados comparables gracias a su construccion jerarquica por splitting, pero el coste crece fuertemente con k (262s para k=256 frente a 10s de MBKMeans).
- **KMeans** clasico queda un escalon por debajo. Todas las diferencias entre algoritmos son pequeñas (<0.5pp), confirmando que en 5D la eleccion del algoritmo importa poco comparada con la eleccion del subset de features.

### 7.4 Analisis por subconjunto de features

| Subconjunto | Dim | Mejor accuracy | Config |
|-------------|-----|----------------|--------|
| **pos_ang_curv** | **5D** | **96.92%** | MBKMeans, k=128 |
| med | 12D | 88.83% | KMeans, k=256 |
| kin_deriv | 8D | 88.50% | MBKMeans, k=256 |
| med_plus | 14D | 87.67% | KMeans, k=256 |
| geom | 9D | 86.67% | KMeans, k=256 |
| min | 7D | 80.08% | KMeans, k=256 |
| full | 21D | 78.25% | KMeans, k=256 |
| med_noxy | 10D | 67.92% | KMeans, k=256 |

**Observaciones:**

1. **`pos_ang_curv` (5D) es abrumadoramente el mejor subset** (+8pp sobre el segundo), pese a tener la menor dimensionalidad. Las 5 features `x, y, sin(angle), cos(angle), dtheta` describen compactamente la geometria del trazo sin introducir magnitudes correlacionadas o ruidosas.

2. **`full` (21D) es contraproducente:** 78.25%, casi 19pp peor que `pos_ang_curv`. En VQ con distancia euclidea no ponderada, añadir dimensiones con escalas heterogeneas (rho, lewiratio, derivadas segundas, etc.) **dilata** la metrica y empeora la clasificacion. Esto es el fenomeno clasico de la "maldicion de la dimensionalidad" en metric learning no parametrico, y un contraste fuerte con HMM, que estima covarianzas por estado y gestiona mejor features heterogeneas.

3. **La posicion (x, y) es critica:** quitarla (`med_noxy`) hunde el rendimiento a 67.92%, la peor cifra del grid. La posicion da una referencia absoluta que el resto de features dinamicas no capturan.

4. **La curvatura (dtheta) es muy discriminativa:** subsets ricos en dinamica sin dtheta o con mucho ruido adicional (`kin_deriv`, `min`) quedan en ~80-88%. El par (angulo sin/cos, dtheta) es suficiente para caracterizar la forma del trazo.

### 7.5 Efecto del numero de centroides

Para la configuracion ganadora (MBKMeans, pos_ang_curv 5D):

| k | Accuracy (val) |
|---|----------------|
| 8 | 91.33% |
| 16 | 93.42% |
| 32 | 95.33% |
| 64 | 96.50% |
| **128** | **96.92%** |
| 256 | 96.08% |

La accuracy crece hasta k=128 y **empeora ligeramente en k=256** (-0.84pp). El codebook optimo es intermedio: con 128 centroides por digito (1280 totales) se captura la estructura esencial sin sobreajustar el ruido del train. En solo 5D, k=256 empieza a sobreparticionar el espacio y perder generalizacion.

---

## 8. Configuracion final optimizada

| Parametro | Valor |
|-----------|-------|
| Algoritmo | MiniBatchKMeans (k-means++, 10 reintentos, batch=1024) |
| N centroides | 128 |
| Features | pos_ang_curv (5D): x, y, sin(angle), cos(angle), dtheta |
| Clasificacion | argmin distorsion media |

---

## 9. Resultados finales

### 9.1 Comparativa baseline vs optimizado

| Escenario | Baseline (7D, k=32) | Optimizado (5D, k=128) | Mejora |
|-----------|----------------------|--------------------------|--------|
| **N=74** | 95.59% | **96.97%** | **+1.38pp** |
| **N=47** | 96.06% | **97.06%** | **+1.00pp** |
| **LOO CV** | 95.75% +/- 4.72% | **96.94% +/- 3.85%** | **+1.19pp** |

### 9.2 Analisis de mejoras

- **Mejora consistente** en los tres escenarios (~1.0-1.4pp), indicando que la ganancia es real y no artefacto de una particion concreta.
- **El optimo usa menos features que el baseline** (5D vs 7D): la clave no es añadir dinamica sino cambiar velocidad (dx, dy, v) por curvatura (dtheta). Menos features discriminativas superan a mas features ruidosas.
- **Reduccion de variabilidad:** El LOO CV pasa de std=4.72% a std=3.85% (-0.87pp), indicando mayor robustez entre usuarios.
- **Coste:** El grid search completo (144 configs) tarda ~2.5 h, dominado por LBG con k grande. El modelo final (MBKMeans k=128 en 5D) entrena en <10s por fold.

---

## 10. Coste computacional

| Fase | Configuraciones | Tiempo |
|------|-----------------|--------|
| Baseline (N=74, N=47 + LOO) | 2 + 93 folds | ~35 min |
| Grid search (validacion) | 144 configs | ~151 min |
| Evaluacion final (N=74, N=47, LOO) | 2 + 93 folds | ~8 min |

**Distribucion del tiempo del grid search** (por algoritmo):

- LBG: ~130 min (85%) — escala cubicamente con k debido a las iteraciones de Lloyd en cada paso de splitting.
- KMeans: ~18 min (12%).
- MiniBatchKMeans: ~3 min (2%) — un orden de magnitud mas rapido que KMeans clasico.

**Comparacion con HMM:** El entrenamiento VQ es ordenes de magnitud mas rapido que HMM. El grid search completo de 144 configs de VQ (~150 min) es comparable al tiempo de evaluar un unico escenario HMM (~4 horas). La evaluacion final (MBKMeans 5D k=128) se completa en ~8 min incluidos los 93 folds de LOO.

---

## 11. Descripcion del codigo

| Fichero | Descripcion |
|---------|-------------|
| `implementacion_VQ.py` | Baseline VQ: extractor local (subset 7D), KMeans k=32, 3 escenarios |
| `busqueda_VQ.py` | Grid search: 3 algos, 8 feature sets (5-21D), 6 centroides, LBG, evaluacion final |

Ambos scripts importan `get_features` del extractor local comun (`Extractores_adaptados/Extractores/Extractor Local/extract_local_features.py`).

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
| **N=74** | 92.89% | 93.03% | 95.59% | **96.97%** |
| **N=47** | 89.78% | 92.61% | 96.06% | **97.06%** |
| **LOO CV** | 78.40% +/- 8.79% | — | 95.75% +/- 4.72% | **96.94% +/- 3.85%** |

VQ supera a HMM por un amplio margen (+3.94pp en N=74, +4.45pp en N=47 respecto al mejor HMM). Las razones probables:

1. **No requiere modelado temporal explicito:** VQ trata cada punto como un vector independiente, evitando errores de estimacion de transiciones de estado del HMM.
2. **Mas datos efectivos por modelo:** Cada codebook se entrena con todos los vectores de features de un digito (miles), mientras que el HMM entrena sobre secuencias completas (cientos).
3. **Menor numero de parametros:** Un codebook de 128 centroides en 5D = 640 parametros por digito (6400 totales), y entrena mediante Lloyd sin las restricciones EM/topologia left-right del HMM.
4. **Mismo extractor de features:** Al usar ambos pipelines `get_features` con los mismos subsets (`min`, `med`, `full`), la comparacion aisla el modelo. Curiosamente, el mejor subset para VQ (`pos_ang_curv`, 5D) no pertenece al conjunto estandar de HMM, pero el sencillo cambio de metrica (distorsion media vs verosimilitud Gaussiana) cambia drasticamente que features son utiles.

---

## 13. Conclusiones

1. **VQ es un clasificador sorprendentemente eficaz** para digitos manuscritos, superando significativamente a todos los modelos HMM evaluados (+4pp en N=74).

2. **El baseline ya es competitivo:** KMeans con 32 centroides y un subset 7D del extractor local alcanza 95.59% (N=74), superando al mejor HMM (93.03%).

3. **La optimizacion aporta una mejora de ~1.0-1.4pp** al pasar a **menos** features (5D) con mas centroides (k=128). Contrario al caso HMM, añadir dimensiones empeora el rendimiento de VQ debido a la distancia euclidea no ponderada.

4. **Las features de posicion (x, y) y angulo/curvatura (sin, cos, dtheta) son las mas discriminativas.** El subconjunto `pos_ang_curv` (5D) domina el top 10 completo y bate a `full` (21D) por ~19pp.

5. **MiniBatchKMeans es el mejor algoritmo** por equilibrio entre calidad y velocidad: 96.92% en 6s frente a 96.42% en 59s de KMeans clasico. LBG alcanza 96.58% pero es un orden de magnitud mas lento.

6. **VQ es ordenes de magnitud mas rapido que HMM**, tanto en entrenamiento como en busqueda de hiperparametros, lo que permite explorar espacios de busqueda mucho mayores.

7. **Lesson learned sobre features en VQ no parametrico:** a diferencia de HMM (que aprende covarianzas por estado), VQ con distancia euclidea requiere seleccion cuidadosa de features de escala similar. Un extractor "rico" (21D) es una desventaja, no una ventaja, sin ponderacion o reduccion dimensional previa.


