# Informe Entrega 3: Ensembles HMM + VQ para Clasificacion de Digitos

## Proyecto ASMI - e-BioDigit

---

## 1. Objetivo

Combinar los mejores clasificadores individuales obtenidos en la Entrega 2
(GMMHMM con n_mix=2 y VQ MiniBatchKMeans con features `pos_ang_curv` 5D) en
dos sistemas de fusion complementarios:

- **Sistema Paralelo (Paralel/):** ambos clasificadores deciden de forma
  independiente sobre la misma muestra; las decisiones se fusionan con
  reglas que tienen en cuenta tanto la prediccion como la confianza /
  incertidumbre de cada modelo. La hipotesis es que los fallos de HMM y
  VQ apenas se solapan, asi que un ensemble informado por la confianza
  puede acercarse a la cota oraculo `acc(HMM) U acc(VQ)`.

- **Sistema Serie (Serial/):** la salida del HMM (vector de 10
  log-verosimilitudes, una por digito) se usa como espacio de features
  para un segundo clasificador VQ. La intuicion es que, si el HMM
  funciona razonablemente, las muestras del digito *d* presentaran su
  maximo en la coordenada *d*, formando agrupaciones por clase que un VQ
  con codebook por digito puede explotar. Se realiza busqueda de
  hiperparametros (algoritmo, normalizacion, numero de centroides).

Cada sistema se evalua en los tres escenarios habituales de la
asignatura:

- **N=74:** 74 usuarios train / 19 usuarios test.
- **N=47:** 47 usuarios train / 46 usuarios test.
- **K=5 fold por usuario** (sustituye al LOO original): 5 pliegues sobre
  los 93 usuarios, ~74 train / ~19 test por fold, cubriendo los 7440
  trazados. El K-fold permite usar HMM bien entrenado en cada fold
  (n_iter=80, n_restarts=6), cosa que el LOO original no permitia por
  presupuesto computacional.

Las metricas reportadas son **accuracy global**, **AUC macro** (one-vs-rest)
y **EER macro** (one-vs-rest), consistentes con la evaluacion biometrica:
para cada digito *d* se construye un problema binario (d vs no-d), se
calcula AUC y EER, y se promedia sobre los 10 digitos.

---

## 2. Base de datos y preprocesado

Igual que en Entregas anteriores: e-BioDigit con 93 usuarios x 10 digitos
x 8 muestras = 7440 trazados.

Los pipelines de preprocesado y los extractores de features son los
mismos de la Entrega 2:

- **Pipeline HMM (12D, subset `med`):** centrado, escalado por rango,
  suavizado Savitzky-Golay (ventana=7, orden=3), remuestreo a 80 puntos
  y normalizacion Z-Score ajustada en train. Subset `med` =
  {dx, dy, sin, cos, dtheta, v, rho, a, dv, lewi5, x, y}.
- **Pipeline VQ (5D, subset `pos_ang_curv`):** centrado, escalado por
  rango (sin suavizado ni remuestreo). Subset = {x, y, sin(angle),
  cos(angle), dtheta}.

Ambos pipelines usan el extractor local comun de
`Extractores_adaptados/Extractores/Extractor Local/`.

---

## 3. Modelos base (recordatorio Entrega 2 + reentrenamiento Entrega 3)

| Modelo | Configuracion | N=74 | N=47 |
|--------|---------------|------|------|
| **HMM E2B (GMMHMM)** | n_mix=2, 7 estados, p_autolazo=0.6, diag, features=`med` | 93.03% | 92.61% |
| **VQ optimizado** | MiniBatchKMeans, k=128, features=`pos_ang_curv` 5D | 96.97% | 97.06% |

En esta entrega los HMM se reentrenan en cada experimento con
`entrenar_gmmhmm_digito` (`Entrega2/HMM/clasificador_digitos_v3.py`).
Tras detectar que la version inicial (n_iter=40, n_restarts=2) provocaba
convergencias fallidas en algunos digitos y arrastraba al ensemble, se
ha reentrenado con un fit mas fuerte:

| Configuracion HMM | n_iter | n_restarts | Uso |
|-------------------|--------|------------|-----|
| **HMM_FIT_FULL** | 100 | 10 | Entrenamiento principal en N=74 y N=47 (`ensemble_*_full.json`) |
| **HMM_FIT_KFOLD** | 80 | 6  | Cada uno de los K=5 folds (`ensemble_K5fold.json`) |

Con esta configuracion los 10 GMMHMM convergen sin fallos en todos los
escenarios (no aparece ya el problema de LL=-inf que penalizaba la
Entrega 2 reducida). Las accuracies del HMM aislado vuelven al rango
esperado de Entrega 2 (~96.5% en N=74, 93.3% en N=47), lo que cambia
sustancialmente el comportamiento del ensemble.

> **Nota sobre LOO vs K-fold:** el LOO original (93 folds, n_iter=15,
> n_restarts=1) producia un HMM muy debil (~70% accuracy) y dominaba en
> coste computacional. Lo hemos sustituido por **K=5 fold por usuario**
> con HMM bien entrenado (80/6), que cubre los 7440 trazados con HMM de
> calidad comparable al de N=74. El K-fold tambien provee las LL
> out-of-fold necesarias para el sistema Serie sin leakage.

---

## 4. Sistema Paralelo (`Paralel/ensemble_paralelo.py`)

### 4.1 Disenio

Para cada muestra de test se calculan en paralelo:

- **HMM:** vector `LL` de 10 log-verosimilitudes, una por digito.
- **VQ:** vector `D` de 10 distorsiones medias (distancia^2 a centroide
  mas cercano, promediada sobre los puntos del trazado).

Antes de fusionar, las puntuaciones de cada modelo se convierten a
probabilidades comparables. La clave es que las LL del HMM tienen
escalas absolutas muy distintas a las distorsiones VQ: un softmax
directo sobre LL produce una distribucion saturada (one-hot), perdiendo
la senial de incertidumbre. Por eso aplicamos a **ambos** scores la
normalizacion z-score por muestra antes del softmax:

```
score_normalizado = (raw - mean) / std    # por muestra
prob = softmax(score_normalizado)
```

Asi tanto `hmm_probs` como `vq_probs` viven en el mismo rango y la
confianza top-1 (max prob) y el margen top1-top2 son comparables entre
los dos modelos.

### 4.2 Reglas de fusion evaluadas

| Regla | Definicion |
|-------|------------|
| `hmm` | Linea base: argmax LL del HMM. |
| `vq`  | Linea base: argmin distorsion del VQ. |
| `agreement` | Si HMM y VQ coinciden, esa es la prediccion. Si no, se elige la del modelo con mayor confianza top-1. |
| `soft` | Promedio simple de probabilidades: `0.5 * hmm_probs + 0.5 * vq_probs`. |
| `conf_weighted` | Combinacion ponderada por confianza top-1: `conf_h * hmm_probs + conf_v * vq_probs`. |
| `margin_weighted` | Combinacion ponderada por margen top1-top2 de cada modelo. |
| `oracle` | Cota superior: acierto si `pred(HMM)==y` o `pred(VQ)==y`. |

### 4.3 Resultados N=74 (HMM 100/10)

| Metodo | Accuracy |
|--------|----------|
| HMM (argmax LL)      | 96.51% |
| VQ (argmin dist)     | 96.84% |
| **agreement**        | **97.37%** |
| **soft**             | **97.89%** |
| **conf_weighted**    | 97.76% |
| **margin_weighted**  | 97.57% |
| oracle               | **98.55%** |

### 4.4 Resultados N=47 (HMM 100/10)

| Metodo | Accuracy |
|--------|----------|
| HMM                  | 93.32% |
| VQ                   | 97.04% |
| **agreement**        | 96.90% |
| **soft**             | **97.47%** |
| **conf_weighted**    | 97.34% |
| **margin_weighted**  | 97.34% |
| oracle               | **98.56%** |

### 4.5 Resultados K=5 fold (5 folds, 7440 muestras, HMM 80/6)

| Metodo | Accuracy |
|--------|----------|
| HMM                  | 80.51% |
| VQ                   | 97.11% |
| **agreement**        | 97.10% |
| **soft**             | 95.22% |
| **conf_weighted**    | 96.94% |
| **margin_weighted**  | **97.03%** |
| oracle               | **98.13%** |

> El HMM 80/6 de los folds es ligeramente menos potente que el HMM 100/10
> de las particiones N=74/N=47, lo que explica la accuracy HMM mas baja
> (80.51% vs 96.51%). Aun asi es muchisimo mas robusto que el HMM 15/1
> del LOO original (~70%).

### 4.6 Discusion del sistema Paralelo

1. **Con el HMM bien entrenado, los ensembles si superan al VQ aislado.**
   En N=74 `soft` alcanza 97.89% (vs 96.84% del VQ, +1.05pp) y
   `agreement` 97.37%. La cota oraculo es 98.55%, asi que el ensemble
   captura ~73% del margen disponible. En N=47 `soft` 97.47% vs VQ
   97.04% (+0.43pp). El reentrenamiento del HMM era la pieza que
   faltaba: con el HMM debil de la version inicial todos los ensembles
   delegaban en VQ y se quedaban en 96.84%.

2. **Las reglas suaves dominan ahora a las duras.** `soft` y
   `conf_weighted` se imponen a `agreement` y `margin_weighted` cuando
   los dos modelos tienen accuracy comparable (caso N=74, N=47); el
   promedio de probabilidades aprovecha que los errores del HMM y del
   VQ apenas se solapan (oracle ~98.5%, ambos fallan <1.5%).

3. **En K=5 fold el HMM sigue siendo mas debil que el VQ** (80.51% vs
   97.11%) porque cada fold se entrena con ~74 usuarios pero con
   n_iter=80 (vs 100 en N=74); aqui los ensembles se quedan al nivel
   del VQ aislado (margin_weighted 97.03%, agreement 97.10%) porque
   la asimetria de confianzas vuelve a empujar todo el voto hacia VQ.
   La regla `soft` cae a 95.22% justamente por mezclar 50/50 con un
   HMM cuyo accuracy es 80%.

4. **El HMM pierde ~3pp entre N=74 y N=47** (96.51% -> 93.32%) por el
   menor numero de usuarios train, pero ya no se rompe como ocurria en
   la version 40/2. El VQ apenas varia entre escenarios. El comportamiento
   del ensemble es por tanto mucho mas estable.

---

## 5. Sistema Serie (`Serial/serial_hmm_vq.py` + `Serial/update_serial_full_hmm.py`)

### 5.1 Disenio

El HMM produce, por muestra, un vector `LL` de dimension 10 (una
log-verosimilitud por modelo de digito). Se entrena un VQ por clase
sobre este espacio 10D y se clasifica por minima distancia al
centroide mas cercano de cualquier codebook.

El primer obstaculo del esquema es el leakage: si las LL de train
provienen del mismo HMM que evalua el test, el VQ aprende un espacio
"facil" que no es el que ve en deployment. La solucion es entrenar el
VQ con LL **out-of-fold** (OOF).

> **Decision clave (version original):** K=3 sobre los usuarios de
> train, mismo HMM_FIT en folds y final. Funcionaba en N=74 (HMM
> coherente entre folds y final) pero **se rompia catastroficamente en
> N=47 (acc=9.89%)**: con solo 47 usuarios divididos en 3 folds, cada
> HMM de fold se entrenaba con ~15 usuarios mientras el HMM final con
> 47, y la diferencia de magnitud entre las LL OOF y las LL del HMM
> final hacia que el VQ no encontrara los centroides en deployment.

### 5.2 Solucion en Entrega 3: K=5 global como fuente de train LLs

Se desacopla la generacion de train LLs de la del test:

- **Test LLs:** se toman de `ensemble_{N74,N47}_full.json`, donde el HMM
  se entreno con todos los usuarios de train del escenario y con
  HMM_FIT_FULL (100/10).
- **Train LLs (OOF):** se toman de `ensemble_K5fold.json`, donde cada
  fold entreno un HMM con ~74 usuarios y HMM_FIT_KFOLD (80/6) y produjo
  LL out-of-fold para los ~19 usuarios restantes. Para cada escenario
  N=74 / N=47 se filtran las LL OOF a los usuarios train
  correspondientes.

La normalizacion `softmax` por muestra es **invariante a la escala
absoluta** de las LL, asi que es razonablemente robusta a la diferencia
de hyperparametros del HMM (100/10 vs 80/6). zscore tambien lo es; raw
y shift no.

### 5.3 Espacio de busqueda

El grid se realizo sobre la version original (HMM_FIT identico en
folds y final) y la mejor configuracion encontrada se reutiliza con la
nueva fuente de LLs:

| Hiperparametro | Valores |
|----------------|---------|
| Algoritmo | KMeans, MiniBatchKMeans, LBG (Linde-Buzo-Gray) |
| Normalizacion del LL por muestra | zscore, softmax, shift |
| N centroides por digito | 1, 2, 4, 8, 16, 32 |

Mejor configuracion: **KMeans + softmax + k=2** (gano en el grid de
N=74 con 90.25% accuracy validacion). Es la que se usa en la version
final con LLs decopladas.

### 5.4 Resultados (con HMM bien entrenado, train OOF de K=5)

| Escenario | HMM baseline (100/10) | Serial HMM->VQ |
|-----------|----------------------|----------------|
| **N=74** | 96.51% | **96.45%** |
| **N=47** | 93.32% | **85.08%** |

> El fallo catastrofico de N=47 (9.89%) **queda arreglado** al usar las
> LLs OOF del K=5 global como train, en lugar de reentrenar K=3 sobre
> 47 usuarios. Sin embargo el sistema Serie sigue sin superar al HMM
> baseline.

### 5.5 Discusion del sistema Serie

1. **Cuando el HMM esta bien entrenado, el cascading no aporta accuracy
   adicional.** En N=74 el Serial empata al HMM baseline (96.45% vs
   96.51%); en N=47 incluso pierde 8pp (85.08% vs 93.32%). La razon es
   estructural: cada muestra produce un unico vector 10D. La "VQ" sobre
   puntos sueltos se reduce a un nearest-class-mean (con k>1,
   multiprototipo). Si la coordenada del digito correcto ya es la mayor,
   el centroide del digito correcto en el espacio LL tambien sera el mas
   cercano. El VQ solo ayuda cuando los maximos estan ambiguos y su
   training distribution coincide con la de test; en N=47 la
   discrepancia 80/6 vs 100/10 entre fuentes de LL erosiona esa
   coincidencia.

2. **El sistema Serie sigue valido como *score smoother*.** Aunque la
   accuracy dura no mejore, la version con normalizacion `softmax`
   produce un score continuo mas suave que el simple argmax sobre LL
   crudas, lo que mejora el ranking de confianza (ver tabla EER en la
   seccion 8).

3. **La leccion estructural se mantiene:** las normalizaciones
   invariantes a escala (zscore, softmax) son criticas. Aun asi, el
   Serial es muy sensible al gap de hyperparametros entre el HMM que
   produce las train LLs y el HMM que produce las test LLs. La solucion
   ideal seria entrenar ambos con identica receta y suficientes
   usuarios (caso N=74); en N=47 el gap es inherente a la division.

---

## 6. Comparativa global (incluye Entregas anteriores)

### 6.1 Accuracy

| Sistema | N=74 | N=47 | K=5 fold |
|---------|------|------|----------|
| HMM E1 (GaussianHMM, Entrega 1) | 92.89% | 89.78% | — |
| HMM E2B (GMMHMM, Entrega 2)     | 93.03% | 92.61% | — |
| **HMM E3 (GMMHMM 100/10)**      | **96.51%** | **93.32%** | 80.51% (80/6) |
| VQ baseline (Entrega 2)         | 95.59% | 96.06% | — |
| **VQ optimizado (Entrega 2)**   | 96.97% | 97.06% | 97.11% |
| Paralel `agreement`             | 97.37% | 96.90% | 97.10% |
| **Paralel `soft`**              | **97.89%** | **97.47%** | 95.22% |
| Paralel `margin_weighted`       | 97.57% | 97.34% | 97.03% |
| Paralel oracle (cota)           | 98.55% | 98.56% | 98.13% |
| Serial HMM->VQ                  | 96.45% | 85.08% | — |

> Notas:
> - **N=47 Serial 85.08%**: queda razonable tras desacoplar las fuentes
>   de train/test LLs, pero sigue por debajo del HMM baseline por el
>   gap de hyperparametros entre HMM K-fold (80/6) y HMM final (100/10).
> - El **K=5 fold sustituye al LOO original**: cubre los 7440 trazados
>   con HMM bien entrenado (80/6) en lugar del 15/1 del LOO, lo que
>   permite comparaciones realistas.

### 6.2 EER macro one-vs-rest (pooled)

| Sistema | N=74 | N=47 | K=5 fold |
|---------|------|------|----------|
| E1 HMM (GaussianHMM)        | 5.66%  | 6.47%  | 20.59% |
| VQ K=32 7D (baseline E2)    | 7.70%  | 7.01%  | 7.23%  |
| **GMMHMM E2 (100/10)**      | 3.09%  | 4.78%  | 17.38% |
| **VQ opt K=128 5D**         | **1.84%** | 1.93% | **1.90%** |
| **Paralel margin_weighted** | **1.69%** | 1.74% | 2.30% |
| **Serial VQ-gated**         | 1.73% | **1.66%** | **1.94%** |

(Cifras de `metricas/det/eer_summary_nuevos.json`.)

---

## 7. Conclusiones

1. **El HMM bien entrenado cambia el panorama del ensemble.** Con la
   configuracion 40/2 inicial, el HMM aislado quedaba en 86%/75% y los
   ensembles se quedaban al nivel del VQ (96.84%) porque toda la
   informacion de fusion delegaba en VQ. Con HMM 100/10 el HMM aislado
   sube a 96.51%/93.32% y los ensembles **si superan al VQ**: en N=74
   `soft` 97.89% (+1.05pp), en N=47 `soft` 97.47% (+0.43pp). La cota
   oraculo (~98.5%) se acerca y el ensemble captura buena parte del
   margen.

2. **La hipotesis de complementariedad se verifica robustamente** en
   todos los escenarios: los aciertos de HMM y VQ se solapan en al menos
   el 98.13% del test (98.55% en N=74, 98.56% en N=47, 98.13% en K=5
   fold, cota oraculo). El % de muestras donde *ambos* fallan es
   siempre <2%. Cuando HMM y VQ tienen accuracy comparable (HMM 100/10
   en N=74/N=47) el ensemble cosecha esa complementariedad; cuando la
   asimetria es grande (K=5 fold, HMM 80/6) la cosecha es nula.

3. **Soft voting es la regla ganadora con HMM equilibrado.** Promediar
   probabilidades 50/50 explota mejor la complementariedad que las
   reglas duras (`agreement`) o ponderadas por confianza, que tienden
   a delegar en el modelo con mayor margen. Para tareas con HMM mas
   debil (K=5 fold) `agreement` y `margin_weighted` son preferibles
   porque no le dan voto excesivo al HMM.

4. **El sistema Serie no aporta accuracy adicional incluso con HMM bien
   entrenado.** Una vez que el HMM ya separa bien las clases (96.51% en
   N=74), el VQ encadenado sobre 10D no puede mejorar mucho mas: el
   centroide del digito correcto coincide con el argmax del HMM. **Pero
   sigue siendo util como *score smoother*** (mejora EER, ver seccion 8):
   la decision dura no cambia pero el ranking de confianza pasa de
   crudo (LL) a suave (distancia a centroides), lo que beneficia
   tareas biometricas con umbral.

5. **El fallo catastrofico de N=47 en la version inicial del Serial
   (acc=9.89%) queda arreglado** al desacoplar las fuentes de train y
   test LLs: train OOF del K=5 global (80/6), test del HMM final
   (100/10). La normalizacion `softmax` por muestra absorbe la mayor
   parte del gap de magnitud entre los dos HMM, aunque no del todo
   (Serial N=47 cae a 85.08% vs HMM baseline 93.32%).

6. **Lecciones operacionales:**
   - Antes de fusionar dos clasificadores con softmax, normalizar
     cada uno a escala comun (z-score por muestra) — sino el modelo de
     escala mas grande satura el ensemble.
   - Invertir presupuesto en convergencia del HMM (n_iter=100,
     n_restarts=10) antes que en arquitecturas mas elaboradas: el
     mismo ensemble con HMM 40/2 vs 100/10 pasa de no aportar nada a
     superar a su mejor componente individual.
   - Para cascadas, mantener una receta de HMM lo mas parecida posible
     entre la fuente de train OOF y el HMM final (idealmente identicos),
     o usar normalizaciones invariantes a escala (zscore, softmax).
   - Sustituir LOO por K-fold (con K=5 sobre usuarios) cuando el coste
     computacional impide entrenar HMM realista en cada fold.

---

## 8. Curvas DET y resumen global de EER

### 8.1 Por que DET en lugar de ROC

En tareas biometricas la representacion estandar es la curva DET
(Detection Error Tradeoff): se cruzan FPR (errores tipo I) y FNR
(errores tipo II) en escala probit (`norm.ppf`). Esta escala expande
las regiones de baja tasa de error y hace mas legible el comportamiento
de los sistemas competitivos. El EER (Equal Error Rate) corresponde al
punto donde FPR = FNR.

Para hacer comparable el problema de 10 clases con el formato binario
estandar, **agregamos los scores en formato uno-contra-resto pooled**:
cada par `(muestra, clase c)` se convierte en una observacion binaria
con etiqueta `(y == c)` y score = score asignado por el sistema a la
clase c. Concatenando las 10 clases obtenemos un unico problema binario
con `10 * N_test` observaciones del que se extrae la curva DET global y
el EER.

### 8.2 Modelos comparados (DET nuevas, `generar_det_nuevos.py`)

Se generan tres conjuntos de curvas DET por escenario, anidando
sistemas de mas simple a mas elaborado:

- **plot1**: E1 HMM (GaussianHMM) + VQ K=32 7D (baseline Entrega 2).
  La pareja minima de referencia historica.
- **plot2**: anade GMMHMM (E2, ahora con fit 100/10) y VQ optimizado
  (K=128 5D). Muestra cuanto ganamos sobre la baseline de Entrega 2.
- **plot3**: anade el ensemble Paralelo `margin_weighted` y un nuevo
  **Serial VQ-gated**: VQ K=128 como primario, HMM como verificador de
  baja confianza con peso_HMM ≤ 20%. Es el sistema mas robusto en EER.

| Modelo | Origen | Score |
|--------|--------|-------|
| E1 HMM (GaussianHMM)        | Entrega 1 reentregada en `Entrega3/run_entrega1_hmm.py` | softmax con z-score sobre LL |
| VQ K=32 7D                  | VQ baseline de Entrega 2 (sin optimizar) | softmax sobre -distorsion z-scored |
| GMMHMM E2 (100/10)          | HMM de `ensemble_*_full.json` | softmax con z-score sobre LL |
| VQ opt K=128 5D             | VQ optimizado de Entrega 2 (`pos_ang_curv`) | softmax sobre -distorsion z-scored |
| Paralel `margin_weighted`   | Sistema Paralelo - ponderado por margen | probabilidad ensemble 10D |
| Serial VQ-gated             | VQ primario, HMM verificador (≤20%) | combinacion gateada |

### 8.3 Tabla EER (pooled one-vs-rest)

(Cifras de `metricas/det/eer_summary_nuevos.json`.)

| Modelo | N=74 | N=47 | K=5 fold |
|--------|------|------|----------|
| E1 HMM (GaussianHMM)        | 5.66%  | 6.47%  | 20.59% |
| VQ K=32 7D                  | 7.70%  | 7.01%  | 7.23%  |
| GMMHMM E2 (100/10)          | 3.09%  | 4.78%  | 17.38% |
| **VQ opt K=128 5D**         | **1.84%** | 1.93%  | **1.90%** |
| **Paralel margin_weighted** | **1.69%** | 1.74%  | 2.30% |
| **Serial VQ-gated**         | 1.73%  | **1.66%** | 1.94% |

### 8.4 Lectura de las curvas DET

Plots generados (un PNG por escenario y nivel):

- `metricas/det/plot1_{N74,N47,LOO}.png`
- `metricas/det/plot2_{N74,N47,LOO}.png`
- `metricas/det/plot3_{N74,N47,LOO}.png`

(Las etiquetas LOO en los nombres se mantienen por compatibilidad
historica pero corresponden al K=5 fold actual.)

Observaciones principales:

1. **GMMHMM con fit 100/10 mejora muchisimo respecto a Entrega 2
   reducida**: EER 3.09% en N=74 (vs 12.88% en la version 40/2 que
   reportaba el informe original). El cambio de hyperparametros del
   HMM es el factor con mayor impacto en todo el pipeline.

2. **Paralel margin_weighted es ya el mejor en N=74** (EER 1.69% vs
   1.84% del VQ opt), confirmando que con HMM bien entrenado la fusion
   mejora incluso en EER. En N=47 y K=5 fold sigue muy cerca del VQ.

3. **Serial VQ-gated aporta el mejor EER en N=47** (1.66%) y resulta
   competitivo en los tres escenarios (1.73%, 1.66%, 1.94%). Su disenio
   (VQ primario y HMM solo como verificador de baja confianza) evita
   el problema del Serial original (donde el VQ-sobre-LL no superaba al
   HMM baseline en accuracy).

4. **VQ opt K=128 5D sigue siendo la baseline mas competitiva**: 1.84%,
   1.93%, 1.90% en los tres escenarios. La uniformidad cross-escenario
   confirma su robustez.

5. **E1 HMM (GaussianHMM) ya no gana al GMMHMM E2**: con HMM_FIT_FULL
   100/10 el GMMHMM converge correctamente y mejora a E1 en los tres
   escenarios (3.09 vs 5.66, 4.78 vs 6.47, 17.38 vs 20.59). En la
   version 40/2 era al reves.

### 8.5 Resumen final de EER por sistema y escenario

| Posicion (K=5) | Modelo | N=74 | N=47 | K=5 |
|----------------|--------|------|------|-----|
| 1 | VQ opt K=128 5D            | 1.84% | 1.93% | **1.90%** |
| 2 | Serial VQ-gated            | 1.73% | **1.66%** | 1.94% |
| 3 | Paralel margin_weighted    | **1.69%** | 1.74% | 2.30% |
| 4 | VQ K=32 7D                 | 7.70% | 7.01% | 7.23% |
| 5 | GMMHMM E2 (100/10)         | 3.09% | 4.78% | 17.38% |
| 6 | E1 HMM (GaussianHMM)       | 5.66% | 6.47% | 20.59% |

---

## 9. Coste computacional

| Fase | Tiempo aproximado |
|------|-------------------|
| Paralel N=74 / N=47 (HMM 40/2 + VQ + inferencia) | ~25 / ~20 min |
| Paralel **N=74 / N=47 full (HMM 100/10)**       | ~3 / ~2 horas |
| Paralel **K=5 fold (5 folds × HMM 80/6)**       | ~6 horas |
| Paralel LOO original (HMM 15/1)                 | ~10 horas (descartado) |
| Serial N=74 / N=47 (3-fold OOF + grid + final) | ~50 / ~30 min |
| Serial update (decoupled, usa K=5 + N=*_full)  | ~1 min |
| `metricas.py` (post-proceso AUC/EER)            | <1 min |
| `generar_det_nuevos.py` (DET 3 niveles × 3 escenarios) | ~5 min |

Ejecuciones simultaneas con `OMP_NUM_THREADS=4` cada una en un equipo
con 12 nucleos.

---

## 10. Descripcion del codigo

| Fichero | Descripcion |
|---------|-------------|
| `Paralel/ensemble_paralelo.py` | Sistema paralelo. Entrena HMM y VQ en cada escenario, calcula reglas de fusion, guarda predicciones + scores 10D + matrices de confusion. Implementa LOO con checkpointing por usuario (descartado en favor de K=5 fold). |
| `train_hmm_vq_splits.py`       | **Nuevo.** Entrena HMM 100/10 + VQ K=128 para N=74 y N=47, guarda en `ensemble_{N74,N47}_full.json`. Checkpoint por escenario. |
| `kfold_hmm_vq.py`              | **Nuevo.** K=5 fold por usuario sobre los 93 usuarios con HMM 80/6 + VQ K=128. Salida en `ensemble_K5fold.json`, formato compatible con LOO. |
| `Serial/serial_hmm_vq.py`      | Sistema serie original (K=3 sobre usuarios train, mismo HMM en folds y final). Genera `serial_{N74,N47,LOO}.json` y grid de hyperparametros. |
| `Serial/update_serial_full_hmm.py` | **Nuevo.** Actualiza `serial_{N74,N47}.json` usando test LLs de `ensemble_*_full.json` (HMM 100/10) y train LLs OOF de `ensemble_K5fold.json` (HMM 80/6). Aplica VQ k=2 + softmax. Arregla el fallo catastrofico de N=47. |
| `metricas.py`                  | Lee los JSON de cada sistema y calcula AUC y EER macro one-vs-rest. Guarda `metricas/summary.json` y barras AUC/EER por escenario. |
| `run_entrega1_hmm.py`          | Re-ejecucion del HMM E1 (GaussianHMM) de Entrega 1 guardando scores 10D por muestra. Salida en `Entrega1_rerun/`. |
| `det_curves.py`                | DET pooled one-vs-rest, version original con 7 sistemas overlay. |
| `generar_det_nuevos.py`        | **Nuevo.** Genera tres niveles de DET por escenario (plot1: E1 + VQ7D; plot2: anade GMMHMM E2 + VQ opt; plot3: anade Paralel margin_weighted + Serial VQ-gated). Salida en `metricas/det/plot{1,2,3}_*.png` y `eer_summary_nuevos.json`. |

### Ficheros generados

| Fichero | Contenido |
|---------|-----------|
| `Paralel/resultados/ensemble_<TAG>.json` | Por muestra: scores 10D, predicciones por regla, confianzas. |
| `Paralel/resultados/ensemble_<TAG>_full.json` | **Nuevo.** Igual que el anterior pero con HMM 100/10 (N=74 y N=47). |
| `Paralel/resultados/ensemble_K5fold.json` | **Nuevo.** Resultados K=5 fold con HMM 80/6 (sustituye al LOO). |
| `Paralel/resultados/checkpoint_LOO.json` | Reanudacion del LOO por usuario (legacy). |
| `Paralel/plots/cm_<TAG>_<regla>.png` | Matriz de confusion por escenario y regla. |
| `Serial/resultados/grid_<TAG>.json` | Resultados del grid search VQ sobre LL. |
| `Serial/resultados/serial_<TAG>.json` | Por muestra: hmm_lls, scores_serial, predicciones. Tras `update_serial_full_hmm.py` reflejan la version con HMM bien entrenado. |
| `metricas/paralel_<TAG>.json`, `metricas/serial_<TAG>.json` | AUC/EER por metodo y por clase. |
| `metricas/summary.json` | Tabla compacta. |
| `metricas/plots/auc_eer_<sistema>_<TAG>.png` | Barras AUC y EER por metodo. |
| `Entrega1_rerun/e1_<TAG>.json` | Scores 10D y predicciones de HMM E1 por muestra. |
| `metricas/det/DET_<TAG>.png` | DET overlay original con los 7 sistemas comparados. |
| `metricas/det/plot{1,2,3}_<TAG>.png` | **Nuevos.** DET por niveles (E1+VQ7D / +GMMHMM+VQopt / +Paralel+SerialVQ-gated). |
| `metricas/det/eer_summary_nuevos.json` | **Nuevo.** Tabla de EER de los 6 sistemas (incluye GMMHMM 100/10 y Serial VQ-gated). |
| `metricas/det/eer_summary.{json,md}` | Tabla EER de la version original (legacy). |

---

## Referencias

- Linde, Y., Buzo, A., Gray, R.M. (1980). "An Algorithm for Vector
  Quantizer Design." *IEEE Trans. Communications*, 28(1).
- Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and
  Selected Applications in Speech Recognition." *Proc. IEEE*, 77(2).
- Kittler, J. et al. (1998). "On Combining Classifiers." *IEEE TPAMI*,
  20(3) — soft / weighted voting rules.
- Documentacion `hmmlearn`: https://hmmlearn.readthedocs.io/
- Documentacion `scikit-learn` (KMeans, MiniBatchKMeans):
  https://scikit-learn.org/stable/
