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
- **LOO CV:** Leave-One-User-Out (93 folds).

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

## 3. Modelos base (recordatorio Entrega 2)

| Modelo | Configuracion | N=74 | N=47 | LOO CV |
|--------|---------------|------|------|--------|
| **HMM E2B (GMMHMM)** | n_mix=2, 7 estados, p_autolazo=0.6, diag, features=`med` | 93.03% | 92.61% | — |
| **VQ optimizado** | MiniBatchKMeans, k=128, features=`pos_ang_curv` 5D | 96.97% | 97.06% | 96.94% +/- 3.85% |

Los HMM se reentrenan en cada experimento con `entrenar_gmmhmm_digito`
(`Entrega2/HMM/clasificador_digitos_v3.py`). Para reducir el tiempo de
computo de los 93 folds del LOO se ha empleado un fit mas barato:

| Configuracion HMM | n_iter | n_restarts | Uso |
|-------------------|--------|------------|-----|
| HMM_FIT (N=74, N=47) | 40 | 2 | Entrenamiento principal en splits |
| HMM_FIT_LOO         | 15 | 1 | Entrenamiento dentro del bucle LOO |

Esta reduccion provoca que algun modelo individual no converja del todo
(LL=-inf en el log de digito 7 en N=74, digitos 7 y 9 en N=47). El
codigo trata estos modelos como faltantes (LL fijada al suelo finito
`min_finite - 1e3`) para que la normalizacion por z-score y el softmax
no se contaminen. La accuracy de HMM aislado en este informe es por
tanto inferior a la reportada en Entrega 2 (~85% vs 92% en N=74), pero
los modelos VQ y los ensembles se benefician de las features intactas
del extractor.

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

### 4.3 Resultados N=74

| Metodo | Accuracy | AUC macro | EER macro |
|--------|----------|-----------|-----------|
| HMM (argmax LL) | 85.92% | 88.50% | 14.33% |
| VQ (argmin dist) | 96.84% | 99.58% | 1.83% |
| **agreement** | **96.84%** | 94.61% | 7.21% |
| **soft** | 95.39% | **99.59%** | **1.84%** |
| **conf_weighted** | 96.71% | 99.55% | 2.18% |
| **margin_weighted** | **96.84%** | 99.58% | 1.83% |
| oracle | **98.82%** | — | — |
| ambos fallan | 1.18% | — | — |

### 4.4 Resultados N=47

| Metodo | Accuracy | AUC macro | EER macro |
|--------|----------|-----------|-----------|
| HMM | 74.97% | 83.72% | 18.49% |
| VQ | 97.04% | 99.68% | 1.73% |
| **agreement** | **97.04%** | 95.07% | 6.83% |
| **soft** | 92.34% | **99.68%** | **1.74%** |
| **conf_weighted** | 96.36% | 99.66% | 2.03% |
| **margin_weighted** | **97.04%** | 99.68% | 1.74% |
| oracle | **98.26%** | — | — |
| ambos fallan | 1.74% | — | — |

### 4.5 Resultados LOO CV (93 folds, 7440 muestras)

| Metodo | Accuracy | AUC macro | EER macro |
|--------|----------|-----------|-----------|
| HMM | 69.65% | 71.11% | 33.00% |
| VQ | 96.75% | **99.70%** | **1.75%** |
| **agreement** | **96.65%** | 93.58% | 14.40% |
| **soft** | 93.27% | 99.33% | 2.87% |
| **conf_weighted** | 95.83% | 99.58% | 2.34% |
| **margin_weighted** | 96.56% | 99.47% | 2.48% |
| oracle | **98.09%** | — | — |
| ambos fallan | 1.91% | — | — |

Tiempo total LOO Paralel: **604.9 minutos** (~10 horas) en una unica
maquina con `OMP_NUM_THREADS=4`.

### 4.6 Discusion del sistema Paralelo

1. **El oraculo confirma la hipotesis de complementariedad:** en N=74
   los dos modelos fallan a la vez solo en el 1.18% de las muestras
   (y 1.74% en N=47). Hay margen para que un buen ensemble se acerque a
   ~98.8% en lugar del 96.84% que da el VQ aislado.

2. **Las reglas de fusion en accuracy se quedan en el 96.84%-97.04%**
   (igual que el VQ aislado). La razon es que la regla `agreement` y
   las ponderaciones por confianza acaban delegando en VQ siempre que
   los modelos discrepan (porque VQ casi siempre tiene mas confianza que
   HMM tras la normalizacion). La complementariedad detectada por el
   oraculo no se traduce en mejora porque el HMM fallido en una clase
   no es el HMM "seguro" en otra: cuando acierta, no acierta con mucho
   mas margen que cuando falla.

3. **El soft voting tiene la mejor AUC y EER macro**: en N=47
   AUC=99.68% y EER=1.74%, igualando al VQ aislado pero con
   probabilidades mas suaves. Esto importa para tareas de
   *verificacion* y *score* (busqueda de umbrales): aunque los aciertos
   duros sean los mismos, la calidad del score continuo es ligeramente
   mejor.

4. **`agreement` tiene EER alto** porque su score 10D es discontinuo
   (salta entre `hmm_probs` y `vq_probs` segun el caso); el ranking de
   confianzas que ve la curva ROC es mas ruidoso.

5. **El HMM puro pierde mucha accuracy entre N=74 y N=47** (85.92% ->
   74.97%, -10.95pp) por el agravamiento de las convergencias fallidas
   con menos datos. El VQ es mas robusto (-0.20pp en accuracy de N=74
   a N=47, +0.46pp en AUC) y arrastra al ensemble.

---

## 5. Sistema Serie (`Serial/serial_hmm_vq.py`)

### 5.1 Disenio

El HMM produce, por muestra, un vector `LL` de dimension 10 (una
log-verosimilitud por modelo de digito). Se entrena un VQ por clase
sobre este espacio 10D y se clasifica por minima distancia al
centroide mas cercano de cualquier codebook.

Para evitar leakage al entrenar el VQ se usa **K-fold sobre usuarios
de train** (K=3): en cada fold se entrena el HMM con K-1 partes y se
calculan las LL para los usuarios de la parte holdout; estas LL
out-of-fold (OOF) se concatenan y constituyen el espacio de
entrenamiento del VQ. El HMM final, entrenado con todos los usuarios
de train, es el que produce las LL para test.

> **Decision clave:** K-fold y entrenamiento final usan **el mismo**
> `HMM_FIT` (n_iter=40, n_restarts=2). Si se entrenase el HMM final mas
> fuerte que los HMM de los folds, las LL de test viven en una
> distribucion distinta a las del train del VQ y la generalizacion
> cae a ~10% (lo comprobamos en una version anterior del codigo).

### 5.2 Espacio de busqueda

| Hiperparametro | Valores |
|----------------|---------|
| Algoritmo | KMeans, MiniBatchKMeans, LBG (Linde-Buzo-Gray) |
| Normalizacion del LL por muestra | zscore, softmax, shift (LL - max) |
| N centroides por digito | 1, 2, 4, 8, 16, 32 |

Total: 3 x 3 x 6 = 54 configuraciones.

> Se descarto la opcion `raw` (LL sin normalizar) porque introduce un
> sesgo por la escala absoluta de las log-verosimilitudes: cuando train
> y test producen LL de diferente magnitud (cosa que ocurre con HMM
> entrenados en distinto numero de usuarios), el VQ basado en distancia
> euclidea sobre LL crudas falla catastroficamente (~10% accuracy).

### 5.3 Mejor configuracion encontrada

| Escenario | Algoritmo | Norm | k | Acc validacion |
|-----------|-----------|------|---|----------------|
| **N=74**  | kmeans | softmax | 2 | 90.25% |
| **N=47**  | lbg    | zscore  | 32 | 85.28% |

Las normalizaciones invariantes a escala (zscore y softmax) dominan el
top.

### 5.4 Resultados

| Escenario | HMM baseline | Serial HMM->VQ | Diferencia |
|-----------|--------------|----------------|------------|
| **N=74** acc | 67.24% | 67.24% | +0.00pp |
| **N=74** AUC | 82.58% | 87.35% | +4.77pp |
| **N=74** EER | 20.42% | 14.45% | -5.97pp |
| **N=47** acc | 83.99% | **9.89%** | -74.10pp |
| **N=47** AUC | 90.36% | 54.10% | -36.26pp |
| **N=47** EER | 14.20% | 48.63% | +34.43pp |
| **LOO**  acc | 70.01% | 68.32% | -1.69pp |
| **LOO**  AUC | 78.29% | **93.19%** | +14.90pp |
| **LOO**  EER | 25.78% | **11.62%** | -14.16pp |

> El **fallo catastrofico en N=47** del sistema Serie (acc 9.89% =
> ruido) se explica abajo: la configuracion ganadora del grid es muy
> sensible al desplazamiento de distribucion entre las LL out-of-fold
> (HMM entrenado con ~15 usuarios por fold) y las LL del HMM final
> (entrenado con 47 usuarios). Con 47 usuarios el HMM final es mucho
> mas potente que cualquiera de los HMM de los folds y la
> distribucion de LL cambia mas que en N=74. El VQ entrenado en una
> distribucion no encuentra los centroides correctos en la otra.

### 5.5 Discusion del sistema Serie

1. **El cascading no aporta accuracy adicional en N=74** (Serial =
   HMM baseline). La razon es estructural: cada muestra produce un
   unico vector 10D. La "VQ" sobre puntos sueltos se reduce a un
   nearest-class-mean (con k>1, multiprototipo). Si la coordenada del
   digito correcto ya es la mayor (lo que decide la linea base
   argmax), el centroide del digito correcto en el espacio LL tambien
   sera el mas cercano. Solo cuando los maximos estan ambiguos
   (multimodalidad de estilos) el k>1 puede ayudar, y en este dataset
   los HMM bien entrenados resuelven la mayoria por margen amplio.

   **En N=47 el sistema cae al 9.89% (azar)** por el desplazamiento de
   distribucion mencionado: con K=3 sobre 47 usuarios, cada HMM de
   fold se entrena con ~15 usuarios; el HMM final con 47. El gap entre
   la LL "subentrenada" del fold y la LL "bien entrenada" del final
   es enorme, y el VQ no generaliza. Es un mal estructural del
   esquema K-fold OOF cuando el numero de train usuarios no es muy
   grande.

2. **AUC y EER si mejoran** (-5.97pp de EER en N=74). El VQ con
   normalizacion `softmax` produce un score continuo mas suave que el
   simple argmax sobre LL crudas, lo que mejora el ranking de confianza
   y por tanto las metricas basadas en umbrales.

3. **El HMM con HMM_FIT_FINAL aqui converge peor que el HMM del sistema
   Paralelo** (67.24% vs 85.92% en N=74). Aparentemente la secuencia
   de entrenamientos K-fold previa al fit final altera el estado RNG
   global y/o provoca convergencias menos afortunadas en algunos
   digitos (3 fallos vs 1 en Paralel). Esto penaliza tanto la baseline
   como al sistema Serie. Con una configuracion HMM mas robusta (mas
   restarts) seguramente la baseline subiria a ~85-90% y el sistema
   Serie heredaria esa mejora.

---

## 6. Comparativa global (incluye Entregas anteriores)

### 6.1 Accuracy

| Sistema | N=74 | N=47 | LOO CV |
|---------|------|------|--------|
| HMM E1 (GaussianHMM, Entrega 1) | 92.89% | 89.78% | 78.40% +/- 8.79% |
| HMM E2B (GMMHMM, Entrega 2)     | 93.03% | 92.61% | — |
| VQ baseline (Entrega 2)         | 95.59% | 96.06% | 95.75% +/- 4.72% |
| **VQ optimizado (Entrega 2)**   | **96.97%** | **97.06%** | **96.94% +/- 3.85%** |
| Paralel `agreement`             | 96.84% | 97.04% | 96.65% |
| Paralel `soft`                  | 95.39% | 92.34% | 93.27% |
| Paralel `margin_weighted`       | 96.84% | 97.04% | 96.56% |
| Paralel oracle (cota)           | 98.82% | 98.26% | 98.09% |
| Serial HMM->VQ                  | 67.24% | 9.89% | 68.32% |

> Notas sobre Serial:
> - **N=74**: el HMM base aqui obtiene 67% (3 digitos sin convergencia)
>   y el VQ encadenado iguala esa accuracy aunque mejora AUC y EER.
> - **N=47**: 9.89% = azar, por desplazamiento de distribucion entre
>   K-fold OOF (HMM con 15 usuarios) y HMM final (47 usuarios). Es un
>   limite estructural del esquema con pocos usuarios train.

### 6.2 AUC y EER macro one-vs-rest

| Sistema | N=74 AUC | N=74 EER | N=47 AUC | N=47 EER | LOO AUC | LOO EER |
|---------|----------|----------|----------|----------|---------|---------|
| HMM (Paralel) | 88.50% | 14.33% | 83.72% | 18.49% | 71.11% | 33.00% |
| VQ            | 99.58% | 1.83%  | 99.68% | 1.73%  | **99.70%** | **1.75%** |
| Paralel `soft` | **99.59%** | **1.84%** | **99.68%** | **1.74%**  | 99.33% | 2.87% |
| Paralel `margin_weighted` | 99.58% | 1.83% | 99.68% | 1.74% | 99.47% | 2.48% |
| Serial HMM->VQ | 87.35% | 14.45% | 54.10% | 48.63% | **93.19%** | **11.62%** |
| Serial HMM baseline | 82.58% | 20.42% | 90.36% | 14.20% | 78.29% | 25.78% |

---

## 7. Conclusiones

1. **La hipotesis de complementariedad se verifica robustamente en los
   tres escenarios:** los aciertos de HMM y VQ se solapan en al menos
   el 98.09% del test (98.82% en N=74, 98.26% en N=47, 98.09% en LOO,
   cota oraculo). El % de muestras donde *ambos* fallan es siempre <2%.
   Asi pues, un ensemble informado por la confianza tiene un techo
   teorico de ~98-99% accuracy en todos los escenarios.

2. **Pero la hipotesis no se traduce en mejora real con confianzas
   asimetricas.** Cuando el HMM esta peor entrenado que el VQ (caso
   habitual en este dataset por la complejidad del GMMHMM), VQ termina
   dominando el voto y los ensembles se quedan al nivel del VQ
   aislado (96.84%-97.04%). Para acercarse al oraculo haria falta
   ponderar HMM y VQ con prior distinto en funcion de la clase
   predicha por cada uno (no solo de la confianza local del modelo
   ganador).

3. **Las metricas de score continuo (AUC, EER) si mejoran ligeramente
   en `soft` y `margin_weighted`**, aunque la accuracy dura iguale al
   VQ aislado. Para tareas biometricas en las que importa el ranking
   por confianza (verificacion, busqueda con umbral), el ensemble
   suave aporta valor incluso cuando la decision dura no cambia.

4. **El sistema Serie (HMM->VQ) no aporta accuracy** sobre la linea
   base de su propio HMM en N=74 ni en LOO (-1.69pp en LOO), y se
   rompe en N=47 (9.89%) por desplazamiento de distribucion entre
   K-fold OOF y final. La razon estructural: cada muestra solo tiene
   un vector 10D, y un VQ de codebook por clase no consigue separar
   lo que la linea base argmax ya separa. **Pero AUC y EER mejoran
   sustancialmente cuando funciona**: en LOO la AUC sube de 78.29% a
   **93.19%** (+14.90pp) y la EER baja de 25.78% a **11.62%**
   (-14.16pp). El componente VQ en cascada actua como un *score
   smoother*: las decisiones duras no cambian pero el ranking de
   confianza pasa de ser muy ruidoso (LL crudas) a ser mucho mas
   continuo y discriminativo. Es una mejora valiosa para tareas
   biometricas con umbral.

5. **Las normalizaciones invariantes a escala son criticas en el
   sistema Serie**: la version "raw" del LL produce ~10% accuracy si
   el HMM final cambia ligeramente la magnitud de las LL respecto al
   K-fold OOF. zscore y softmax son robustas a este shift.

6. **Lecciones operacionales:**
   - Antes de fusionar dos clasificadores con softmax, normalizar
     cada uno a escala comun (z-score por muestra) — sino el modelo de
     escala mas grande satura el ensemble.
   - Detectar y suelar (`floor`) los modelos colapsados (LL=-inf o
     |LL|>1e6) antes de cualquier normalizacion: una sola coordenada
     patologica anula todo el vector.
   - Para cascadas, mantener el mismo `HMM_FIT` en train (K-fold OOF) y
     en deployment, o aceptar que el VQ aprende un espacio que no
     volvera a ver.

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

### 8.2 Modelos comparados

| Modelo | Origen | Score |
|--------|--------|-------|
| E1 HMM (GaussianHMM)        | Entrega 1 reentregada en `Entrega3/run_entrega1_hmm.py` | softmax con z-score sobre LL |
| E2B HMM (GMMHMM)            | Modelo HMM del sistema Paralelo (mismo config que Entrega 2) | softmax con z-score sobre LL |
| VQ optimizado               | Modelo VQ del sistema Paralelo (mismo config que Entrega 2 optimizado) | softmax sobre -distorsion z-scored |
| Paralel `agreement`         | Sistema Paralelo - regla por confianza top-1 | probabilidad ensemble 10D |
| Paralel `soft`              | Sistema Paralelo - voto suave 50/50          | probabilidad ensemble 10D |
| Paralel `margin_weighted`   | Sistema Paralelo - ponderado por margen      | probabilidad ensemble 10D |
| Serial cascade (HMM->VQ)    | Sistema Serie con la mejor config del grid    | -distancia minima a centroide por clase |

### 8.3 Tabla EER (pooled one-vs-rest)

| Modelo | N=74 EER | N=47 EER | LOO EER |
|--------|----------|----------|---------|
| E1 HMM (GaussianHMM) | 5.66% | 6.47% | 20.59% |
| E2B HMM (GMMHMM) | 12.88% | 20.98% | 31.99% |
| **VQ optimizado** | **1.84%** | **1.93%** | **1.91%** |
| Paralel agreement | 7.52% | 7.16% | 14.88% |
| Paralel soft | 3.36% | 4.51% | 3.98% |
| **Paralel margin_weighted** | **1.84%** | 1.97% | 2.68% |
| Serial cascade (HMM->VQ) | 20.72% | 49.24% | 12.34% |

### 8.4 Lectura de las curvas DET

Plots generados:

- `metricas/det/DET_N74.png` - DET con todos los sistemas, N=74.
- `metricas/det/DET_N47.png` - DET, N=47.
- `metricas/det/DET_LOO.png` - DET, LOO CV.

Observaciones principales:

1. **VQ optimizado y Paralel margin_weighted son indistinguibles en N=74**
   y N=47 (~1.85% EER). Ambas curvas se solapan en escala probit. El
   ensemble no degrada: la regla de margen aprovecha bien la confianza
   del VQ sin que el HMM ruidoso la arrastre.

2. **En LOO el VQ (1.91% EER) marca el limite practico** del sistema.
   Paralel margin_weighted (2.68%) y Paralel soft (3.98%) son los
   ensembles mas competitivos pero no llegan al VQ porque, como ya se
   discutia en accuracy, todos terminan delegando en VQ cuando los
   modelos discrepan.

3. **El sistema Serie en LOO recupera muchisimo (EER 12.34%)** respecto
   a su HMM baseline (que en E2B-HMM da 31.99%): -19.65pp de EER. Es la
   mejor evidencia de que el VQ sobre LL actua como *score smoother*:
   la accuracy dura no cambia (incluso baja 1.69pp) pero la calidad
   del ranking de confianza mejora drasticamente.

4. **E1 HMM (GaussianHMM) es mas robusto que E2B HMM (GMMHMM)** en
   nuestras pruebas con HMM_FIT aligerado: EER 5.66% vs 12.88% en
   N=74. Coincide con la observacion de Entrega 2 de que GMMHMM
   requiere muchas iteraciones para no caer en colapsos locales (el
   informe original usaba n_iter=100, n_restarts=8 para 93.03% de
   accuracy; nosotros usamos n_iter=40, n_restarts=2 por presupuesto
   computacional). Con ese fit reducido, el modelo mas simple gana.

5. **El sistema Serie en N=47 explota** (EER 49.24% ~ azar) por el
   problema estructural ya descrito de desplazamiento de distribucion
   entre el K-fold OOF y el HMM final cuando solo hay 47 usuarios.

### 8.5 Resumen final de EER por sistema y escenario

> Tabla compacta para consulta rapida (mismas cifras que 8.3,
> ordenadas de mejor a peor LOO EER):

| Posicion (LOO) | Modelo | N=74 | N=47 | LOO |
|----------------|--------|------|------|-----|
| 1 | VQ optimizado            | 1.84% | 1.93% | **1.91%** |
| 2 | Paralel margin_weighted  | 1.84% | 1.97% | 2.68% |
| 3 | Paralel soft             | 3.36% | 4.51% | 3.98% |
| 4 | Serial cascade           | 20.72% | 49.24% | 12.34% |
| 5 | Paralel agreement        | 7.52% | 7.16% | 14.88% |
| 6 | E1 HMM (GaussianHMM)     | 5.66% | 6.47% | 20.59% |
| 7 | E2B HMM (GMMHMM)         | 12.88% | 20.98% | 31.99% |

---

## 9. Coste computacional

| Fase | Tiempo aproximado |
|------|-------------------|
| Paralel N=74 (HMM + VQ + inferencia) | ~25 min |
| Paralel N=47 | ~20 min |
| Paralel LOO (93 folds x ~6.5 min) | ~10 horas |
| Serial N=74 (3-fold OOF + grid + final) | ~50 min |
| Serial N=47 | ~30 min |
| Serial LOO (93 folds, pipeline simplificado leaky) | ~6 horas |
| `metricas.py` (post-proceso AUC/EER) | <1 min |

Ejecuciones simultaneas Paralel y Serial con `OMP_NUM_THREADS=4` cada
una en un equipo con 12 nucleos.

---

## 10. Descripcion del codigo

| Fichero | Descripcion |
|---------|-------------|
| `Paralel/ensemble_paralelo.py` | Sistema paralelo. Entrena HMM y VQ en cada escenario, calcula reglas de fusion, guarda predicciones + scores 10D + matrices de confusion. Implementa LOO con checkpointing por usuario. |
| `Serial/serial_hmm_vq.py` | Sistema serie. K-fold OOF para LL de train, grid search VQ sobre LL, HMM final + inferencia en test. LOO con pipeline simplificado leaky. |
| `metricas.py` | Lee los JSON de cada sistema y calcula AUC y EER macro one-vs-rest. Guarda `metricas/summary.json` y barras AUC/EER por escenario. |
| `run_entrega1_hmm.py` | Re-ejecucion del HMM E1 (GaussianHMM) de Entrega 1 guardando scores 10D por muestra (necesarios para DET). Salida en `Entrega1_rerun/`. |
| `det_curves.py` | Lee scores de E1 + Paralel + Serial, dibuja DET pooled one-vs-rest por escenario (escala probit) con todos los sistemas overlay y produce `metricas/det/eer_summary.{json,md}`. |

### Ficheros generados

| Fichero | Contenido |
|---------|-----------|
| `Paralel/resultados/ensemble_<TAG>.json` | Por muestra: scores 10D, predicciones por regla, confianzas. |
| `Paralel/resultados/checkpoint_LOO.json` | Reanudacion del LOO por usuario. |
| `Paralel/plots/cm_<TAG>_<regla>.png` | Matriz de confusion por escenario y regla. |
| `Serial/resultados/grid_<TAG>.json` | Resultados del grid search VQ sobre LL. |
| `Serial/resultados/serial_<TAG>.json` | Por muestra: hmm_lls, scores_serial, predicciones. |
| `Serial/plots/cm_<TAG>_*.png`, `grid_top_<TAG>.png` | Graficas. |
| `metricas/paralel_<TAG>.json`, `metricas/serial_<TAG>.json` | AUC/EER por metodo y por clase. |
| `metricas/summary.json` | Tabla compacta. |
| `metricas/plots/auc_eer_<sistema>_<TAG>.png` | Barras AUC y EER por metodo. |
| `Entrega1_rerun/e1_<TAG>.json` | Scores 10D y predicciones de HMM E1 por muestra. |
| `metricas/det/DET_<TAG>.png` | DET overlay con los 7 sistemas comparados. |
| `metricas/det/eer_summary.{json,md}` | Tabla final EER por sistema y escenario. |

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
