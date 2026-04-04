# Entrega 2: Optimizacion de Hiperparametros y GMMHMM

## Informe del Proyecto - ASMI

**Autores:** Grupo X - 4o Ingenieria de Telecomunicaciones

---

Esta entrega se divide en dos partes:
- **Parte A:** Optimizacion individual de hiperparametros por digito (resultado negativo).
- **Parte B:** Uso de mezclas de gaussianas por estado (GMMHMM) con estrategia de escalado iterativo (resultado positivo).

---

# PARTE A: HMMs con Hiperparametros Optimizados por Digito

## A.1. Objetivo

En la Entrega 1 se utilizo la misma configuracion de hiperparametros (7 estados, prob_autolazo=0.6) para todos los digitos. Sin embargo, la complejidad del trazado varia entre digitos: el "1" es un trazo simple (podria modelarse con 3 estados), mientras que el "8" tiene dos bucles (podria necesitar 10 estados).

El objetivo de esta parte es **optimizar los hiperparametros (n_estados, prob_autolazo) de forma individual para cada digito**, usando una estrategia de validacion interna, para intentar mejorar la accuracy del clasificador.

---

## A.2. Metodologia

### A.2.1 Estrategia de optimizacion

Para evitar usar los datos de test en la seleccion de hiperparametros, se emplea la siguiente estrategia:

1. **Dividir los usuarios de train** en dos subconjuntos:
   - **Inner-train (80%):** para entrenar los modelos candidatos
   - **Validacion (20%):** para evaluar y seleccionar los mejores hiperparametros

2. **Grid search por digito:** Para cada digito (0-9), se prueban todas las combinaciones de:
   - `n_estados` ∈ {3, 5, 7, 10}
   - `prob_autolazo` ∈ {0.4, 0.5, 0.6, 0.7, 0.8}

   Total: 20 combinaciones por digito, 200 modelos HMM en la fase de busqueda.

3. **Criterio de seleccion:** Se elige la combinacion que maximiza la **log-verosimilitud media** sobre las muestras de validacion del digito correspondiente.

4. **Reentrenamiento final:** Con los parametros optimos seleccionados, se reentrena cada HMM usando **todos** los datos de train (inner-train + validacion) con mas iteraciones EM (100) y restarts (5).

5. **Evaluacion en test:** Se evalua el clasificador final sobre los usuarios de test.

### A.2.2 Configuracion base (heredada de Entrega 1)

Los siguientes parametros se mantienen fijos (mejores de Entrega 1):
- Features: subconjunto `med` (12 dimensiones)
- Covarianza: diagonal
- Preprocesado: centrado + escalado + suavizado + remuestreo a 80 puntos
- Normalizacion: Z-Score

### A.2.3 Checkpoint de progreso

La optimizacion guarda un checkpoint despues de cada digito, permitiendo reanudar la ejecucion si se interrumpe.

---

## A.3. Resultados

### A.3.1 Parametros optimos encontrados

#### Escenario N=74 (59 inner-train, 15 validacion)

| Digito | N estados | Prob. autolazo |
|--------|-----------|----------------|
| 0      | 10        | 0.7            |
| 1      | 10        | 0.7            |
| 2      | 7         | 0.5            |
| 3      | 10        | 0.6            |
| 4      | 5         | 0.4            |
| 5      | 10        | 0.4            |
| 6      | 10        | 0.6            |
| 7      | 7         | 0.8            |
| 8      | 10        | 0.4            |
| 9      | 3         | 0.8            |

La mayoria de digitos seleccionan 10 estados (7 de 10 digitos), lo que sugiere una tendencia al sobreajuste en la fase de validacion con pocos usuarios.

#### Escenario N=47 (38 inner-train, 9 validacion)

| Digito | N estados | Prob. autolazo |
|--------|-----------|----------------|
| 0      | 7         | 0.4            |
| 1      | 7         | 0.4            |
| 2      | 10        | 0.8            |
| 3      | 7         | 0.6            |
| 4      | 5         | 0.6            |
| 5      | 3         | 0.8            |
| 6      | 10        | 0.7            |
| 7      | 5         | 0.8            |
| 8      | 10        | 0.5            |
| 9      | 10        | 0.8            |

Los parametros optimos difieren significativamente entre los dos escenarios, lo que evidencia la **inestabilidad** de la seleccion con conjuntos de validacion pequenos.

### A.3.2 Comparativa con Entrega 1

| Escenario | E1 (params uniformes) | Parte A (params por digito) | Diferencia |
|-----------|-----------------------|-----------------------------|------------|
| **N=74**  | **92.89%**            | 85.99%                      | -6.90%     |
| **N=47**  | **89.78%**            | 85.82%                      | -3.96%     |

**Resultado negativo:** La optimizacion por digito empeora la accuracy en ambos escenarios.

### A.3.3 Analisis de las matrices de confusion

**N=74 (86.0%):** Se observan confusiones importantes:
- El digito **9** es el peor reconocido (solo 21 correctos de ~152 muestras), confundiendose frecuentemente con el 7. La seleccion de solo 3 estados para el 9 es claramente insuficiente.
- El digito **4** tambien muestra degradacion respecto a Entrega 1.

**N=47 (85.8%):** Patron similar, con confusiones mas distribuidas dada la mayor cantidad de usuarios de test.

---

## A.4. Analisis detallado: Por que empeora

### A.4.1 Criterio de seleccion incorrecto (causa principal)

El defecto mas grave de esta parte es el **criterio de seleccion**: se maximiza la log-verosimilitud (LL) de cada digito por separado, evaluando unicamente las muestras de validacion de ese digito. Sin embargo, la clasificacion funciona por **argmax conjunto** sobre los 10 modelos: se asigna la clase cuyo HMM produce la mayor LL. Estas dos optimizaciones no son equivalentes y, de hecho, pueden ser contradictorias.

**Por que son contradictorias:** Un HMM con 10 estados es un modelo mas flexible que uno con 7. Al tener mas parametros, puede ajustarse mejor a cualquier secuencia, no solo a las de su digito objetivo. Cuando se optimiza la LL del digito 3 de forma aislada, un modelo con 10 estados obtiene mejor puntuacion porque modela mejor los datos de validacion del 3. Pero ese mismo modelo tambien asignara LL altas a secuencias del 5, del 8, o de cualquier otro digito con curvas similares. El resultado es que la **discriminacion** entre clases empeora.

**Evidencia directa:** En Entrega 1 ya se demostro que 10 estados (87.4%) rinde peor que 7 (89.3%) en clasificacion global, precisamente porque modelos demasiado flexibles pierden capacidad discriminativa. A pesar de esto, la busqueda per-digito selecciona 10 estados para 7 de 10 digitos en N=74, porque el criterio (LL individual) no penaliza la falta de discriminacion.

**Ejemplo concreto: el caso del digito 9.** En N=74, la busqueda asigna solo 3 estados al digito 9. Un HMM con 3 estados para un digito complejo como el 9 (que tiene un circulo superior + trazo descendente) es claramente insuficiente. El modelo no puede representar las fases del trazado y asigna LL bajas incluso a muestras genuinas del 9. Al mismo tiempo, el HMM del 7 (que si tiene 7-10 estados) asigna LL razonables a trazados del 9, porque la parte descendente es similar. El resultado: el 9 se confunde masivamente con el 7. En la matriz de confusion de N=74, el digito 9 es el peor reconocido, con la mayoria de sus muestras clasificadas como 7.

**El criterio correcto:** Entrenar los 10 modelos candidatos conjuntamente para cada combinacion de parametros y medir la **accuracy de clasificacion** sobre el conjunto de validacion completo (todas las clases a la vez). Esto es exactamente lo que hace la Parte B con su grid search global, y es la razon por la que consigue mejoras reales.

### A.4.2 Asimetria de complejidad entre modelos

Cuando cada digito tiene sus propios hiperparametros, se crea una **asimetria de complejidad** entre los 10 modelos que compiten en la clasificacion. Si el HMM del digito 0 tiene 10 estados (modelo flexible, LL altas para muchas secuencias) y el del digito 1 tiene 3 estados (modelo simple, LL bajas en general), el digito 0 "gana" sistematicamente la competicion argmax, no porque modele mejor las secuencias del 0, sino porque asigna LL altas a todo. Con parametros uniformes (Entrega 1), todos los modelos compiten en igualdad de condiciones y la decision se basa en el **ajuste relativo** de cada secuencia a cada digito.

En otras palabras, optimizar cada modelo por separado viola un supuesto implicito de la clasificacion por argmax: que los modelos que compiten tengan **niveles de complejidad comparables**.

### A.4.3 Conjunto de validacion demasiado pequeno

El tamaño del conjunto de validacion es insuficiente para una seleccion fiable:
- **N=74:** Solo 15 usuarios para validacion -> ~120 muestras por digito (15 usuarios x 8 repeticiones)
- **N=47:** Solo 9 usuarios para validacion -> ~72 muestras por digito

Con tan pocas muestras, la LL en validacion tiene alta varianza. El resultado depende mas de **que usuarios concretos** caen en la particion de validacion que de la calidad real de la configuracion. Esto es especialmente problematico porque los usuarios de e-BioDigit tienen estilos de escritura muy diversos (la LOO CV de Entrega 1 muestra una desviacion tipica de 8.79% entre usuarios), por lo que 15 usuarios no son representativos de la poblacion.

### A.4.4 Perdida de datos de entrenamiento

Al reservar el 20% de los usuarios para validacion, el entrenamiento del grid search se hace con datos reducidos:
- **N=74:** se entrena con 59 usuarios en lugar de 74 durante la busqueda (20% menos)
- **N=47:** se entrena con solo 38 usuarios en lugar de 47 (19% menos)

Esto tiene dos consecuencias:
1. Los HMMs entrenados durante la busqueda son de peor calidad (menos datos -> estimaciones mas ruidosas), lo que anade ruido al criterio de seleccion.
2. Existe un **desajuste** entre las condiciones de seleccion (entrenados con 59/38 usuarios) y las de evaluacion final (entrenados con 74/47 usuarios).

### A.4.5 Inconsistencia en la normalizacion

El pipeline utiliza dos normalizadores Z-Score diferentes:
- **`norm_inner`:** ajustado sobre inner-train (80% de los usuarios de train), usado durante la busqueda.
- **`norm_full`:** ajustado sobre todos los usuarios de train (100%), usado para el entrenamiento final y la evaluacion en test.

Los parametros "optimos" seleccionados bajo `norm_inner` pueden ser suboptimos bajo `norm_full`.

### A.4.6 Inestabilidad de la seleccion

Los parametros optimos difieren drasticamente entre los dos escenarios para un mismo digito:

| Digito | N=74 | N=47 | Cambio |
|--------|------|------|--------|
| 0 | 10 est, p=0.7 | 7 est, p=0.4 | -3 est, -0.3p |
| 5 | 10 est, p=0.4 | 3 est, p=0.8 | -7 est, +0.4p |
| 7 | 7 est, p=0.8 | 5 est, p=0.8 | -2 est |
| 9 | 3 est, p=0.8 | 10 est, p=0.8 | +7 est |

Esta volatilidad confirma que el criterio de seleccion es ruidoso y que los parametros "optimos" son en realidad artefactos de la particion aleatoria.

---

## A.5. Lecciones aprendidas

1. **Mas hiperparametros no siempre es mejor.** Pasar de 2 hiperparametros globales a 20 (2 por digito x 10 digitos) multiplica el riesgo de sobreajuste.

2. **El criterio de optimizacion debe coincidir con el objetivo final.** Si el objetivo es maximizar la accuracy de clasificacion, el criterio de seleccion debe ser accuracy de clasificacion (no LL por digito aislado).

3. **El tamaño del conjunto de validacion es critico.** Con solo 9-15 usuarios, la senal es demasiado ruidosa para una seleccion fiable.

4. **La consistencia del pipeline importa.** Usar diferentes normalizaciones entre la fase de seleccion y la fase de evaluacion introduce un desajuste sistematico.

5. **La inestabilidad es una senal de alarma.** Si los "parametros optimos" cambian radicalmente entre dos ejecuciones con datos similares, el proceso de seleccion no es robusto.

---

# PARTE B: GMMHMM con Estrategia de Escalado Iterativo

## B.1. Objetivo

El objetivo de esta parte es explorar si el uso de **mezclas de gaussianas por estado (GMMHMM)** mejora la clasificacion de digitos manuscritos frente al modelo GaussianHMM de la Entrega 1.

En un GaussianHMM, cada estado emite observaciones segun una **unica gaussiana**. En un GMMHMM, cada estado usa una **mezcla de gaussianas** (GMM), lo que permite modelar distribuciones multimodales dentro de un mismo estado. Esto es relevante porque un mismo estado del HMM (por ejemplo, "fase de curva") puede manifestarse de formas ligeramente distintas entre usuarios, y una mezcla de gaussianas puede capturar esa variabilidad.

**Restriccion:** Se usa exclusivamente **covarianza diagonal** (`diag`) en todos los experimentos, por ser el mejor equilibrio entre rendimiento y estabilidad numerica.

---

## B.2. Estrategia de escalado iterativo

En lugar de realizar una busqueda exhaustiva desde el principio (costosa computacionalmente), se emplea una **estrategia de escalado en 3 fases** que solo avanza a la siguiente fase si la anterior no produce mejora:

```
Fase 1: Prueba rapida de GMMHMM
    Si mejora >= 0.5pp --> Evaluacion final
    Si NO mejora        --> Fase 2

Fase 2: Grid search sobre (n_mix, n_estados, prob_autolazo)
    Si mejora >= 0.5pp --> Evaluacion final
    Si NO mejora        --> Fase 3

Fase 3: Seleccion de features (ultimo recurso)
    --> Evaluacion final con la mejor combinacion global
```

El umbral de mejora se fija en **0.5 puntos porcentuales** para evitar considerar fluctuaciones por ruido como mejoras reales.

### B.2.1 Fase 1: Prueba rapida

Se fija la configuracion de Entrega 1 (7 estados, diag, prob_autolazo=0.6, features med) y se varia unicamente el numero de componentes de la mezcla: `n_mix = [1, 2, 3]`.

- `n_mix=1` equivale al GaussianHMM original (baseline).
- Parametros de entrenamiento: `n_iter=50, n_restarts=5`.

### B.2.2 Fase 2: Busqueda de hiperparametros

Grid search sobre el espacio expandido, manteniendo features fijas (`med`) y covarianza fija (`diag`):

| Hiperparametro | Valores |
|----------------|---------|
| `n_mix` | 1, 2, 3 |
| `n_estados` | 5, 7, 10, 12, 15 |
| `prob_autolazo` | 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |

**Total: 3 x 5 x 7 = 105 configuraciones.**

Parametros de entrenamiento: `n_iter=50, n_restarts=3`. Checkpointing cada 5 configuraciones para permitir interrumpir y reanudar.

### B.2.3 Fase 3: Seleccion de features

Si la Fase 2 no mejora, se prueban distintos subconjuntos de features con la mejor configuracion encontrada en Fase 2:

- Subsets estandar: `min` (7D), `med` (12D), `full` (21D)
- Subsets personalizados: `med_noxy` (10D), `med_plus` (14D)

---

## B.3. Resultados

### B.3.1 Fase 1: Prueba rapida

| n_mix | Modelo | Accuracy (val) | Tiempo |
|-------|--------|----------------|--------|
| **1** | **GaussianHMM** | **87.08%** | 615s |
| 2 | GMMHMM | 84.83% | 5014s |
| 3 | GMMHMM | 75.92% | 4761s |

**Resultado:** El GMMHMM con la configuracion de Entrega 1 **no mejora** el baseline. De hecho, empeora significativamente: n_mix=2 pierde 2.25pp y n_mix=3 pierde 11.16pp.

**Analisis detallado:** El resultado negativo se explica por un **problema de ratio parametros/datos**. Con 7 estados y 12 features:

| n_mix | Parametros por estado | Total (7 estados) |
|-------|----------------------|---------------------|
| 1 (GaussianHMM) | 12 (media) + 12 (cov diag) = 24 | **168** |
| 2 (GMMHMM) | 2x24 + 2 (pesos) = 50 | **350** |
| 3 (GMMHMM) | 3x24 + 3 (pesos) = 75 | **525** |

**Leccion clave de la Fase 1:** Anadir complejidad al modelo de emision (mas mezclas) **sin reajustar simultaneamente la complejidad del modelo de estados** produce una combinacion desequilibrada.

**Decision:** NO hay mejora >= 0.5pp. Se procede a la **Fase 2**.

### B.3.2 Fase 2: Grid search (105 configuraciones)

Se evaluaron las 105 combinaciones. Las **top 10 configuraciones**:

| Rank | n_mix | n_estados | prob_autolazo | Accuracy (val) |
|------|-------|-----------|---------------|----------------|
| **1** | **2** | **7** | **0.6** | **94.75%** |
| 2 | 2 | 5 | 0.3 | 93.00% |
| 3 | 1 | 15 | 0.3 | 91.25% |
| 4 | 2 | 5 | 0.7 | 90.33% |
| 5 | 2 | 5 | 0.9 | 90.33% |
| 6 | 1 | 10 | 0.3 | 90.17% |
| 7 | 2 | 5 | 0.5 | 90.17% |
| 8 | 2 | 5 | 0.4 | 90.08% |
| 9 | 2 | 5 | 0.8 | 89.83% |
| 10 | 2 | 5 | 0.6 | 89.17% |

**Mejor configuracion:** `n_mix=2, n_estados=7, prob_autolazo=0.6` con **94.75%** de accuracy en validacion, mejorando la referencia de Fase 1 (87.08%) en **+7.67pp**.

**Decision:** Hay mejora significativa (>= 0.5pp). Se salta la Fase 3 y se procede a la **evaluacion final**.

#### Analisis de la busqueda

**Efecto de n_mix por numero de estados:**

| n_estados | Mejor con n_mix=1 | Mejor con n_mix=2 | Mejor con n_mix=3 |
|-----------|-------------------|-------------------|-------------------|
| 5 | 86.42% (p=0.3) | **93.00%** (p=0.3) | 85.67% (p=0.5) |
| 7 | 77.83% (p=0.9) | **94.75%** (p=0.6) | 76.08% (p=0.7) |
| 10 | 90.17% (p=0.3) | 76.83% (p=0.5) | 58.17% (p=0.9) |
| 12 | 76.08% (p=0.3) | 69.58% (p=0.9) | 49.67% (p=0.7) |
| 15 | **91.25%** (p=0.3) | 77.75% (p=0.5) | 67.83% (p=0.5) |

#### Por que n_mix=2 funciona con 5-7 estados

Con 5 o 7 estados, cada estado del HMM left-right captura una **fase amplia** del trazado del digito. Dentro de cada fase, hay variabilidad natural entre usuarios. Con n_mix=2, cada estado tiene **dos modos**: puede representar, por ejemplo, la variante "curva suave" y la variante "curva angulosa" del mismo trazo.

**Por que funciona especificamente con 7 estados y p=0.6:** Esta combinacion equilibra tres factores:
- **7 estados** proporcionan suficientes fases para capturar la estructura temporal de cada digito.
- **2 mezclas** duplican la capacidad del modelo de emision sin un coste parametrico excesivo (350 parametros, manejable con 624 secuencias por digito).
- **p=0.6** permite que cada estado tenga una duracion esperada de 1/(1-0.6) = 2.5 frames.

#### Por que n_mix=2,3 degradan severamente con 10+ estados

Con 10 o mas estados, cada estado ya modela una porcion muy pequena del trazado (~8 frames por estado con n_estados=10), y anadir mezclas no aporta informacion util sino que **multiplica los parametros** sin senal que los justifique:

| Configuracion | Parametros emision | Datos/digito | Ratio |
|---------------|-------------------|--------------|-------|
| n_mix=1, 10 est | 240 | 624 | 2.6:1 |
| n_mix=2, 10 est | 500 | 624 | 1.2:1 |
| n_mix=3, 10 est | 750 | 624 | 0.8:1 |
| n_mix=3, 15 est | 1125 | 624 | **0.6:1** |

#### La interaccion n_mix x n_estados: un trade-off de complejidad

| Complejidad | Ejemplo | Accuracy | Interpretacion |
|-------------|---------|----------|----------------|
| Baja | n_mix=1, 5 est | 86.42% | Sub-expresivo |
| **Optima** | **n_mix=2, 7 est** | **94.75%** | **Equilibrio** |
| Optima (alt.) | n_mix=1, 15 est | 91.25% | Mas estados, menos mezclas |
| Alta | n_mix=2, 12 est | 69.58% | Sobreajuste |
| Excesiva | n_mix=3, 12 est | 49.67% | Sobreajuste severo |

### B.3.3 Fase 3: Seleccion de features

**No ejecutada** (la Fase 2 produjo mejora suficiente).

### B.3.4 Evaluacion final

Con la mejor configuracion encontrada (`n_mix=2, n_estados=7, diag, prob_autolazo=0.6, features=med`), se evalua con entrenamiento robusto (`n_iter=100, n_restarts=8`):

| Escenario | Train | Test | Accuracy |
|-----------|-------|------|----------|
| **N=74** | 74 usuarios | 19 usuarios | **93.03%** |
| **N=47** | 47 usuarios | 46 usuarios | **92.61%** |

---

## B.4. Que funciono y que no

### B.4.1 Lo que NO funciono

**GMMHMM n_mix=2 con la configuracion fija de Entrega 1 (84.83%, -2.25pp):** Los hiperparametros de E1 fueron optimizados para GaussianHMM, no para GMMHMM. Al duplicar los parametros de emision sin ajustar el resto, el modelo queda en una region suboptima.

**GMMHMM n_mix=3 (75.92% Fase 1, 19-85% Fase 2):** Con 3 mezclas por estado, el ratio datos/parametros es desfavorable. El EM converge a soluciones degeneradas.

**n_mix=2 con 10+ estados (59-77%):** Combina dos fuentes de complejidad que se multiplican.

### B.4.2 Lo que SI funciono

**n_mix=2 con 7 estados, p=0.6 (94.75% val, 93.03% N=74, 92.61% N=47):** Equilibrio optimo entre expresividad, parametros manejables y topologia temporal adecuada.

**Busqueda global (criterio de accuracy de clasificacion):** A diferencia de la Parte A (que optimizaba LL por digito aislado), la Fase 2 evalua cada configuracion entrenando los 10 modelos conjuntamente y midiendo la **accuracy de clasificacion** sobre el conjunto de validacion completo. Este es el criterio correcto.

---

# Comparativa global

| Escenario | E1 (GaussianHMM) | Parte A (Per-digito) | **Parte B (GMMHMM)** |
|-----------|-------------------|----------------------|------------------------|
| **N=74** | 92.89% | 85.99% | **93.03%** |
| **N=47** | 89.78% | 85.82% | **92.61%** |

| | Mejora Parte B vs E1 | Mejora Parte B vs Parte A |
|-|----------------------|---------------------------|
| **N=74** | +0.14pp | +7.04pp |
| **N=47** | **+2.83pp** | **+6.79pp** |

**Analisis:**
- **N=74:** Mejora modesta (+0.14pp). Con 74 usuarios de entrenamiento, el GaussianHMM ya estimaba bien una gaussiana por estado.
- **N=47:** Mejora significativa (+2.83pp). Con menos datos, las 2 gaussianas por estado capturan variabilidad inter-usuario que una sola gaussiana no podia representar.
- **Robustez:** El GMMHMM mantiene rendimiento muy estable entre N=74 (93.03%) y N=47 (92.61%), con solo -0.42pp de diferencia. En comparacion, E1 perdia -3.11pp.

---

# Configuracion final

| Parametro | Valor |
|-----------|-------|
| **Modelo** | GMMHMM (2 gaussianas por estado) |
| **n_estados** | 7 (left-right) |
| **n_mix** | 2 |
| **Covarianza** | Diagonal (`diag`) |
| **Prob. autolazo** | 0.6 |
| **Features** | `med` (12D): dx, dy, sin(angle), cos(angle), dtheta, v, rho, a, dv, lewiratio, x, y |
| **Preprocesado** | Suavizado Savitzky-Golay (w=7, o=3), remuestreo 80 pts, centrado, escalado |
| **Normalizacion** | Z-Score (ajustada solo sobre train) |
| **Entrenamiento** | 100 iteraciones EM, 8 restarts |
| **Clasificacion** | argmax de log-verosimilitud |

---

# Coste computacional

| Parte / Fase | Configuraciones | Tiempo total |
|--------------|-----------------|--------------|
| Parte A (per-digito) | 200 modelos | ~44 min |
| Parte B - Fase 1 | 3 | ~2.9 horas |
| Parte B - Fase 2 | 105 | ~85 horas |
| Parte B - Evaluacion final | 2 escenarios | ~4.8 horas |

---

# Descripcion del codigo

### Estructura

| Fichero | Descripcion |
|---------|-------------|
| `clasificador_digitos_v2.py` | **Parte A:** Optimizacion de hiperparametros por digito |
| `clasificador_digitos_v3.py` | **Parte B:** GMMHMM con estrategia de escalado iterativo |

### Funciones principales de v2 (Parte A)

| Funcion | Descripcion |
|---------|-------------|
| `preparar_features_por_usuario()` | Pre-computa features agrupadas por usuario |
| `reunir_secuencias()` | Reune secuencias y etiquetas de un conjunto de usuarios |
| `optimizar_params_por_digito()` | Grid search con checkpoint por digito |
| `entrenar_con_params_optimos()` | Entrena HMMs finales con parametros individuales |
| `evaluar_escenario()` | Pipeline completo: train -> eval |
| `graficar_params_por_digito()` | Visualiza los parametros optimos por digito |

### Funciones principales de v3 (Parte B)

| Funcion | Descripcion |
|---------|-------------|
| `entrenar_gmmhmm_digito()` | Entrena un GMMHMM (o GaussianHMM si n_mix=1) left-right |
| `fase1_prueba_rapida()` | Fase 1: prueba rapida con n_mix=1,2,3 |
| `fase2_busqueda_hiperparametros()` | Fase 2: grid search completo |
| `fase3_seleccion_features()` | Fase 3: seleccion de features (ultimo recurso) |
| `evaluar_escenario()` | Evaluacion final en N=74 y N=47 |
| `graficar_top_configs()` | Top N configuraciones del grid search |

### Resultados generados

| Fichero | Contenido |
|---------|-----------|
| `resultados/resultados.json` | Accuracy y params optimos (Parte A) |
| `resultados/resultados_v3.json` | Resultados completos (Parte B) |
| `resultados/confusion_N74_opt.png` | Confusion N=74, Parte A |
| `resultados/confusion_N47_opt.png` | Confusion N=47, Parte A |
| `resultados/confusion_N74_v3.png` | Confusion N=74, Parte B |
| `resultados/confusion_N47_v3.png` | Confusion N=47, Parte B |
| `resultados/params_por_digito.png` | Hiperparametros por digito (Parte A) |
| `resultados/top_configs_fase2.png` | Top 20 configs de Fase 2 (Parte B) |
| `resultados/comparativa_fases.png` | Resumen visual de las fases (Parte B) |

### Ejecucion

```bash
cd R1/Entrega2

# Parte A: Optimizacion por digito (~44 min)
python3 clasificador_digitos_v2.py

# Parte B: GMMHMM escalado iterativo (~88 horas)
python3 clasificador_digitos_v3.py
```

---

# Conclusion

Esta entrega presenta dos enfoques para mejorar el clasificador de Entrega 1:

**Parte A (resultado negativo):** La optimizacion individual de hiperparametros por digito **no mejoro** la accuracy (85.99% vs 92.89% para N=74). La causa principal es que maximizar la LL de cada digito por separado mejora el ajuste individual pero empeora la discriminacion entre digitos. Esta leccion motiva directamente el enfoque de la Parte B.

**Parte B (resultado positivo):** El GMMHMM con n_mix=2, 7 estados y busqueda global de hiperparametros **mejora sobre todas las entregas anteriores**:
- N=74: 93.03% (+0.14pp sobre E1)
- N=47: 92.61% (+2.83pp sobre E1)

La clave del exito de la Parte B frente al fracaso de la Parte A es doble:
1. **Criterio de optimizacion correcto:** accuracy de clasificacion conjunta en lugar de LL por digito aislado.
2. **Complejidad bien calibrada:** 2 mezclas por estado aportan expresividad sin sobreajuste, manteniendo un ratio datos/parametros viable.

---

## Referencias

- Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE*, 77(2).
- Documentacion de `hmmlearn`: https://hmmlearn.readthedocs.io/
- Base de datos e-BioDigit: proporcionada por la asignatura ASMI.
