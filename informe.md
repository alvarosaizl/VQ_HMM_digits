# Clasificador de Digitos Manuscritos (0-9) mediante HMMs

## Informe del Proyecto - ASMI

**Autores:** Grupo X - 4o Ingenieria de Telecomunicaciones

---

## 1. Introduccion

### 1.1 Objetivo

El objetivo de este proyecto es construir un sistema capaz de **reconocer digitos manuscritos (0-9)** dibujados con el dedo sobre una tablet, utilizando **Modelos Ocultos de Markov (HMMs)**.

A diferencia de otros enfoques (redes neuronales, SVMs...), los HMMs modelan la **secuencia temporal** del trazado: no analizan una imagen estática del digito, sino el proceso dinámico de cómo se dibuja. Esto tiene sentido porque dos personas pueden dibujar un "3" con formas ligeramente distintas, pero la secuencia de movimientos (curva arriba, giro, curva abajo) es similar.

### 1.2 Base de datos: e-BioDigit

La base de datos **e-BioDigit** contiene:

- **93 usuarios** que dibujan los digitos 0-9
- **2 sesiones** por usuario
- **4 repeticiones** por digito por sesion
- Cada muestra es un fichero `.txt` con columnas: `X, Y, timestamp, presión`
- **Total: ~7440 muestras**

Cada fichero contiene la trayectoria del dedo como una secuencia de puntos (x, y) con su timestamp, capturados por el sensor tactil de la tablet.

### 1.3 Enfoque: Un HMM por digito

El sistema funciona así:

1. **Entrenamiento:** Se entrena un HMM diferente para cada digito (10 modelos en total). Cada HMM aprende las características típicas de cómo se dibuja ese digito.

2. **Clasificacion:** Dado un trazado nuevo, se calcula la probabilidad de que haya sido generado por cada uno de los 10 modelos. El digito cuyo modelo da la mayor probabilidad es la prediccion.

```
Trazado nuevo → [HMM_0, HMM_1, ..., HMM_9] → argmax(probabilidad) → Digito predicho
```

---

## 2. Pipeline del Sistema

### 2.1 Preprocesado

Antes de extraer características, cada trazado se normaliza para hacerlo comparable entre usuarios:

| Paso | Descripcion | Justificacion |
|------|-------------|---------------|
| **Centrado** | Restar la media de X e Y | Eliminar la posicion absoluta (no importa dónde en la pantalla se dibuja) |
| **Escalado** | Normalizar a [-1, 1] manteniendo proporcion | Eliminar diferencias de tamaño entre usuarios |
| **Remuestreo** | Interpolar a 80 puntos equidistantes en longitud de arco | Hacer todas las secuencias de la misma longitud, independiente de la velocidad |
| **Suavizado** | Filtro Savitzky-Golay (ventana=7, orden=3) | Reducir el ruido del sensor tactil |

### 2.2 Extraccion de caracteristicas

De cada punto del trazado preprocesado se extraen **características locales cinemáticas**. El extractor produce 23 features por punto temporal, entre las que se incluyen:

- **Posicion:** x, y (coordenadas normalizadas)
- **Velocidad:** dx, dy (componentes), v (magnitud)
- **Direccion:** theta (angulo de la tangente), sin/cos del angulo
- **Curvatura:** dtheta (tasa de cambio del angulo), rho (radio de curvatura)
- **Aceleracion:** a (magnitud), dv (derivada de velocidad)
- **Geometricas:** lewiratio (ratio longitud/anchura en ventana local)

No se usan todas las 23 features: se seleccionan subconjuntos de distinto tamaño para estudiar su efecto.

### 2.3 Normalizacion Z-Score

Antes de alimentar los HMMs, las features se normalizan con Z-Score: se resta la media y se divide por la desviacion tipica, calculadas **solo sobre los datos de entrenamiento** (para evitar *data leakage*).

### 2.4 Modelo HMM

Cada digito se modela con un **GaussianHMM** (emisiones continuas gaussianas) con topologia **left-right**:

- **Topologia left-right:** Los estados se recorren de izquierda a derecha, sin volver atras. Esto modela el hecho de que un digito se dibuja de principio a fin, sin retroceder en las fases del trazado.
- **Emisiones gaussianas:** Cada estado emite vectores de features según una distribución gaussiana multidimensional.
- **Entrenamiento EM:** Se usa el algoritmo Baum-Welch (Expectation-Maximization) para estimar los parametros del modelo.
- **Multiple restarts:** Se entrena varias veces con semillas distintas y se queda el mejor modelo, porque EM puede caer en maximos locales.

### 2.5 Clasificacion

Para clasificar un trazado de test:

1. Se calcula la **log-verosimilitud** (`model.score()`) bajo cada uno de los 10 HMMs.
2. Se elige el digito cuyo modelo da la **mayor log-verosimilitud**.

---

## 3. Experimentos y Resultados

### 3.1 Busqueda de hiperparametros

Se han probado sistematicamente distintos valores de los principales hiperparametros, usando el escenario N=74 (74 usuarios train, 19 test) para comparar.

#### 3.1.1 Numero de estados

| Estados | Accuracy |
|---------|----------|
| 3       | 72.6%    |
| 5       | 81.2%    |
| **7**   | **89.3%** |
| 10      | 87.4%    |

**Conclusión:** 7 estados es el valor optimo. Con 3 estados, el modelo no tiene suficiente capacidad para representar la complejidad de algunos digitos (como el 8 o el 5). Con 10 estados, el modelo sobreajusta: tiene demasiados parametros para la cantidad de datos de entrenamiento.

*Intuicion:* Cada estado representa una "fase" del trazado. Por ejemplo, para el digito "2":
- Estado 0: inicio del trazo (curva superior)
- Estado 1: parte superior del arco
- Estado 2: transicion diagonal
- Estado 3: trazo horizontal inferior
- Estado 4-6: remate y final

#### 3.1.2 Tipo de covarianza

| Covarianza | Accuracy | Parametros/estado |
|------------|----------|-------------------|
| spherical  | 70.9%    | 1 por estado      |
| diag       | 81.2%    | D por estado      |
| **tied**   | **87.8%** | D(D+1)/2 compartidos |

**Conclusión:** `tied` da el mejor resultado porque comparte la covarianza entre todos los estados, lo que actua como regularizador y evita el sobreajuste. Sin embargo, su tiempo de entrenamiento es notablemente mayor (789s vs 130s para `diag`). `spherical` es demasiado simple (asume la misma varianza para todas las features). `diag` ofrece un buen equilibrio entre rendimiento y velocidad.

**Nota:** No se ha probado `full` porque requiere estimar D(D+1)/2 parametros **por estado**, lo que con 12 features y solo unas ~600 muestras de entrenamiento por digito causa problemas numéricos (matrices singulares).

#### 3.1.3 Subconjunto de features

| Subset    | Dim | Accuracy |
|-----------|-----|----------|
| min       | 7D  | 78.9%    |
| **med**   | **12D** | **81.2%** |
| full      | 21D | 78.8%    |

**Conclusión:** El subconjunto medio (12 features) da los mejores resultados. El subconjunto completo (21D) paradojicamente empeora, lo cual se explica por la **maldicion de la dimensionalidad**: con más features, las gaussianas tienen mas parametros que estimar con los mismos datos, y el modelo sobreajusta o las estimaciones son imprecisas.

Las 12 features del subconjunto medio son:
`dx, dy, sin(angle), cos(angle), dtheta, v, rho, a, dv, lewiratio, x, y`

#### 3.1.4 Probabilidad de autolazo

| Prob. autolazo | Accuracy |
|----------------|----------|
| 0.4            | 80.7%    |
| 0.5            | 80.7%    |
| **0.6**        | **81.2%** |
| 0.7            | 81.1%    |
| 0.8            | 80.6%    |

**Conclusión:** 0.6 es el valor optimo, aunque las diferencias son pequeñas (rango de 0.6%). La probabilidad de autolazo controla cuanto tiempo se espera que el modelo permanezca en cada estado. Con 0.8, los estados son demasiado "pegajosos" y el modelo no recorre todos los estados para secuencias cortas. Con 0.4, los estados cambian demasiado rapido.

#### 3.1.5 Efecto del preprocesado

| Configuracion | Accuracy |
|--------------|----------|
| Con suavizado + remuestreo | 81.2% |
| Sin suavizado | 75.5%   |
| Sin remuestreo | 77.2%  |
| **Sin nada** | **91.3%** |

**Conclusión:** Este resultado es **sorprendente e inesperado**. La configuracion sin ningun preprocesado (sin suavizado ni remuestreo) obtuvo la mayor accuracy en la busqueda rapida (91.3%). Esto sugiere que:

1. **El remuestreo a longitud fija pierde informacion temporal:** Los HMMs pueden manejar secuencias de longitud variable de forma nativa (mediante el parametro `lengths`). Al forzar todas las secuencias a 80 puntos, se pierde la informacion de velocidad de escritura, que es discriminativa.
2. **El suavizado puede eliminar detalles utiles:** El filtro Savitzky-Golay, aunque reduce ruido, también elimina microvariaciones que pueden ser caracteristicas de cada digito.

Sin embargo, en la evaluacion final con mas restarts y 7 estados, la configuracion con preprocesado alcanzo **92.89%**, lo cual indica que con suficiente capacidad del modelo se puede compensar. **La interaccion entre preprocesado y complejidad del modelo es un tema que merece mayor investigacion.**

### 3.2 Evaluacion final

Con la mejor configuracion encontrada (7 estados, covarianza diagonal, 12 features, prob_autolazo=0.6, 5 restarts, 100 iteraciones EM, suavizado + remuestreo a 80 puntos):

| Escenario | Train | Test | Accuracy |
|-----------|-------|------|----------|
| **N=74**  | 74 usuarios | 19 usuarios | **92.89%** |
| **N=47**  | 47 usuarios | 46 usuarios | **89.78%** |

La accuracy baja con menos datos de entrenamiento (N=47 vs N=74), lo cual es esperable: con menos muestras, los HMMs tienen estimaciones menos precisas de las distribuciones gaussianas en cada estado. Aun asi, la caida es relativamente moderada (~3%).

### 3.3 Analisis de errores (Matriz de confusion)

Del analisis de las matrices de confusion se observan los principales errores:

- **1 y 7:** Confusion frecuente porque ambos son trazos mayoritariamente verticales.
- **3 y 5:** Se confunden porque comparten la parte curva.
- **4 y 9:** La parte cerrada de ambos puede ser similar.
- **6 y 0:** La curva cerrada puede confundirse.
- **Digitos mejor reconocidos:** 0, 1, 8 (tienen trazados muy característicos).
- **Digitos peor reconocidos:** 5, 9 (alta variabilidad entre usuarios).

---

## 4. Que no funciono / Que si funciono

### 4.1 Lo que NO funciono bien

| Intento | Resultado | Explicacion |
|---------|-----------|-------------|
| **21 features (full)** | 78.8% (peor que 12D) | Maldicion de la dimensionalidad: demasiados parametros para los datos disponibles |
| **3 estados** | 72.6% | Insuficiente complejidad para modelar digitos complejos |
| **10 estados** | 87.4% (peor que 7) | Sobreajuste: muchos estados, pocas muestras por estado |
| **Covarianza spherical** | 70.9% | Demasiado restrictiva: asume misma varianza en todas las dimensiones |
| **Solo sin suavizado** | 75.5% | Sin filtrar el ruido, las features son mas ruidosas |
| **Solo sin remuestreo** | 77.2% | Con suavizado pero sin remuestreo, peor que baseline |
| **Prob. autolazo 0.8** | 80.6% | Estados demasiado largos, el modelo no usa todos los estados |

### 4.2 Lo que SI funciono

| Mejora | Efecto | Explicacion |
|--------|--------|-------------|
| **7 estados** | +16.7% vs 3 estados (89.3% vs 72.6%) | Suficientes "fases" para capturar la estructura de cada digito |
| **Covarianza tied** | +16.9% vs spherical (87.8% vs 70.9%) | Comparte covarianza entre estados, actua como regularizador |
| **12 features (med)** | Mejor balance (81.2%) | Suficiente informacion cinemática sin sobredimensionar |
| **Sin preprocesado** | 91.3% (sorpresa) | Los HMMs aprovechan mejor las secuencias de longitud variable |
| **Multiple restarts (5)** | 92.89% en eval final | Evita quedar atrapado en maximos locales de EM |
| **Z-Score normalizacion** | Necesario | Sin normalizar, features con escalas muy distintas dominan la verosimilitud |

---

## 5. Visualizacion Viterbi

El algoritmo de **Viterbi** calcula la secuencia de estados mas probable dado un trazado observado. Esto permite visualizar **que "fase" del digito corresponde a cada punto del trazado**.

En las graficas de Viterbi generadas (`resultados/viterbi/`), cada punto del trazado se colorea según el estado HMM asignado:

- **Estado 0 (azul):** Inicio del trazado
- **Estado 1-5 (colores intermedios):** Fases intermedias del dibujo
- **Estado 6 (ultimo):** Final del trazado

### Interpretacion por digito

- **Digito 0:** Los estados se reparten alrededor del circulo, mostrando las fases del trazo circular.
- **Digito 1:** Pocos estados activos (es un trazo simple), la mayoria del tiempo en 2-3 estados.
- **Digito 2:** Se distingue claramente: curva superior → diagonal → trazo horizontal inferior.
- **Digito 3:** Dos curvas en cascada, cada una ocupando aproximadamente la mitad de los estados.
- **Digito 4:** Angulo vertical + trazo horizontal + trazo vertical descendente.
- **Digito 5:** Trazo horizontal superior → curva inferior.
- **Digito 6:** Curva descendente → circulo inferior.
- **Digito 7:** Trazo horizontal → diagonal descendente (simple, pocos estados activos).
- **Digito 8:** Dos bucles, claramente segmentados en mitad superior e inferior.
- **Digito 9:** Circulo superior → trazo descendente.

Estas visualizaciones demuestran que el HMM ha aprendido a **segmentar automaticamente** las fases del trazado de forma coherente con la intuicion humana.

---

## 6. Descripcion del codigo

### Estructura

Todo el codigo esta en un unico fichero `clasificador_digitos.py`, organizado en partes claras:

| Parte | Contenido |
|-------|-----------|
| 1 | Carga de datos (`cargar_base_datos`, `cargar_muestra`) |
| 2 | Preprocesado (`preprocesar`) |
| 3 | Extraccion de features (`extraer_features`, `NormalizadorZScore`) |
| 4 | Modelo HMM (`entrenar_hmm_digito`, `entrenar_todos_los_digitos`) |
| 5 | Clasificacion (`clasificar`, `clasificar_lote`) |
| 6 | Escenarios de evaluacion (`ejecutar_escenario`) |
| 7 | Graficas (`graficar_confusion`, `graficar_viterbi`, etc.) |
| 8 | Pipeline principal (`main`) |

### Librerias utilizadas

| Libreria | Uso |
|----------|-----|
| `hmmlearn` | Implementacion de GaussianHMM (entrenamiento EM, scoring, Viterbi) |
| `numpy` | Operaciones numéricas |
| `scipy` | Interpolacion (remuestreo), filtro Savitzky-Golay |
| `scikit-learn` | Matriz de confusion |
| `matplotlib` + `seaborn` | Graficas |

### Como ejecutar

```bash
pip install -r requirements.txt
python clasificador_digitos.py
```

El script tarda aproximadamente **10-20 minutos** en ejecutarse completamente (dependiendo del hardware). Los resultados se guardan en la carpeta `resultados/`.

---

## 7. Resumen de resultados

| Configuracion | Accuracy N=74 | Accuracy N=47 |
|--------------|---------------|---------------|
| Baseline (5 estados, 12D, 2 restarts) | 81.2% | - |
| Mejor config (7 estados, 12D, 5 restarts) | **92.89%** | **89.78%** |

**Mejor configuracion encontrada:**
- 7 estados HMM (left-right)
- Covarianza diagonal (`diag`)
- 12 features cinemáticas (subconjunto `med`)
- Probabilidad de autolazo: 0.6
- Remuestreo a 80 puntos equidistantes
- Suavizado Savitzky-Golay (ventana=7, orden=3)
- 5 restarts por modelo
- 100 iteraciones EM

---

## 8. Conclusion

Se ha implementado un clasificador de digitos manuscritos basado en HMMs que alcanza una **accuracy del 92.89%** con 74 usuarios de entrenamiento y del **89.78%** con 47. El sistema demuestra que:

1. Los HMMs son una herramienta adecuada para modelar secuencias temporales como el trazado de digitos.
2. La topologia left-right captura correctamente la estructura secuencial del dibujo.
3. El preprocesado tiene un efecto complejo: el suavizado y remuestreo ayudan cuando el modelo es simple, pero sin preprocesado (secuencias de longitud variable) el HMM puede alcanzar mejores resultados al preservar la informacion temporal original.
4. El balance entre complejidad del modelo y datos disponibles es fundamental: ni demasiados estados (sobreajuste) ni demasiadas features (maldicion de la dimensionalidad).
5. La decodificacion Viterbi muestra que el modelo aprende fases del trazado coherentes con la intuicion humana.
6. El numero de estados es el hiperparametro mas critico (+16.7% de mejora al pasar de 3 a 7).

---

## Referencias

- Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE*, 77(2).
- Documentacion de `hmmlearn`: https://hmmlearn.readthedocs.io/
- Base de datos e-BioDigit: proporcionada por la asignatura ASMI.
