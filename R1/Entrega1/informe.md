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

| Configuracion | Suavizado | Remuestreo | Centrado | Escalado | Accuracy |
|--------------|-----------|------------|----------|----------|----------|
| **con_todo** | si | 80 | si | si | **81.2%** |
| sin_suavizado | no | 80 | si | si | 75.5% |
| sin_resample | si | no | si | si | 77.2% |
| sin_centrado | si | 80 | no | si | 77.1% |
| sin_escalado | si | 80 | si | no | 53.9% |
| sin_cent_esc | si | 80 | no | no | 40.5% |
| sin_nada | no | no | no | no | 41.9% |

**Conclusiones:**

1. **El escalado es el paso mas critico del preprocesado.** Sin escalado, la accuracy cae drasticamente de 81.2% a 53.9%. Esto se debe a que usuarios distintos dibujan con tamaños muy diferentes; sin normalizar la escala, las distribuciones gaussianas de los HMMs no pueden generalizar entre usuarios.

2. **El centrado tiene un efecto moderado.** Sin centrado (pero con escalado) la accuracy baja de 81.2% a 77.1%. La posicion absoluta en la pantalla no deberia ser relevante, pero el centrado ayuda a que las features de posicion (x, y) sean comparables.

3. **Sin centrado ni escalado el sistema se degrada gravemente** (40.5%), confirmando que la normalizacion espacial es fundamental para la generalizacion entre usuarios.

4. **El suavizado aporta ~5.7% de mejora** (81.2% vs 75.5%), ya que reduce ruido del sensor tactil.

5. **El remuestreo aporta ~4% de mejora** (81.2% vs 77.2%), al hacer las secuencias comparables en longitud.

### 3.2 Evaluacion final

Con la mejor configuracion encontrada (7 estados, covarianza diagonal, 12 features, prob_autolazo=0.6, 5 restarts, 100 iteraciones EM, suavizado + remuestreo a 80 puntos):

| Escenario | Train | Test | Accuracy |
|-----------|-------|------|----------|
| **N=74**  | 74 usuarios | 19 usuarios | **92.89%** |
| **N=47**  | 47 usuarios | 46 usuarios | **89.78%** |
| **LOO CV** | 92 usuarios | 1 usuario (x93 folds) | **78.40% +/- 8.79%** |

**Analisis de los 3 escenarios:**

- **N=74 (92.89%):** Con 74 usuarios de entrenamiento y solo 19 de test, el modelo tiene abundantes datos para estimar bien los parametros. La accuracy es alta, aunque el conjunto de test es pequeño y el resultado puede tener varianza.

- **N=47 (89.78%):** Con menos datos de entrenamiento la accuracy baja ~3%, lo cual es esperable: los HMMs tienen estimaciones menos precisas de las distribuciones gaussianas en cada estado.

- **LOO CV (78.40% +/- 8.79%):** La validacion cruzada Leave-One-User-Out es la evaluacion mas robusta y realista. Cada fold entrena con 92 usuarios y evalua sobre 1, repitiendo para los 93 usuarios. La accuracy media es menor que en los otros escenarios porque: (1) se evalua sobre **todos** los usuarios, incluyendo los mas dificiles, (2) cada usuario tiene ~80 muestras de test (todos sus digitos y sesiones), por lo que un unico usuario difícil puede bajar mucho su accuracy individual. La desviacion tipica de 8.79% refleja la variabilidad entre usuarios: algunos alcanzan >95% y otros bajan a ~50%, lo cual es esperable dada la diversidad de estilos de escritura. Ademas, se uso una configuracion mas ligera (1 restart, 30 iter EM) por coste computacional, lo cual penaliza ligeramente el resultado.

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
| **Escalado** | Critico (~27% de mejora) | Sin escalado la accuracy cae a 53.9%; normalizar el tamaño es esencial |
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

| Configuracion | Accuracy N=74 | Accuracy N=47 | Accuracy LOO CV |
|--------------|---------------|---------------|-----------------|
| Baseline (5 estados, 12D, 2 restarts) | 81.2% | - | - |
| Mejor config (7 estados, 12D, 5 restarts) | **92.89%** | **89.78%** | **78.40% +/- 8.79%** |

**Mejor configuracion encontrada:**
- 7 estados HMM (left-right)
- Covarianza diagonal (`diag`)
- 12 features cinemáticas (subconjunto `med`)
- Probabilidad de autolazo: 0.6
- Remuestreo a 80 puntos equidistantes
- Suavizado Savitzky-Golay (ventana=7, orden=3)
- Centrado y escalado activados
- 5 restarts por modelo (2 para LOO por coste computacional)
- 100 iteraciones EM (30 para LOO)

---

## 8. Conclusion

Se ha implementado un clasificador de digitos manuscritos basado en HMMs que alcanza una **accuracy del 92.89%** con 74 usuarios de entrenamiento, del **89.78%** con 47, y del **78.40%** en validacion cruzada Leave-One-User-Out sobre los 93 usuarios. El sistema demuestra que:

1. Los HMMs son una herramienta adecuada para modelar secuencias temporales como el trazado de digitos.
2. La topologia left-right captura correctamente la estructura secuencial del dibujo.
3. El preprocesado es fundamental: el **escalado** es el paso mas critico (sin el, la accuracy cae a 53.9%), seguido del centrado, suavizado y remuestreo. Sin ningun preprocesado la accuracy se desploma a 40.5%.
4. El balance entre complejidad del modelo y datos disponibles es fundamental: ni demasiados estados (sobreajuste) ni demasiadas features (maldicion de la dimensionalidad).
5. La decodificacion Viterbi muestra que el modelo aprende fases del trazado coherentes con la intuicion humana.
6. El numero de estados es el hiperparametro mas critico (+16.7% de mejora al pasar de 3 a 7).
7. La evaluacion LOO CV es la mas realista y muestra una alta variabilidad entre usuarios (std=8.79%), reflejando la diversidad de estilos de escritura en la base de datos.

---

## Referencias

- Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE*, 77(2).
- Documentacion de `hmmlearn`: https://hmmlearn.readthedocs.io/
- Base de datos e-BioDigit: proporcionada por la asignatura ASMI.
