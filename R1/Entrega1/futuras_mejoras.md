# Futuras Mejoras e Implementaciones

## Proyecto: Clasificador de Digitos Manuscritos mediante HMMs

---

## 1. Mejoras en el modelo HMM

### 1.1 Numero de estados variable por digito

Actualmente se usa el mismo numero de estados (7) para todos los digitos. Sin embargo, algunos digitos son mas simples que otros (el "1" es un trazo casi recto, mientras que el "8" tiene dos bucles). Se podria usar el **criterio BIC** (Bayesian Information Criterion) para determinar el numero optimo de estados para cada digito individualmente.

```
BIC = -2 * log_likelihood + k * log(N)
```

donde `k` es el numero de parametros y `N` el numero de observaciones. hmmlearn ya incluye `model.bic()`.

**Dificultad:** Baja - Solo hay que añadir un bucle que entrene cada digito con distintos N y seleccione el mejor por BIC.

### 1.2 GMMHMM: Mezclas de gaussianas por estado

En lugar de una sola gaussiana por estado (GaussianHMM), se podria usar una **mezcla de gaussianas** (GMM) en cada estado. Esto permitiria modelar distribuciones multimodales dentro de un mismo estado.

hmmlearn tiene `GMMHMM` con parametro `n_mix` (numero de componentes de la mezcla).

**Dificultad:** Media - Mas parametros a estimar, posibles problemas de convergencia con pocos datos.

### 1.3 Topologia Bakis (saltos de 2)

La topologia left-right actual solo permite transiciones al mismo estado o al siguiente. Una extension es la topologia **Bakis**, que permite saltar 2 estados hacia delante:

```
A[i, i]   = p_self
A[i, i+1] = p_next
A[i, i+2] = p_skip     (salto de 2)
```

Esto podria ayudar a modelar mejor a usuarios que dibujan muy rapido y "se saltan" fases.

**Dificultad:** Baja - Solo modificar la funcion `_transmat_left_right()`.

---

## 2. Mejoras en las features

### 2.1 Seleccion automatica de features

En lugar de probar subconjuntos fijos (7D, 12D, 21D), se podria usar un metodo automatico:

- **Analisis de varianza entre clases:** Seleccionar las features cuyas medias por estado difieren mas entre digitos.
- **Forward selection:** Empezar con 1 feature e ir añadiendo la que mas mejore la accuracy.
- **PCA:** Reducir la dimensionalidad manteniendo el 95% de la varianza.

**Dificultad:** Media.

### 2.2 Features globales como complemento

El extractor global (`base_extractor.py`) produce 114 features por muestra completa (duracion total, velocidad media, ratios geometricos...). Estas no se pueden usar directamente con HMMs (que necesitan secuencias), pero se podrian usar como **features adicionales en un clasificador de segundo nivel**:

1. El HMM da las log-verosimilitudes para cada digito.
2. Se combinan con las 114 features globales.
3. Un clasificador sencillo (k-NN, SVM lineal) toma la decision final.

**Dificultad:** Media - Requiere implementar un sistema de fusion.

### 2.3 Delta y delta-delta features

En reconocimiento de voz es comun añadir las derivadas temporales de las features (delta) y las derivadas de las derivadas (delta-delta). Aunque nuestro extractor ya incluye algunas derivadas (dv, dtheta...), se podrian calcular de forma sistematica sobre todas las features base.

**Dificultad:** Baja.

---

## 3. Mejoras en la evaluacion

### 3.1 Leave-One-Out Cross-Validation (LOOCV)

Implementar LOOCV completo: para cada uno de los 93 usuarios, entrenar con los 92 restantes y testear con ese usuario. Esto da una estimacion mas robusta del rendimiento, aunque tarda mucho mas (~93 veces el entrenamiento normal).

El codigo base ya esta preparado para esto (la funcion `ejecutar_escenario` acepta listas de usuarios arbitrarias).

**Dificultad:** Baja (solo bucle), pero alto coste computacional (~2-3 horas).

### 3.2 EER y curvas DET

Ademas de la accuracy, calcular el **Equal Error Rate (EER)** y las **curvas DET (Detection Error Tradeoff)** por digito. Estas metricas son estandar en biometria y dan una vision mas completa del rendimiento.

Para el EER: separar las puntuaciones en "genuinas" (el HMM correcto) e "impostoras" (los otros 9 HMMs) y encontrar el umbral donde FAR = FRR.

**Dificultad:** Media.

### 3.3 Analisis de errores por usuario

Analizar que usuarios tienen peor accuracy y por que. Posibles causas:

- Usuarios que dibujan de forma muy atipica.
- Usuarios zurdos vs diestros.
- Usuarios con trazos muy cortos o muy largos.

Esto ayudaria a entender las limitaciones del sistema.

**Dificultad:** Baja.

---

## 4. Mejoras en el preprocesado

### 4.1 Normalizacion temporal adaptativa

En lugar de remuestrear siempre a 80 puntos, adaptar el numero de puntos al percentil de longitud de la base de datos, o usar un numero proporcional a la complejidad del digito.

**Dificultad:** Baja.

### 4.2 Eliminacion de outliers

Detectar y eliminar muestras anomalas antes de entrenar (por ejemplo, muestras con muy pocos puntos, o con coordenadas fuera de rango). Esto podria mejorar la calidad de los modelos.

**Dificultad:** Baja.

### 4.3 Data augmentation

Generar mas muestras de entrenamiento aplicando transformaciones:

- Rotaciones pequeñas (+-5 grados)
- Escalado aleatorio (+-10%)
- Añadir ruido gaussiano a las coordenadas
- Variacion de velocidad (remuestrear con distinta densidad)

Esto podria mejorar la generalizacion, especialmente para digitos con pocas muestras.

**Dificultad:** Media.

---

## 5. Enfoques alternativos a explorar

### 5.1 HMM Discreto con Vector Quantization (VQ)

En lugar de usar emisiones gaussianas continuas, cuantificar los vectores de features con **K-Means** y usar los indices de cluster como observaciones discretas para un **MultinomialHMM**:

1. Entrenar K-Means sobre todas las features de entrenamiento (K = 32, 64, 128).
2. Asignar cada frame al cluster mas cercano → secuencia de simbolos discretos.
3. Entrenar un MultinomialHMM por digito sobre estas secuencias discretas.

Esto conecta con la tecnologia VQ que también se estudia en la asignatura y permite comparar HMM continuo vs discreto.

**Dificultad:** Media.

### 5.2 Clasificador hibrido HMM + k-NN

Usar las log-verosimilitudes de los 10 HMMs como un vector de features de 10 dimensiones, y alimentar un clasificador k-NN o SVM para la decision final. Esto podria capturar relaciones entre las puntuaciones que el argmax simple no aprovecha.

**Dificultad:** Baja.

### 5.3 Modelo HMM ergodico (no left-right)

Probar un HMM sin restricciones en la topologia (ergodico, donde cualquier transicion es posible) y comparar con el left-right. Aunque la topologia left-right tiene mas sentido fisico para trazados, el ergodico podria funcionar mejor para digitos que se dibujan con "retrocesos" (como algunas variantes del 4 o el 5).

**Dificultad:** Baja - Solo cambiar la inicializacion de la matriz de transicion.

---

## 6. Escalabilidad y despliegue

### 6.1 Interfaz grafica

Crear una interfaz simple (con `tkinter` o `pygame`) donde el usuario pueda dibujar un digito con el raton y ver la prediccion en tiempo real, junto con la visualizacion Viterbi.

**Dificultad:** Media.

### 6.2 Optimizacion de velocidad

El entrenamiento actual es secuencial (un digito tras otro). Se podria paralelizar con `multiprocessing` ya que los 10 modelos son independientes.

**Dificultad:** Baja.

### 6.3 Exportar modelos a formato ligero

En lugar de usar pickle (que depende de la version de Python), guardar los parametros de los modelos (medias, covarianzas, transiciones) en formato JSON o HDF5 para mayor portabilidad.

**Dificultad:** Baja.

---

## 7. Tabla resumen de prioridades

| Mejora | Impacto esperado | Dificultad | Prioridad |
|--------|-------------------|------------|-----------|
| N estados variable por digito (BIC) | Medio | Baja | Alta |
| GMMHMM (mezclas de gaussianas) | Alto | Media | Alta |
| LOOCV | Mejor evaluacion | Baja | Alta |
| VQ + HMM discreto | Comparativa interesante | Media | Alta |
| Seleccion automatica de features | Medio | Media | Media |
| Data augmentation | Medio | Media | Media |
| Topologia Bakis | Bajo | Baja | Media |
| EER y curvas DET | Mejor evaluacion | Media | Media |
| Interfaz grafica | Demo visual | Media | Baja |
| Clasificador hibrido | Bajo-Medio | Baja | Baja |

---

## 8. Notas finales

Estas mejoras estan ordenadas aproximadamente por su potencial impacto y viabilidad. Para un proyecto de 1-2 semanas adicionales, las mejoras de **prioridad alta** son las mas recomendables:

1. **BIC por digito** y **LOOCV** son rapidas de implementar y mejoran significativamente el analisis.
2. **GMMHMM** es la mejora mas prometedora en terminos de accuracy.
3. **VQ + HMM discreto** es especialmente relevante para la asignatura porque permite comparar dos paradigmas distintos.
