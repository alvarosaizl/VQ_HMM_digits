# Informe HMM: Clasificacion de Digitos Manuscritos con Hidden Markov Models

## Proyecto ASMI - e-BioDigit

---

## 1. Objetivo

Clasificar digitos manuscritos (0-9) a partir de trazados dinamicos capturados en una tableta digitalizadora, utilizando **Hidden Markov Models (HMM)** con topologia left-right. Se evaluan tres enfoques progresivos:

- **Entrega 1:** GaussianHMM con parametros uniformes y busqueda de hiperparametros.
- **Entrega 2A:** Optimizacion individual de hiperparametros por digito (resultado negativo).
- **Entrega 2B:** GMMHMM con mezclas de gaussianas y estrategia de escalado iterativo (resultado positivo).

---

## 2. Base de datos

- **e-BioDigit:** 93 usuarios, 10 digitos, 2 sesiones x 4 repeticiones = 80 muestras por usuario.
- **Total:** 7440 muestras.
- **Escenarios de evaluacion:**
  - **N=74:** 74 usuarios train / 19 usuarios test.
  - **N=47:** 47 usuarios train / 46 usuarios test.
  - **LOO CV:** Leave-One-User-Out (93 folds).

---

## 3. Preprocesado

Pipeline aplicado a cada trazado:

1. **Centrado:** restar media de x e y.
2. **Escalado:** normalizar por el rango maximo.
3. **Suavizado:** filtro Savitzky-Golay (ventana=7, orden=3).
4. **Remuestreo:** interpolacion a 80 puntos equiespaciados.
5. **Normalizacion Z-Score:** ajustada solo sobre datos de entrenamiento.

### Estudio de ablacion del preprocesado

| Configuracion | Accuracy | Diferencia |
|---------------|----------|------------|
| Con todo | 81.25% | referencia |
| Sin suavizado | 75.46% | -5.79pp |
| Sin remuestreo | 77.17% | -4.08pp |
| Sin nada (raw) | 91.32% | +10.07pp |

**Nota:** El resultado "sin nada" utiliza la configuracion final optimizada (7 estados, diag, med, p=0.6), mientras que la referencia de 81.25% es la accuracy de validacion durante la busqueda inicial con 2 restarts. La evaluacion final con 5 restarts y 100 iteraciones EM produce accuracies de 92.89% (N=74).

---

## 4. Extraccion de features

Se computan 21 features locales por punto del trazado. Se evaluaron tres subconjuntos:

| Subconjunto | Dimensiones | Features | Accuracy (val) |
|-------------|-------------|----------|----------------|
| `min` | 7D | dx, dy, sin, cos, v, dtheta, rho | 78.88% |
| **`med`** | **12D** | **dx, dy, sin, cos, dtheta, v, rho, a, dv, lewiratio, x, y** | **81.25%** |
| `full` | 21D | todas las features disponibles | 78.82% |

El subconjunto `med` (12D) ofrece el mejor equilibrio: mas informativo que `min`, sin el sobreajuste de `full` (maldicion de la dimensionalidad con 21D).

---

## 5. Entrega 1: GaussianHMM con parametros uniformes

### 5.1 Busqueda de hiperparametros

Busqueda secuencial variando un parametro a la vez, con 78 usuarios train / 15 val:

**Numero de estados:**

| N estados | Accuracy (val) |
|-----------|----------------|
| 3 | 72.57% |
| 5 | 81.25% |
| **7** | **89.34%** |
| 10 | 87.37% |

**Tipo de covarianza:**

| Covarianza | Accuracy (val) |
|------------|----------------|
| Spherical | 70.86% |
| **Diagonal** | **81.25%** |
| Tied | 87.83% |

Se selecciona `diag` por ofrecer mejor estabilidad numerica y menor coste computacional que `tied`.

**Probabilidad de autolazo:**

| Prob. autolazo | Accuracy (val) |
|----------------|----------------|
| 0.4 | 80.72% |
| 0.5 | 80.72% |
| **0.6** | **81.25%** |
| 0.7 | 81.05% |
| 0.8 | 80.59% |

### 5.2 Configuracion final E1

| Parametro | Valor |
|-----------|-------|
| Modelo | GaussianHMM |
| Topologia | Left-right |
| N estados | 7 |
| Covarianza | Diagonal |
| Prob. autolazo | 0.6 |
| Features | med (12D) |
| Entrenamiento | 100 iter EM, 5 restarts |
| Clasificacion | argmax log-verosimilitud |

### 5.3 Resultados E1

| Escenario | Accuracy |
|-----------|----------|
| N=74 | **92.89%** |
| N=47 | **89.78%** |
| LOO CV | **78.40% +/- 8.79%** |

La alta variabilidad del LOO CV (std=8.79%) refleja la diversidad de estilos de escritura entre usuarios.

---

## 6. Entrega 2A: Optimizacion por digito (resultado negativo)

### 6.1 Metodologia

Optimizar (n_estados, prob_autolazo) individualmente para cada digito:
- Split interno: 80% inner-train, 20% validacion.
- Grid: n_estados in {3, 5, 7, 10}, prob_autolazo in {0.4, 0.5, 0.6, 0.7, 0.8}.
- Criterio: log-verosimilitud por digito aislado.

### 6.2 Resultados

| Escenario | E1 (uniforme) | E2A (por digito) | Diferencia |
|-----------|---------------|-------------------|------------|
| N=74 | **92.89%** | 85.99% | **-6.90pp** |
| N=47 | **89.78%** | 85.82% | **-3.96pp** |

### 6.3 Analisis del fracaso

1. **Criterio incorrecto:** Maximizar la LL de cada digito por separado mejora el ajuste individual pero empeora la discriminacion entre digitos. Un HMM con 10 estados asigna LL altas a todo, no solo a su digito.

2. **Asimetria de complejidad:** Modelos con distinto numero de estados no compiten en igualdad de condiciones en la clasificacion por argmax.

3. **Validacion insuficiente:** Solo 9-15 usuarios para validacion -> seleccion ruidosa. Los parametros "optimos" difieren drasticamente entre escenarios (ej.: digito 5 pasa de 10 a 3 estados).

4. **Inconsistencia en normalizacion:** El normalizador Z-Score difiere entre la busqueda (80% datos) y la evaluacion final (100% datos).

**Leccion clave:** El criterio de optimizacion debe coincidir con el objetivo final. Para clasificacion, hay que optimizar accuracy de clasificacion conjunta, no LL individual.

---

## 7. Entrega 2B: GMMHMM con escalado iterativo (resultado positivo)

### 7.1 Motivacion

Reemplazar la unica gaussiana por estado del GaussianHMM con una **mezcla de gaussianas (GMM)**, permitiendo modelar distribuciones multimodales dentro de cada estado (ej.: variantes de trazado entre usuarios).

### 7.2 Estrategia de escalado iterativo

```
Fase 1: Prueba rapida (n_mix=1,2,3 con config E1)
    Si mejora >= 0.5pp -> evaluacion final
    Si NO mejora -> Fase 2

Fase 2: Grid search global (n_mix x n_estados x prob_autolazo)
    Si mejora >= 0.5pp -> evaluacion final
    Si NO mejora -> Fase 3

Fase 3: Seleccion de features (ultimo recurso)
```

### 7.3 Fase 1: Prueba rapida

| n_mix | Modelo | Accuracy (val) |
|-------|--------|----------------|
| **1** | **GaussianHMM** | **87.08%** |
| 2 | GMMHMM | 84.83% |
| 3 | GMMHMM | 75.92% |

GMMHMM con la config de E1 no mejora: los hiperparametros de E1 fueron optimizados para GaussianHMM. Se procede a Fase 2.

### 7.4 Fase 2: Grid search (105 configuraciones)

Espacio de busqueda:
- n_mix in {1, 2, 3}
- n_estados in {5, 7, 10, 12, 15}
- prob_autolazo in {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}

**Top 5 configuraciones:**

| Rank | n_mix | n_estados | prob_autolazo | Accuracy (val) |
|------|-------|-----------|---------------|----------------|
| **1** | **2** | **7** | **0.6** | **94.75%** |
| 2 | 2 | 5 | 0.3 | 93.00% |
| 3 | 1 | 15 | 0.3 | 91.25% |
| 4 | 2 | 5 | 0.7 | 90.33% |
| 5 | 2 | 5 | 0.9 | 90.33% |

### 7.5 Trade-off n_mix x n_estados

| n_estados | Mejor n_mix=1 | Mejor n_mix=2 | Mejor n_mix=3 |
|-----------|---------------|---------------|---------------|
| 5 | 86.42% | **93.00%** | 85.67% |
| 7 | 77.83% | **94.75%** | 76.08% |
| 10 | 90.17% | 76.83% | 58.17% |
| 12 | 76.08% | 69.58% | 49.67% |
| 15 | **91.25%** | 77.75% | 67.83% |

**Analisis:** Con 5-7 estados, n_mix=2 mejora significativamente: cada estado captura una fase amplia del trazado, y las 2 gaussianas modelan la variabilidad entre usuarios. Con 10+ estados, anadir mezclas causa sobreajuste severo (ratio datos/parametros < 1:1).

### 7.6 Configuracion final E2B

| Parametro | Valor |
|-----------|-------|
| Modelo | GMMHMM |
| n_mix | 2 |
| N estados | 7 (left-right) |
| Covarianza | Diagonal |
| Prob. autolazo | 0.6 |
| Features | med (12D) |
| Entrenamiento | 100 iter EM, 8 restarts |

### 7.7 Resultados E2B

| Escenario | Accuracy |
|-----------|----------|
| N=74 | **93.03%** |
| N=47 | **92.61%** |

---

## 8. Comparativa global HMM

| Escenario | E1 (GaussianHMM) | E2A (per-digito) | E2B (GMMHMM) |
|-----------|-------------------|-------------------|---------------|
| **N=74** | 92.89% | 85.99% | **93.03%** |
| **N=47** | 89.78% | 85.82% | **92.61%** |
| **LOO CV** | 78.40% +/- 8.79% | — | — |

### Mejoras de E2B respecto a E1

| Escenario | Mejora |
|-----------|--------|
| N=74 | +0.14pp |
| **N=47** | **+2.83pp** |

- **N=74:** Mejora modesta. Con 74 usuarios, el GaussianHMM ya estimaba bien una gaussiana por estado.
- **N=47:** Mejora significativa. Con menos datos de entrenamiento, las 2 gaussianas capturan variabilidad inter-usuario que una sola gaussiana no podia representar.
- **Robustez:** El GMMHMM pierde solo 0.42pp entre N=74 y N=47, frente a los 3.11pp del GaussianHMM.

---

## 9. Coste computacional

| Fase | Configuraciones | Tiempo |
|------|-----------------|--------|
| E1 busqueda hiperparametros | ~20 configs | ~2 horas |
| E1 evaluacion final | 2 escenarios + LOO | ~4 horas |
| E2A optimizacion per-digito | 200 modelos | ~44 min |
| E2B Fase 1 | 3 configs | ~2.9 horas |
| E2B Fase 2 | 105 configs | ~85 horas |
| E2B evaluacion final | 2 escenarios | ~4.8 horas |

---

## 10. Descripcion del codigo

| Fichero | Descripcion |
|---------|-------------|
| `Entrega1/clasificador_digitos.py` | Base completa: carga de datos, preprocesado, features, HMM, evaluacion |
| `Entrega1/ejecutar_loo.py` | LOO Cross-Validation con checkpoint |
| `Entrega2/clasificador_digitos_v2.py` | E2A: optimizacion per-digito |
| `Entrega2/clasificador_digitos_v3.py` | E2B: GMMHMM con escalado iterativo |

### Resultados generados

| Fichero | Contenido |
|---------|-----------|
| `Entrega1/resultados/busqueda_hiperparametros.json` | Resultados de la busqueda de E1 |
| `Entrega1/resultados/confusion_LOO.png` | Confusion LOO CV |
| `Entrega2/resultados/resultados.json` | E2A: params por digito |
| `Entrega2/resultados/resultados_v3.json` | E2B: todas las fases y config final |
| `Entrega2/resultados/confusion_N74_opt.png` | Confusion E2A N=74 |
| `Entrega2/resultados/confusion_N74_v3.png` | Confusion E2B N=74 |
| `Entrega2/resultados/confusion_N47_v3.png` | Confusion E2B N=47 |
| `Entrega2/resultados/top_configs_fase2.png` | Top 20 configs Fase 2 |

---

## 11. Conclusiones

1. **GaussianHMM con 7 estados y covarianza diagonal** es un baseline robusto (92.89% N=74) que sirve como regularizador implicito al mantener complejidad uniforme entre clases.

2. **Optimizar hiperparametros por digito empeora la clasificacion** (-6.90pp) porque maximizar la LL individual no equivale a maximizar la accuracy conjunta. La asimetria de complejidad entre modelos rompe la equidad del argmax.

3. **GMMHMM con n_mix=2 es la mejor configuracion** (93.03% N=74, 92.61% N=47), con la misma topologia que E1 pero mayor expresividad en el modelo de emision. La mejora es especialmente notable con menos datos de entrenamiento (+2.83pp en N=47).

4. **El trade-off n_mix x n_estados es critico:** se puede tener muchos estados con pocas mezclas, o pocos estados con mas mezclas, pero no ambos. La combinacion ganadora (n_mix=2, 7 estados) se situa en el optimo de este trade-off.

5. **El criterio de optimizacion debe coincidir con el objetivo:** la Fase 2 de E2B usa accuracy de clasificacion conjunta como criterio (no LL individual), lo que explica su exito frente al fracaso de E2A.

---

## Referencias

- Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE*, 77(2).
- Documentacion de `hmmlearn`: https://hmmlearn.readthedocs.io/
- Base de datos e-BioDigit: proporcionada por la asignatura ASMI.
