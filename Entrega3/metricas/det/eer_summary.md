# Resumen EER global (pooled one-vs-rest)

Para cada modelo, EER es el punto de la curva DET donde FPR = FNR (calculada sobre los scores agregados de las 10 clases en formato uno-contra-resto).

| Modelo | N=74 EER | N=47 EER | LOO EER |
|--------|----------|----------|---------|
| E1 HMM (GaussianHMM) |   5.66% |   6.47% |  20.59% |
| E2B HMM (GMMHMM) |  12.88% |  20.98% |  31.99% |
| VQ optimizado |   1.84% |   1.93% |   1.91% |
| Paralel agreement |   7.52% |   7.16% |  14.88% |
| Paralel soft |   3.36% |   4.51% |   3.98% |
| Paralel margin_weighted |   1.84% |   1.97% |   2.68% |
| Serial cascade (HMM->VQ) |  20.72% |  49.24% |  12.34% |
