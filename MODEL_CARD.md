# Model Card - Clasificador de Contaminaciones v1.0

**Autor:** Luis Aguirre (C10153) | **Fecha:** Mayo 2026

## Intended use
Detección de granos de arroz en líneas de producción simuladas (fondo blanco).

**Out-of-scope:** Fondo texturizado, otros tipos de contaminación, video en tiempo real.

## Data summary
- **Recolección:** 5 estudiantes, esperado 150 muestras → obtenido 27 únicas
- **Causa de pérdida:** 33 duplicados (mismos estudiantes fotografiaron lo mismo)
- **PCA aplicado:** 16384 → 5 componentes (53.88% varianza explicada)

## Metrics (resultados reales)

| Modelo | Accuracy | F1 | Tiempo |
|--------|----------|-----|--------|
| **Decision Tree** | **0.8333** | **0.8571** | 0.5s |
| Naive Bayes | 0.8333 | 0.8571 | 0.1s |
| SVM | 0.8333 | 0.8571 | 0.6s |
| KNN | 0.6667 | 0.7500 | 0.4s |

**Matriz de confusión (Decision Tree):**
