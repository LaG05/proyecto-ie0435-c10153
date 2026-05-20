
---

## 📄 2. DATASET.md (Documentación completa)

```markdown
# Documentación del Dataset

## Resumen

| Característica | Valor |
|----------------|-------|
| Total original | 60 imágenes |
| Únicas después limpieza | 27 |
| Duplicados eliminados | 33 |
| Clase positiva (con arroz) | 12 (44.4%) |
| Clase negativa (sin arroz) | 15 (55.6%) |

## Método de recolección
- Cada estudiante: 30 fotos (15 positivas + 15 negativas)
- Fondo: hoja blanca
- Variaciones: luz natural/artificial, ángulos 0-45°

## Preprocesamiento
1. Escala de grises
2. Redimensionamiento a 128×128
3. Binarización (1=blanco/fondo, 0=objeto)
4. Aplanamiento a vector de 16384 elementos

## Limitaciones documentadas
- Dataset pequeño (27 muestras)
- Solo fondo blanco (no realista)
- Posible sesgo por condiciones de luz
