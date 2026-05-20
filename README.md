# Proyecto 1 IE0435 - Clasificación de Contaminaciones

**Autor:** Luis Adrián Aguirre (C10153)  
**Curso:** Inteligencia Artificial Aplicada a la Ingeniería Eléctrica  
**Fecha:** Mayo 2026

## Resultados del mejor modelo

| Métrica | Valor |
|---------|-------|
| Accuracy | 83.33% |
| Precision | 85.71% |
| Recall | 85.71% |
| F1-Score | 85.71% |

## Reproducibilidad exacta

```bash
# 1. Clonar
git clone https://github.com/tuusuario/proyecto-ie0435-c10153.git
cd proyecto-ie0435-c10153

# 2. Crear entorno
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Entrenar (reproduce exactamente mis resultados)
python src/train.py --data data/dataset_grupo_completo.csv \
                    --test_size 0.2 \
                    --random_state 42 \
                    --pca_components 5

# 5. Predicción
python src/predict.py --image ruta/imagen.csv
