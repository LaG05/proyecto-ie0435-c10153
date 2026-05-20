import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'C10153_Luis_Aguirre.joblib')

def cargar_modelo():
    return joblib.load(MODEL_PATH)

def predecir_contaminacion(imagen_vector):
    model_package = cargar_modelo()
    classifier = model_package['classifier']
    pca = model_package['pca']
    
    imagen_reducida = pca.transform([imagen_vector])
    prediccion = classifier.predict(imagen_reducida)[0]
    
    return {
        'contaminacion': 'SÍ (con arroz)' if prediccion == 1 else 'NO (sin arroz)',
        'clase': int(prediccion)
    }
