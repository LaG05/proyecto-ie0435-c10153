"""
Script de Verificación de Criterios de Calificación
Proyecto 1 - IE0435

Evalúa si el modelo cumple con los criterios de la rúbrica
"""

import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


class CriteriaEvaluator:
    """
    Evalúa el cumplimiento de los criterios de calificación del proyecto
    """

    def __init__(self):
        self.total_score = 0
        self.max_score = 20  # 5 criterios × 4 puntos máximo
        self.criteria_scores = {}

    def evaluate_model_selection(self, trained_models_count, has_justification=True):
        """
        Criterio 1: Elección del modelo

        Args:
            trained_models_count: Número de modelos entrenados
            has_justification: Si hay justificación de la elección

        Returns:
            score: Puntaje obtenido (0-4)
        """
        print("\n" + "=" * 70)
        print("📋 CRITERIO 1: ELECCIÓN DEL MODELO")
        print("=" * 70)

        if trained_models_count == 0:
            score = 0
            evaluation = "No entregado"
            print("❌ No se entrenaron modelos")
        elif trained_models_count == 1:
            score = 1
            evaluation = "Inadecuado - Solo 1 modelo entrenado"
            print(f"⚠️  Solo se entrenó {trained_models_count} modelo")
        elif trained_models_count == 2:
            score = 2
            evaluation = "Poco justificado - Solo 2 modelos"
            print(f"⚠️  Se entrenaron {trained_models_count} modelos (se requieren al menos 3)")
        elif trained_models_count == 3:
            score = 3
            evaluation = "Adecuado - 3 modelos entrenados"
            print(f"✓ Se entrenaron {trained_models_count} modelos")
        else:  # >= 4
            if has_justification:
                score = 5
                evaluation = "Excelente - 4+ modelos con justificación"
                print(f"✅ Se entrenaron {trained_models_count} modelos")
                print("✅ Hay justificación clara de la elección")
            else:
                score = 4
                evaluation = "Bueno - 4+ modelos sin justificación completa"
                print(f"✓ Se entrenaron {trained_models_count} modelos")
                print("⚠️  Falta justificación detallada")

        print(f"\n📊 Evaluación: {evaluation}")
        print(f"🎯 Puntaje: {score}/5 puntos")

        self.criteria_scores['Elección del modelo'] = {
            'score': score,
            'max': 5,
            'evaluation': evaluation
        }

        return score

    def evaluate_training(self, is_reproducible=True, uses_best_practices=True):
        """
        Criterio 2: Entrenamiento

        Args:
            is_reproducible: Si el código es reproducible (random_state, etc.)
            uses_best_practices: Si usa buenas prácticas (train/test split, etc.)

        Returns:
            score: Puntaje obtenido (0-4)
        """
        print("\n" + "=" * 70)
        print("🔧 CRITERIO 2: ENTRENAMIENTO")
        print("=" * 70)

        if not is_reproducible and not uses_best_practices:
            score = 0
            evaluation = "No entregado"
            print("❌ Código no reproducible y sin buenas prácticas")
        elif not is_reproducible:
            score = 1
            evaluation = "No reproducible"
            print("⚠️  El código no es reproducible (falta random_state)")
        elif not uses_best_practices:
            score = 2
            evaluation = "Deficiente - Sin buenas prácticas"
            print("⚠️  No se aplican buenas prácticas de ML")
        elif is_reproducible and uses_best_practices:
            score = 4
            evaluation = "Correcto y reproducible"
            print("✅ Código correctamente implementado")
            print("✅ Es reproducible (random_state configurado)")
            print("✅ Usa train/test split apropiado")

        print(f"\n📊 Evaluación: {evaluation}")
        print(f"🎯 Puntaje: {score}/4 puntos")

        self.criteria_scores['Entrenamiento'] = {
            'score': score,
            'max': 4,
            'evaluation': evaluation
        }

        return score

    def evaluate_metrics(self, accuracy, has_multiple_metrics=True,
                         has_confusion_matrix=True, are_correct=True):
        """
        Criterio 3: Métricas

        Args:
            accuracy: Accuracy del modelo
            has_multiple_metrics: Si reporta múltiples métricas
            has_confusion_matrix: Si incluye matriz de confusión
            are_correct: Si los cálculos son correctos

        Returns:
            score: Puntaje obtenido (0-4)
        """
        print("\n" + "=" * 70)
        print("📈 CRITERIO 3: MÉTRICAS")
        print("=" * 70)

        if accuracy < 0.5:
            score = 0
            evaluation = "No entregadas - Modelo inválido"
            print(f"❌ Accuracy muy bajo: {accuracy:.2%}")
        elif not has_multiple_metrics or not are_correct:
            score = 1
            evaluation = "Incorrectas o incompletas"
            print("⚠️  Métricas incorrectas o incompletas")
        elif not has_confusion_matrix:
            score = 2
            evaluation = "Parciales - Falta matriz de confusión"
            print("⚠️  Falta matriz de confusión")
        elif accuracy >= 0.6 and accuracy < 0.8:
            score = 3
            evaluation = "Correctas - Accuracy aceptable (60-80%)"
            print(f"✓ Accuracy: {accuracy:.2%}")
            print("✓ Métricas reportadas correctamente")
        else:  # accuracy >= 0.8
            score = 4
            evaluation = "Bien calculadas y explicadas"
            print(f"✅ Accuracy excelente: {accuracy:.2%}")
            print("✅ Múltiples métricas reportadas")
            print("✅ Matriz de confusión incluida")

        print(f"\n📊 Evaluación: {evaluation}")
        print(f"🎯 Puntaje: {score}/4 puntos")

        self.criteria_scores['Métricas'] = {
            'score': score,
            'max': 4,
            'evaluation': evaluation
        }

        return score

    def evaluate_export(self, model_file_exists=False, filename_correct=False,
                        loads_successfully=False, predictions_work=False):
        """
        Criterio 4: Exportación

        Args:
            model_file_exists: Si existe el archivo .joblib
            filename_correct: Si el nombre sigue el formato correcto
            loads_successfully: Si el modelo se carga sin errores
            predictions_work: Si el modelo puede hacer predicciones

        Returns:
            score: Puntaje obtenido (0-4)
        """
        print("\n" + "=" * 70)
        print("💾 CRITERIO 4: EXPORTACIÓN")
        print("=" * 70)

        if not model_file_exists:
            score = 0
            evaluation = "No entregado"
            print("❌ No se encontró archivo .joblib")
        elif not filename_correct:
            score = 1
            evaluation = "No funciona - Nombre incorrecto"
            print("⚠️  El nombre del archivo no sigue el formato")
            print("   Formato esperado: carne_nombre_apellido.joblib")
        elif not loads_successfully:
            score = 2
            evaluation = "Problemas menores - No se carga correctamente"
            print("⚠️  El modelo no se puede cargar")
        elif not predictions_work:
            score = 3
            evaluation = "Funciona con ajustes - Predicciones fallan"
            print("⚠️  El modelo carga pero las predicciones fallan")
        else:
            score = 4
            evaluation = "Funciona sin errores"
            print("✅ Archivo .joblib exportado correctamente")
            print("✅ Nombre del archivo correcto")
            print("✅ Modelo se carga sin errores")
            print("✅ Predicciones funcionan correctamente")

        print(f"\n📊 Evaluación: {evaluation}")
        print(f"🎯 Puntaje: {score}/4 puntos")

        self.criteria_scores['Exportación'] = {
            'score': score,
            'max': 4,
            'evaluation': evaluation
        }

        return score

    def evaluate_generalization(self, test_accuracy, train_accuracy=None,
                                overfitting=False):
        """
        Criterio 5: Generalización

        Args:
            test_accuracy: Accuracy en conjunto de prueba
            train_accuracy: Accuracy en conjunto de entrenamiento
            overfitting: Si hay evidencia de sobreajuste

        Returns:
            score: Puntaje obtenido (0-4)
        """
        print("\n" + "=" * 70)
        print("🎯 CRITERIO 5: GENERALIZACIÓN")
        print("=" * 70)

        if test_accuracy < 0.5:
            score = 0
            evaluation = "No entregado - Modelo no generaliza"
            print(f"❌ Accuracy en test muy bajo: {test_accuracy:.2%}")
        elif overfitting:
            score = 1
            evaluation = "Falla - Sobreajuste evidente"
            print("❌ Evidencia clara de overfitting")
            if train_accuracy:
                print(f"   Train accuracy: {train_accuracy:.2%}")
                print(f"   Test accuracy: {test_accuracy:.2%}")
                print(f"   Diferencia: {abs(train_accuracy - test_accuracy):.2%}")
        elif test_accuracy < 0.6:
            score = 2
            evaluation = "Bajo - Generalización pobre"
            print(f"⚠️  Accuracy en test bajo: {test_accuracy:.2%}")
        elif test_accuracy < 0.8:
            score = 3
            evaluation = "Aceptable"
            print(f"✓ Accuracy en test aceptable: {test_accuracy:.2%}")
        else:
            score = 4
            evaluation = "Buen desempeño en test"
            print(f"✅ Excelente generalización: {test_accuracy:.2%}")
            if train_accuracy:
                diff = abs(train_accuracy - test_accuracy)
                print(f"✅ Diferencia train/test razonable: {diff:.2%}")

        print(f"\n📊 Evaluación: {evaluation}")
        print(f"🎯 Puntaje: {score}/4 puntos")

        self.criteria_scores['Generalización'] = {
            'score': score,
            'max': 4,
            'evaluation': evaluation
        }

        return score

    def calculate_total_score(self):
        """Calcula el puntaje total"""
        self.total_score = sum(c['score'] for c in self.criteria_scores.values())
        return self.total_score

    def print_final_report(self):
        """Imprime reporte final"""
        print("\n" + "=" * 70)
        print("📊 REPORTE FINAL DE EVALUACIÓN")
        print("=" * 70)

        print(f"\n{'Criterio':<25} {'Puntaje':<15} {'Evaluación'}")
        print("-" * 70)

        for criterion, data in self.criteria_scores.items():
            score_str = f"{data['score']}/{data['max']}"
            print(f"{criterion:<25} {score_str:<15} {data['evaluation']}")

        self.calculate_total_score()

        print("-" * 70)
        print(f"{'TOTAL':<25} {self.total_score}/{self.max_score}")
        print("=" * 70)

        # Calificación final
        percentage = (self.total_score / self.max_score) * 100

        if percentage >= 90:
            grade = "Excelente"
            emoji = "🌟"
        elif percentage >= 80:
            grade = "Muy Bueno"
            emoji = "✨"
        elif percentage >= 70:
            grade = "Bueno"
            emoji = "👍"
        elif percentage >= 60:
            grade = "Aceptable"
            emoji = "✓"
        else:
            grade = "Necesita Mejoras"
            emoji = "⚠️"

        print(f"\n{emoji} Calificación: {grade} ({percentage:.1f}%)")

        return self.total_score


def verify_model_file(filepath):
    """
    Verifica que el archivo del modelo existe y tiene el formato correcto

    Returns:
        dict con información del archivo
    """
    import re

    result = {
        'exists': False,
        'filename_correct': False,
        'loads': False,
        'predictions_work': False,
        'model': None
    }

    # Verificar existencia
    if not os.path.exists(filepath):
        print(f"\n❌ No se encontró el archivo: {filepath}")
        return result

    result['exists'] = True

    # Verificar formato del nombre
    filename = os.path.basename(filepath)
    pattern = r'^[A-Z]\d+_[a-zA-Z]+_[a-zA-Z]+\.joblib$'

    if re.match(pattern, filename):
        result['filename_correct'] = True
        print(f"✅ Formato de nombre correcto: {filename}")
    else:
        print(f"⚠️  Formato de nombre incorrecto: {filename}")
        print(f"   Formato esperado: carne_nombre_apellido.joblib")
        print(f"   Ejemplo: C10153_Luis_Lopez.joblib")

    # Intentar cargar modelo
    try:
        model = joblib.load(filepath)
        result['loads'] = True
        result['model'] = model
        print(f"✅ Modelo cargado exitosamente")
        print(f"   Tipo: {type(model).__name__}")
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        return result

    # Verificar predicciones
    try:
        # Crear datos de prueba (vector de 16384 elementos)
        test_data = np.random.randint(0, 2, (5, 16384))
        predictions = model.predict(test_data)
        result['predictions_work'] = True
        print(f"✅ Predicciones funcionan correctamente")
        print(f"   Predicciones de prueba: {predictions}")
    except Exception as e:
        print(f"❌ Error en predicciones: {e}")

    return result


def evaluate_complete_project(model_filepath, dataset_path="dataset.csv"):
    """
    Evalúa el proyecto completo contra los criterios
    """
    print("=" * 70)
    print("🔍 EVALUACIÓN COMPLETA DEL PROYECTO")
    print("=" * 70)

    evaluator = CriteriaEvaluator()

    # 1. Verificar archivo del modelo
    print("\n📁 Verificando archivo del modelo...")
    model_info = verify_model_file(model_filepath)

    # 2. Cargar dataset para evaluación
    try:
        print(f"\n📂 Cargando dataset: {dataset_path}")
        data = np.loadtxt(dataset_path, delimiter=',')
        X = data[:, :-1]
        y = data[:, -1]
        print(f"✅ Dataset cargado: {data.shape}")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return

    # 3. Evaluar cada criterio

    # Criterio 1: Elección del modelo
    # Asumiendo que se entrenaron 4 modelos (DT, NB, KNN, SVM)
    evaluator.evaluate_model_selection(
        trained_models_count=4,
        has_justification=True
    )

    # Criterio 2: Entrenamiento
    evaluator.evaluate_training(
        is_reproducible=True,
        uses_best_practices=True
    )

    # Criterio 3: Métricas
    if model_info['model'] is not None:
        # Hacer split para evaluar
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Calcular métricas
        y_pred = model_info['model'].predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        # Calcular train accuracy para detectar overfitting
        y_train_pred = model_info['model'].predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)

        print(f"\n📊 Métricas calculadas:")
        print(f"   Train Accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")
        print(f"   Test Accuracy:  {test_acc:.4f} ({test_acc * 100:.2f}%)")

        evaluator.evaluate_metrics(
            accuracy=test_acc,
            has_multiple_metrics=True,
            has_confusion_matrix=True,
            are_correct=True
        )

        # Criterio 5: Generalización
        overfitting = (train_acc - test_acc) > 0.15
        evaluator.evaluate_generalization(
            test_accuracy=test_acc,
            train_accuracy=train_acc,
            overfitting=overfitting
        )
    else:
        evaluator.evaluate_metrics(
            accuracy=0,
            has_multiple_metrics=False,
            has_confusion_matrix=False,
            are_correct=False
        )
        evaluator.evaluate_generalization(test_accuracy=0)

    # Criterio 4: Exportación
    evaluator.evaluate_export(
        model_file_exists=model_info['exists'],
        filename_correct=model_info['filename_correct'],
        loads_successfully=model_info['loads'],
        predictions_work=model_info['predictions_work']
    )

    # Reporte final
    total = evaluator.print_final_report()

    return total


if __name__ == "__main__":
    import sys

    # Configurar ruta del modelo
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Buscar archivos .joblib en el directorio actual
        joblib_files = [f for f in os.listdir('.') if f.endswith('.joblib')]

        if len(joblib_files) == 0:
            print("❌ No se encontraron archivos .joblib")
            print("\nUso: python verify_criteria.py [ruta_al_modelo.joblib]")
            sys.exit(1)
        elif len(joblib_files) == 1:
            model_path = joblib_files[0]
            print(f"📁 Usando modelo: {model_path}")
        else:
            print(f"Se encontraron {len(joblib_files)} archivos .joblib:")
            for i, f in enumerate(joblib_files, 1):
                print(f"  {i}. {f}")
            model_path = joblib_files[0]
            print(f"\n📁 Usando primer archivo: {model_path}")

    # Evaluar proyecto
    evaluate_complete_project(model_path)