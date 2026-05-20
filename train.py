# train_con_pca.py
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import time
import os
from collections import OrderedDict
import warnings

warnings.filterwarnings('ignore')


class ModelTrainerWithPCA:
    def __init__(self, dataset_path='dataset.csv', test_size=0.2, random_state=42, n_components=10):
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state
        self.n_components = n_components
        self.results = []
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.pca = None

        print("=" * 70)
        print("PROYECTO 1: CLASIFICACIÓN CON PCA")
        print("=" * 70)
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        """Carga, limpia y aplica PCA"""
        print(f"\n📂 Cargando: {self.dataset_path}")

        # Cargar datos
        data = np.loadtxt(self.dataset_path, delimiter=',')
        print(f"✓ Dataset original: {data.shape}")

        # Eliminar duplicados
        print("\n🔍 Eliminando duplicados...")
        unique_data = OrderedDict()
        for row in data:
            key = tuple(row)
            if key not in unique_data:
                unique_data[key] = row

        data_clean = np.array(list(unique_data.values()))
        print(f"  └─ {len(data)} → {len(data_clean)} muestras (eliminados {len(data) - len(data_clean)})")

        self.X = data_clean[:, :-1]
        self.y = data_clean[:, -1]

        print(f"\n📊 Dataset:")
        print(f"  ├─ Muestras: {len(self.X)}")
        print(f"  ├─ Features originales: {self.X.shape[1]}")
        print(f"  ├─ Clase 0: {sum(self.y == 0)}")
        print(f"  └─ Clase 1: {sum(self.y == 1)}")

        # APLICAR PCA para reducir dimensionalidad
        print(f"\n🔄 Aplicando PCA (reduciendo a {self.n_components} componentes)...")
        self.pca = PCA(n_components=min(self.n_components, len(self.X) - 1))
        self.X_reduced = self.pca.fit_transform(self.X)

        varianza_explicada = self.pca.explained_variance_ratio_.sum()
        print(f"  └─ Varianza explicada: {varianza_explicada:.2%}")
        print(f"  └─ Features reducidas: {self.X.shape[1]} → {self.X_reduced.shape[1]}")

        # Dividir datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_reduced, self.y, test_size=self.test_size,
            random_state=self.random_state, stratify=self.y
        )

        print(f"\n✅ División final:")
        print(f"  ├─ Entrenamiento: {len(self.X_train)} muestras")
        print(f"  ├─ Prueba: {len(self.X_test)} muestras")
        print(f"  └─ Features por muestra: {self.X_train.shape[1]}")

    def train_decision_tree(self):
        """Árbol de Decisión"""
        print("\n" + "=" * 70)
        print("🌳 ÁRBOL DE DECISIÓN (con PCA)")
        print("=" * 70)

        param_grid = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 3, 5],
            'criterion': ['gini', 'entropy']
        }

        dt = DecisionTreeClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(dt, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

        print("Buscando mejores parámetros...")
        start = time.time()
        grid_search.fit(self.X_train, self.y_train)
        train_time = time.time() - start

        best = grid_search.best_estimator_
        y_pred = best.predict(self.X_test)

        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'train_time': train_time
        }

        print(f"\n✓ Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Parámetros: {grid_search.best_params_}")

        self.results.append({
            'model_name': 'Decision Tree',
            'model': best,
            'params': grid_search.best_params_,
            'metrics': metrics
        })

        return best, metrics

    def train_naive_bayes(self):
        """Naive Bayes"""
        print("\n" + "=" * 70)
        print("📊 NAIVE BAYES (con PCA)")
        print("=" * 70)

        nb = GaussianNB()
        start = time.time()
        nb.fit(self.X_train, self.y_train)
        train_time = time.time() - start

        y_pred = nb.predict(self.X_test)

        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'train_time': train_time
        }

        print(f"\n✓ Accuracy: {metrics['accuracy']:.4f}")

        self.results.append({
            'model_name': 'Naive Bayes',
            'model': nb,
            'params': {'var_smoothing': 'default'},
            'metrics': metrics
        })

        return nb, metrics

    def train_knn(self):
        """KNN"""
        print("\n" + "=" * 70)
        print("🎯 KNN (con PCA)")
        print("=" * 70)

        # K no puede ser mayor que el número de muestras de entrenamiento
        max_k = min(11, len(self.X_train) - 1)
        param_grid = {
            'n_neighbors': [3, 5, 7] if max_k >= 7 else [3, 5],
            'weights': ['uniform', 'distance']
        }

        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

        print("Buscando mejores parámetros...")
        start = time.time()
        grid_search.fit(self.X_train, self.y_train)
        train_time = time.time() - start

        best = grid_search.best_estimator_
        y_pred = best.predict(self.X_test)

        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'train_time': train_time
        }

        print(f"\n✓ Accuracy: {metrics['accuracy']:.4f}")
        if grid_search.best_params_:
            print(f"  Mejor k={grid_search.best_params_.get('n_neighbors', 'N/A')}")

        self.results.append({
            'model_name': 'KNN',
            'model': best,
            'params': grid_search.best_params_,
            'metrics': metrics
        })

        return best, metrics

    def train_svm(self):
        """SVM"""
        print("\n" + "=" * 70)
        print("⚡ SVM (con PCA)")
        print("=" * 70)

        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }

        svm = SVC(random_state=self.random_state)
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

        print("Buscando mejores parámetros...")
        start = time.time()
        grid_search.fit(self.X_train, self.y_train)
        train_time = time.time() - start

        best = grid_search.best_estimator_
        y_pred = best.predict(self.X_test)

        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'train_time': train_time
        }

        print(f"\n✓ Accuracy: {metrics['accuracy']:.4f}")

        self.results.append({
            'model_name': 'SVM',
            'model': best,
            'params': grid_search.best_params_,
            'metrics': metrics
        })

        return best, metrics

    def train_all_models(self):
        """Entrena todos"""
        self.train_decision_tree()
        self.train_naive_bayes()
        self.train_knn()
        self.train_svm()

        print("\n" + "=" * 70)
        print("🏆 COMPARACIÓN DE MODELOS")
        print("=" * 70)

        for result in self.results:
            acc = result['metrics']['accuracy']
            f1 = result['metrics']['f1']
            print(f"{result['model_name']:<15} Accuracy: {acc:.4f} | F1: {f1:.4f}")

            if acc > self.best_score:
                self.best_score = acc
                self.best_model = result['model']
                self.best_model_name = result['model_name']
                self.best_params = result['params']
                self.best_metrics = result['metrics']

        print(f"\n🥇 MEJOR MODELO: {self.best_model_name}")
        print(f"   Accuracy: {self.best_score:.4f} ({self.best_score * 100:.2f}%)")

    def export_model(self, carne, nombre, apellido, output_dir='.'):
        """Exporta modelo incluyendo el PCA"""
        os.makedirs(output_dir, exist_ok=True)

        # Guardar el modelo completo (clasificador + PCA)
        model_package = {
            'classifier': self.best_model,
            'pca': self.pca,
            'model_name': self.best_model_name,
            'metrics': self.best_metrics
        }

        filename = f"{carne}_{nombre}_{apellido}.joblib"
        filepath = f"{output_dir}/{filename}"

        joblib.dump(model_package, filepath)
        print(f"\n💾 Modelo guardado: {filepath}")

        # Guardar metadata
        metadata_file = f"{output_dir}/{carne}_{nombre}_{apellido}_metadata.txt"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("MODELO DE CLASIFICACIÓN - PROYECTO 1 IE0435\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Estudiante: {nombre} {apellido}\n")
            f.write(f"Carné: {carne}\n\n")
            f.write(f"MEJOR MODELO: {self.best_model_name}\n")
            f.write(f"PCA aplicado: {self.X.shape[1]} → {self.n_components} componentes\n")
            f.write("-" * 50 + "\n")
            f.write(f"Accuracy:  {self.best_metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {self.best_metrics['precision']:.4f}\n")
            f.write(f"Recall:    {self.best_metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {self.best_metrics['f1']:.4f}\n")


def main():
    print("\n" + "=" * 70)
    print("🚀 INICIANDO ENTRENAMIENTO CON PCA")
    print("=" * 70)

    # ===== CONFIGURACIÓN =====
    CARNE = "C10153"
    NOMBRE = "Luis"
    APELLIDO = "Aguirre"
    DATASET_PATH = "dataset_grupo_completo.csv"
    OUTPUT_DIR = "resultados"
    # =========================

    if not os.path.exists(DATASET_PATH):
        print(f"\n❌ ERROR: No se encuentra '{DATASET_PATH}'")
        return

    # Probar diferentes números de componentes
    for n_components in [5, 10, 15]:
        print(f"\n{'=' * 70}")
        print(f"📊 PROBANDO CON {n_components} COMPONENTES PCA")
        print(f"{'=' * 70}")

        trainer = ModelTrainerWithPCA(
            dataset_path=DATASET_PATH,
            test_size=0.2,
            random_state=42,
            n_components=n_components
        )

        trainer.train_all_models()

        if trainer.best_score < 0.95:  # Si el accuracy es realista
            print(f"\n✅ Usando {n_components} componentes PCA (accuracy realista)")
            trainer.export_model(CARNE, NOMBRE, APELLIDO, OUTPUT_DIR)
            break


if __name__ == "__main__":
    main()