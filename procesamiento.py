import os
import sys
import numpy as np
from PIL import Image
import glob


def preprocess_and_create_dataset(
        positive_dir='imagenes/positivas',
        negative_dir='imagenes/negativas',
        output_csv='dataset.csv',
        output_preprocessed='preprocessed_images',
        target_size=(128, 128),
        threshold=128
):
    """
    Función principal que realiza todo el preprocesamiento
    """

    print("=" * 70)
    print("PROYECTO 1: PROCESAMIENTO DE IMÁGENES - IE0435")
    print("=" * 70)

    # Verificar que existan las carpetas
    if not os.path.exists(positive_dir):
        print(f"\n❌ ERROR: No existe la carpeta '{positive_dir}'")
        print("Por favor crea la carpeta y coloca las imágenes positivas (con arroz)")
        return None

    if not os.path.exists(negative_dir):
        print(f"\n❌ ERROR: No existe la carpeta '{negative_dir}'")
        print("Por favor crea la carpeta y coloca las imágenes negativas (sin arroz)")
        return None

    # Crear carpeta para imágenes preprocesadas
    os.makedirs(output_preprocessed, exist_ok=True)
    os.makedirs(os.path.join(output_preprocessed, 'positivas'), exist_ok=True)
    os.makedirs(os.path.join(output_preprocessed, 'negativas'), exist_ok=True)

    dataset = []

    # Procesar imágenes POSITIVAS (con arroz) - Label = 1
    print(f"\n📂 Procesando imágenes POSITIVAS desde: {positive_dir}")
    print("-" * 70)

    positive_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        positive_files.extend(glob.glob(os.path.join(positive_dir, ext)))

    if len(positive_files) == 0:
        print(f"⚠️  ADVERTENCIA: No se encontraron imágenes en {positive_dir}")

    for idx, img_path in enumerate(positive_files, 1):
        try:
            # 1. Cargar imagen
            img = Image.open(img_path)

            # 2. Convertir a escala de grises
            img_gray = img.convert('L')

            # 3. Redimensionar a 128x128
            img_resized = img_gray.resize(target_size, Image.Resampling.LANCZOS)

            # Guardar imagen preprocesada
            save_path = os.path.join(
                output_preprocessed,
                'positivas',
                f'pos_{idx:03d}_{os.path.basename(img_path)}'
            )
            img_resized.save(save_path)

            # 4. Convertir a array numpy
            img_array = np.array(img_resized)

            # 5. Binarizar: 1 si >= threshold (blanco), 0 si < threshold (oscuro)
            binary_matrix = (img_array >= threshold).astype(int)

            # 6. Aplanar a vector (128*128 = 16,384 elementos)
            feature_vector = binary_matrix.flatten()

            # 7. Agregar label = 1 (positiva)
            feature_vector_with_label = np.append(feature_vector, 1)

            dataset.append(feature_vector_with_label)

            # Estadísticas
            white_pct = 100 * np.sum(binary_matrix == 1) / binary_matrix.size
            black_pct = 100 * np.sum(binary_matrix == 0) / binary_matrix.size

            print(f"  [{idx:2d}] ✓ {os.path.basename(img_path):40s} | "
                  f"Blancos: {white_pct:5.1f}% | Oscuros: {black_pct:5.1f}%")

        except Exception as e:
            print(f"  [{idx:2d}] ✗ {os.path.basename(img_path):40s} | ERROR: {e}")

    # Procesar imágenes NEGATIVAS (sin arroz) - Label = 0
    print(f"\n📂 Procesando imágenes NEGATIVAS desde: {negative_dir}")
    print("-" * 70)

    negative_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        negative_files.extend(glob.glob(os.path.join(negative_dir, ext)))

    if len(negative_files) == 0:
        print(f"⚠️  ADVERTENCIA: No se encontraron imágenes en {negative_dir}")

    for idx, img_path in enumerate(negative_files, 1):
        try:
            img = Image.open(img_path)
            img_gray = img.convert('L')
            img_resized = img_gray.resize(target_size, Image.Resampling.LANCZOS)

            save_path = os.path.join(
                output_preprocessed,
                'negativas',
                f'neg_{idx:03d}_{os.path.basename(img_path)}'
            )
            img_resized.save(save_path)

            img_array = np.array(img_resized)
            binary_matrix = (img_array >= threshold).astype(int)
            feature_vector = binary_matrix.flatten()
            feature_vector_with_label = np.append(feature_vector, 0)

            dataset.append(feature_vector_with_label)

            white_pct = 100 * np.sum(binary_matrix == 1) / binary_matrix.size
            black_pct = 100 * np.sum(binary_matrix == 0) / binary_matrix.size

            print(f"  [{idx:2d}] ✓ {os.path.basename(img_path):40s} | "
                  f"Blancos: {white_pct:5.1f}% | Oscuros: {black_pct:5.1f}%")

        except Exception as e:
            print(f"  [{idx:2d}] ✗ {os.path.basename(img_path):40s} | ERROR: {e}")

    # Convertir a array numpy y guardar
    if len(dataset) == 0:
        print("\n❌ ERROR: No se procesaron imágenes")
        return None

    dataset_array = np.array(dataset)

    print(f"\n💾 Guardando dataset en: {output_csv}")
    np.savetxt(output_csv, dataset_array, delimiter=',', fmt='%d')

    # Resumen
    print("\n" + "=" * 70)
    print("📊 RESUMEN DEL PROCESAMIENTO")
    print("=" * 70)
    print(f"Total de imágenes procesadas:     {len(dataset)}")
    print(f"  └─ Positivas (con arroz):       {len(positive_files)}")
    print(f"  └─ Negativas (sin arroz):       {len(negative_files)}")
    print(f"\nDimensiones del dataset:          {dataset_array.shape}")
    print(f"  └─ Filas (imágenes):            {dataset_array.shape[0]}")
    print(f"  └─ Columnas (píxeles + label):  {dataset_array.shape[1]}")
    print(f"\nDistribución de clases:")
    print(f"  └─ Clase 1 (arroz):             {int(np.sum(dataset_array[:, -1] == 1))}")
    print(f"  └─ Clase 0 (no arroz):          {int(np.sum(dataset_array[:, -1] == 0))}")
    print(f"\nArchivos generados:")
    print(f"  ✓ {output_csv}")
    print(f"  ✓ {output_preprocessed}/positivas/ ({len(positive_files)} imágenes)")
    print(f"  ✓ {output_preprocessed}/negativas/ ({len(negative_files)} imágenes)")
    print("=" * 70)

    # Mostrar ejemplo de una fila
    print("\n📋 Ejemplo de una fila del dataset (primeros 20 píxeles + label):")
    print(f"   {dataset_array[0, :20]} ... {int(dataset_array[0, -1])}")

    print("\n✅ Procesamiento completado exitosamente")
    print(f"\n💡 Siguiente paso: Entrenar modelos con {output_csv}")

    return dataset_array


def main():
    """
    Punto de entrada principal
    """
    # Configuración
    config = {
        'positive_dir': 'imagenes/positivas',
        'negative_dir': 'imagenes/negativas',
        'output_csv': 'dataset.csv',
        'output_preprocessed': 'preprocessed_images',
        'target_size': (128, 128),
        'threshold': 128  # Ajusta este valor si las matrices no se ven bien
    }

    # Procesar
    dataset = preprocess_and_create_dataset(**config)

    if dataset is not None:
        print("\n" + "=" * 70)
        print("🎓 DATOS LISTOS PARA EL PUNTO 3.4 (MODELADO)")
        print("=" * 70)
        print("\nPuedes continuar con el entrenamiento de modelos usando:")
        print("  - sklearn.tree.DecisionTreeClassifier")
        print("  - sklearn.naive_bayes.GaussianNB")
        print("  - sklearn.neighbors.KNeighborsClassifier")
        print("  - sklearn.svm.SVC")
        print("\nVer ejemplo en el README.md")
        print("=" * 70)


if __name__ == "__main__":
    main()