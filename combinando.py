# combinar_datasets.py
import numpy as np
import os
from glob import glob


def combinar_todos_datasets(carpeta_datasets="datasets", archivo_salida="dataset_grupo_completo.csv"):
    """
    Combina todos los CSV de una carpeta en un solo dataset

    Args:
        carpeta_datasets: Carpeta donde están los CSV de tus compañeros
        archivo_salida: Nombre del archivo combinado
    """
    print("=" * 70)
    print("🔗 COMBINANDO DATASETS DEL GRUPO")
    print("=" * 70)

    # Buscar todos los archivos .csv en la carpeta
    archivos_csv = glob(f"{carpeta_datasets}/*.csv")

    if not archivos_csv:
        print(f"❌ No se encontraron archivos CSV en la carpeta '{carpeta_datasets}'")
        print(f"   Asegúrate de que la carpeta exista y contenga los CSV")
        return None

    print(f"\n📂 Encontrados {len(archivos_csv)} archivos CSV:")
    for archivo in archivos_csv:
        print(f"   └─ {os.path.basename(archivo)}")

    # Cargar y verificar cada dataset
    datasets = []
    total_muestras = 0

    print("\n🔍 Verificando cada dataset...")
    print("-" * 70)

    for i, archivo in enumerate(archivos_csv, 1):
        try:
            # Cargar dataset
            data = np.loadtxt(archivo, delimiter=',')
            n_muestras = data.shape[0]
            n_columnas = data.shape[1]

            # Verificar formato
            n_features = n_columnas - 1  # menos la columna de label

            print(f"\n{i}. {os.path.basename(archivo)}:")
            print(f"   ├─ Muestras: {n_muestras}")
            print(f"   ├─ Features: {n_features} ({int(n_features ** 0.5)}x{int(n_features ** 0.5)} pixeles)")
            print(f"   └─ Labels: 0={sum(data[:, -1] == 0)}, 1={sum(data[:, -1] == 1)}")

            # Verificar que las features sean binarias (0 y 1)
            valores_features = np.unique(data[:, :-1])
            if not np.all(np.isin(valores_features, [0, 1])):
                print(f"   ⚠️  ADVERTENCIA: Features no binarios detectados: {valores_features[:5]}")

            # Verificar labels correctos
            labels_unicos = np.unique(data[:, -1])
            if not np.all(np.isin(labels_unicos, [0, 1])):
                print(f"   ❌ ERROR: Labels inválidos: {labels_unicos}")
                print(f"      Deben ser solo 0 (sin arroz) y 1 (con arroz)")
                continue

            datasets.append(data)
            total_muestras += n_muestras

        except Exception as e:
            print(f"   ❌ Error al cargar {archivo}: {e}")
            continue

    if not datasets:
        print("\n❌ No se pudo cargar ningún dataset válido")
        return None

    # Combinar todos los datasets
    print("\n" + "=" * 70)
    print("🔄 COMBINANDO DATASETS...")
    print("=" * 70)

    data_combinada = np.vstack(datasets)

    # Verificar el resultado final
    print(f"\n✅ DATASET COMBINADO FINAL:")
    print(f"   ├─ Total muestras: {data_combinada.shape[0]}")
    print(f"   ├─ Total features: {data_combinada.shape[1] - 1}")
    print(f"   ├─ Muestras con arroz (clase 1): {sum(data_combinada[:, -1] == 1)}")
    print(f"   ├─ Muestras sin arroz (clase 0): {sum(data_combinada[:, -1] == 0)}")
    print(
        f"   └─ Proporción clases: {100 * sum(data_combinada[:, -1] == 1) / len(data_combinada):.1f}% / {100 * sum(data_combinada[:, -1] == 0) / len(data_combinada):.1f}%")

    # Guardar dataset combinado
    np.savetxt(archivo_salida, data_combinada, delimiter=',', fmt='%.0f')
    print(f"\n💾 Dataset guardado en: {archivo_salida}")

    # Guardar metadata
    with open("dataset_metadata.txt", "w", encoding="utf-8") as f:
        f.write("METADATA DEL DATASET COMBINADO\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total estudiantes: {len(archivos_csv)}\n")
        f.write(f"Total muestras: {data_combinada.shape[0]}\n")
        f.write(f"Features por imagen: {data_combinada.shape[1] - 1}\n")
        f.write(f"Muestras positivas (con arroz): {sum(data_combinada[:, -1] == 1)}\n")
        f.write(f"Muestras negativas (sin arroz): {sum(data_combinada[:, -1] == 0)}\n\n")
        f.write("Archivos incluidos:\n")
        for archivo in archivos_csv:
            f.write(f"  - {os.path.basename(archivo)}\n")

    print(f"📝 Metadata guardada en: dataset_metadata.txt")

    return archivo_salida


if __name__ == "__main__":
    # Combinar todos los datasets
    dataset_combinado = combinar_todos_datasets(
        carpeta_datasets="datasets",  # Carpeta donde están los CSV
        archivo_salida="dataset_grupo_completo.csv"
    )

    if dataset_combinado:
        print("\n✅ ¡Listo! Ahora puedes ejecutar tu entrenamiento con este dataset")
        print(f"   python entrenamiento.py")