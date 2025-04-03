import os
import shutil
import yaml

# Rutas de los datasets originales
dataset_paths = [
    "content/License-Plate-Recognition-4",
    "content/License-Plates-of-Vehicles-in-Turkey-150",
    "content/Vehicle-Registration-Plates-2",
]

# Ruta para el dataset combinado
output_dir = "data"

# Crear estructura de directorios para el dataset combinado
for split in ["train", "valid", "test"]:
    for subdir in ["images", "labels"]:
        os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)


# Función para combinar los datasets
def combine_datasets():
    # Contador para el número de imágenes por conjunto
    count = {"train": 0, "valid": 0, "test": 0}

    # Procesar cada dataset
    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        print(f"Procesando dataset: {dataset_name}")

        # Copiar imágenes y etiquetas para cada conjunto (train, valid, test)
        for split in ["train", "valid", "test"]:
            # Path para las imágenes y etiquetas
            images_src = os.path.join(dataset_path, split, "images")
            labels_src = os.path.join(dataset_path, split, "labels")

            # Path destino
            images_dst = os.path.join(output_dir, split, "images")
            labels_dst = os.path.join(output_dir, split, "labels")

            # Verificar si existen los directorios de origen
            if not os.path.exists(images_src):
                print(f"No se encontró el directorio: {images_src}")
                continue

            if not os.path.exists(labels_src):
                print(f"No se encontró el directorio: {labels_src}")
                continue

            # Copiar las imágenes
            for img_file in os.listdir(images_src):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    # Generar un nombre único para evitar colisiones
                    base_name, ext = os.path.splitext(img_file)
                    new_name = f"{dataset_name}_{base_name}{ext}"

                    # Copiar la imagen con el nuevo nombre
                    shutil.copy(
                        os.path.join(images_src, img_file),
                        os.path.join(images_dst, new_name),
                    )

                    # Si hay un archivo de etiqueta correspondiente, copiarlo también
                    label_file = f"{base_name}.txt"
                    label_path = os.path.join(labels_src, label_file)
                    if os.path.exists(label_path):
                        shutil.copy(
                            label_path,
                            os.path.join(labels_dst, f"{dataset_name}_{label_file}"),
                        )

                    count[split] += 1

    return count


# Crear el archivo data.yaml para YOLOv11
def create_data_yaml(count):
    data_yaml = {
        "train": "./train/images",
        "val": "./valid/images",
        "test": "./test/images",
        "nc": 1,  # Número de clases (solo placas de matrícula)
        "names": ["license_plate"],  # Nombre de la clase
    }

    # Escribir el archivo data.yaml
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print("\nResumen de imágenes copiadas:")
    print(f"- Train: {count['train']} imágenes")
    print(f"- Validation: {count['valid']} imágenes")
    print(f"- Test: {count['test']} imágenes")
    print(f"Total: {sum(count.values())} imágenes")
    print(
        f"\nSe ha creado el archivo data.yaml en {os.path.join(output_dir, 'data.yaml')}"
    )


# Ejecutar el proceso de combinación
if __name__ == "__main__":
    print("Iniciando la combinación de datasets...")
    count = combine_datasets()
    create_data_yaml(count)
    print(
        "\nProceso completado. Ahora puedes entrenar YOLOv11s con el dataset combinado."
    )
    print("Comando para entrenar:")
    print(
        '!yolo task=detect mode=train model=yolo11s.pt data="./data/data.yaml" save=True epochs=20 val=True plots=True imgsz=640'
    )
