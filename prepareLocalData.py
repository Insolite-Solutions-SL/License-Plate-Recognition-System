#!/usr/bin/env python
import os
import yaml
import argparse
import shutil
import sys


def create_minimal_dataset(
    output_dir="data", num_classes=1, class_names=["license_plate"]
):
    """
    Crea una estructura mínima de directorio para datos de YOLOv8/11 sin imágenes reales,
    pero con la estructura correcta para que los scripts de evaluación puedan ejecutarse sin errores.

    Args:
        output_dir (str): Directorio donde crear la estructura
        num_classes (int): Número de clases
        class_names (list): Nombres de las clases
    """
    print("Creando estructura mínima de dataset en:", output_dir)

    # Crear estructura de directorios
    for split in ["train", "valid", "test"]:
        for subdir in ["images", "labels"]:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)

    # Crear archivo data.yaml
    data_yaml = {
        "train": "./train/images",
        "val": "./valid/images",
        "test": "./test/images",
        "nc": num_classes,
        "names": class_names,
    }

    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print("\nEstructura de directorios creada:")
    print(f"- {output_dir}/")
    for split in ["train", "valid", "test"]:
        print(f"  - {split}/")
        for subdir in ["images", "labels"]:
            print(f"    - {subdir}/")

    print(f"\nArchivo data.yaml creado en: {os.path.join(output_dir, 'data.yaml')}")
    print("\nLa estructura está vacía (sin imágenes ni etiquetas).")
    print(
        "Esta estructura permite que el script evaluateModel.py se ejecute sin errores,"
    )
    print(
        "aunque no es posible generar visualizaciones ni resultados reales sin datos."
    )

    print("\nPróximos pasos:")
    print(
        "1. Para evaluación real, descargue los datos y ejecútelos a través de combineDatasets.py"
    )
    print(
        "2. O coloque manualmente sus imágenes y etiquetas en los directorios correspondientes:"
    )
    print("   - Imágenes: data/test/images/ y data/valid/images/")
    print("   - Etiquetas: data/test/labels/ y data/valid/labels/")
    print("3. Para evaluar el modelo sin errores:")
    print(
        "   python evaluateModel.py --model runs/detect/train2/weights/best.pt --data $(pwd)/data/data.yaml"
    )


def import_existing_data(source_dir, output_dir="data"):
    """
    Importa datos existentes de otro directorio a la estructura local.

    Args:
        source_dir (str): Directorio de origen con estructura de datos YOLOv8/11
        output_dir (str): Directorio de destino
    """
    if not os.path.exists(source_dir):
        print(f"¡Error! El directorio de origen {source_dir} no existe.")
        return False

    yaml_files = [f for f in os.listdir(source_dir) if f.endswith(".yaml")]
    if not yaml_files:
        print(f"¡Error! No se encontraron archivos YAML en {source_dir}.")
        return False

    # Usar el primer archivo YAML encontrado
    yaml_file = os.path.join(source_dir, yaml_files[0])

    try:
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error al leer el archivo YAML: {e}")
        return False

    # Crear estructura de directorios en el destino
    create_minimal_dataset(
        output_dir, config.get("nc", 1), config.get("names", ["object"])
    )

    # Copiar el archivo YAML configurado correctamente
    shutil.copy(yaml_file, os.path.join(output_dir, "data.yaml"))

    # Intentar copiar algunas imágenes de ejemplo si existen
    for split in ["test", "valid"]:
        src_img_dir = os.path.join(source_dir, split, "images")
        dst_img_dir = os.path.join(output_dir, split, "images")
        src_label_dir = os.path.join(source_dir, split, "labels")
        dst_label_dir = os.path.join(output_dir, split, "labels")

        if os.path.exists(src_img_dir):
            # Copiar hasta 10 imágenes de ejemplo
            image_files = [
                f
                for f in os.listdir(src_img_dir)
                if f.endswith((".jpg", ".jpeg", ".png"))
            ][:10]
            for img_file in image_files:
                try:
                    shutil.copy(
                        os.path.join(src_img_dir, img_file),
                        os.path.join(dst_img_dir, img_file),
                    )
                    # Intentar copiar la etiqueta correspondiente si existe
                    base_name = os.path.splitext(img_file)[0]
                    label_file = f"{base_name}.txt"
                    if os.path.exists(os.path.join(src_label_dir, label_file)):
                        shutil.copy(
                            os.path.join(src_label_dir, label_file),
                            os.path.join(dst_label_dir, label_file),
                        )
                except Exception as e:
                    print(f"Error al copiar archivo: {e}")

    print(f"\nSe ha importado la configuración desde {source_dir} a {output_dir}")
    print(f"Se han copiado algunas imágenes de ejemplo para pruebas.")
    print("\nPara evaluar el modelo:")
    print(
        f"python evaluateModel.py --model runs/detect/train2/weights/best.pt --data $(pwd)/{output_dir}/data.yaml"
    )

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preparar datos locales para evaluación de YOLOv8/11"
    )
    parser.add_argument(
        "--output", type=str, default="data", help="Directorio de salida para los datos"
    )
    parser.add_argument(
        "--import-from",
        type=str,
        help="Importar datos desde un directorio existente",
        dest="import_from",
    )

    args = parser.parse_args()

    if args.import_from:
        success = import_existing_data(args.import_from, args.output)
        if not success:
            sys.exit(1)
    else:
        create_minimal_dataset(args.output)
