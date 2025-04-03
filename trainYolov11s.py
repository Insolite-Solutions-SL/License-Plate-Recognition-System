#!/usr/bin/env python
import os
import subprocess
import argparse


def train_yolov11s(data_path, epochs=20, batch_size=16, image_size=640, device="0"):
    """
    Entrena un modelo YOLOv11s utilizando el dataset combinado.

    Args:
        data_path (str): Ruta al archivo data.yaml
        epochs (int): Número de épocas para el entrenamiento
        batch_size (int): Tamaño del batch para entrenamiento
        image_size (int): Tamaño de las imágenes para entrenamiento (cuadrado)
        device (str): Dispositivo para entrenar ('0' para primera GPU, 'cpu' para CPU)
    """
    # Verifica que el archivo data.yaml exista
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo data.yaml en {data_path}")

    # Comando para entrenar el modelo YOLOv11s
    train_cmd = [
        "yolo",
        "task=detect",
        "mode=train",
        f"model=yolo11s.pt",
        f"data={data_path}",
        f"device={device}",
        "save=True",
        f"epochs={epochs}",
        f"batch={batch_size}",
        "val=True",
        "plots=True",
        f"imgsz={image_size}",
    ]

    # Ejecutar el comando
    print(f"Ejecutando: {' '.join(train_cmd)}")
    subprocess.run(train_cmd)

    # Ruta donde se guardan los resultados del entrenamiento (por defecto)
    results_path = "./runs/detect/train"
    best_weights = os.path.join(results_path, "weights/best.pt")

    # Validar el modelo después del entrenamiento
    if os.path.exists(best_weights):
        print(f"\nEntrenamiento completado. Mejor modelo guardado en: {best_weights}")

        # Comando para evaluar el modelo
        val_cmd = [
            "yolo",
            "task=detect",
            "mode=val",
            f"model={best_weights}",
            f"data={data_path}",
            "save_json=True",
            "plots=True",
        ]

        print(f"\nEvaluando modelo: {' '.join(val_cmd)}")
        subprocess.run(val_cmd)

        print("\nProceso completo. Revisa los resultados en la carpeta 'runs/detect'.")
    else:
        print(
            f"\nEntrenamiento completado, pero no se encontró el archivo de pesos: {best_weights}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenar modelo YOLOv11s para detección de placas"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.yaml",
        help="Ruta al archivo data.yaml",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Número de épocas para entrenar"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Tamaño del batch para entrenar"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Tamaño de las imágenes (cuadrado)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Dispositivo para entrenar ('0' para primera GPU, 'cpu' para CPU)",
    )

    args = parser.parse_args()

    print("Iniciando entrenamiento de YOLOv11s para detección de placas de matrícula")
    print(f"Archivo de datos: {args.data}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Tamaño de imagen: {args.imgsz}x{args.imgsz}")
    print(f"Dispositivo: {args.device}")
    print("-" * 50)

    train_yolov11s(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        device=args.device,
    )
