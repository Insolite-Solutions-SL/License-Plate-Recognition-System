#!/usr/bin/env python
import os
import subprocess
import argparse
import glob
import json
import matplotlib.pyplot as plt
import sys
from pathlib import Path


def evaluate_model(model_path, data_path, batch_size=16, image_size=640, device="0"):
    """
    Evalúa un modelo YOLOv11 entrenado usando los conjuntos de validación y prueba.

    Args:
        model_path (str): Ruta al archivo .pt del modelo
        data_path (str): Ruta al archivo data.yaml
        batch_size (int): Tamaño del batch para evaluación
        image_size (int): Tamaño de las imágenes
        device (str): Dispositivo para evaluación ('0' para primera GPU, 'cpu' para CPU)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo data.yaml en {data_path}")

    # Verificar que los directorios de datos existen
    # Leer la configuración de data.yaml
    import yaml

    try:
        with open(data_path, "r") as f:
            data_config = yaml.safe_load(f)

        # Verificar si los directorios de imágenes existen
        data_dir = os.path.dirname(data_path)
        val_dir = os.path.join(data_dir, data_config.get("val", ""))
        test_dir = os.path.join(data_dir, data_config.get("test", ""))

        if not os.path.exists(val_dir):
            print(f"ADVERTENCIA: No se encontró el directorio de validación: {val_dir}")
            print(
                "Es posible que la evaluación falle si los datos no están disponibles localmente."
            )

        if not os.path.exists(test_dir):
            print(f"ADVERTENCIA: No se encontró el directorio de prueba: {test_dir}")
            print(
                "Es posible que la evaluación falle si los datos no están disponibles localmente."
            )
    except Exception as e:
        print(f"Error al leer la configuración del archivo data.yaml: {e}")
        print(
            "Continuando de todos modos, pero pueden ocurrir errores durante la evaluación."
        )

    # Nombre base para los directorios de resultados
    model_name = os.path.basename(model_path).split(".")[0]

    # 1. Evaluar en el conjunto de validación
    print(f"\n=== Evaluando {model_name} en el conjunto de VALIDACIÓN ===")
    val_cmd = [
        "yolo",
        "task=detect",
        "mode=val",
        f"model={model_path}",
        f"data={data_path}",
        f"batch={batch_size}",
        f"imgsz={image_size}",
        f"device={device}",
        "save_json=True",
        "save_txt=True",
        "save_conf=True",
        "plots=True",
    ]

    print(f"Ejecutando: {' '.join(val_cmd)}")
    try:
        subprocess.run(val_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error durante la evaluación en el conjunto de validación: {e}")
        print(
            "Esto puede ocurrir si los datos de validación no están disponibles localmente."
        )

    # 2. Evaluar en el conjunto de prueba
    print(f"\n=== Evaluando {model_name} en el conjunto de PRUEBA ===")
    test_cmd = [
        "yolo",
        "task=detect",
        "mode=val",
        f"model={model_path}",
        f"data={data_path}",
        "split=test",  # Especificar que se use el conjunto de prueba
        f"batch={batch_size}",
        f"imgsz={image_size}",
        f"device={device}",
        "save_json=True",
        "save_txt=True",
        "save_conf=True",
        "name=test_results",
        "plots=True",
    ]

    print(f"Ejecutando: {' '.join(test_cmd)}")
    try:
        subprocess.run(test_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error durante la evaluación en el conjunto de prueba: {e}")
        print(
            "Esto puede ocurrir si los datos de prueba no están disponibles localmente."
        )

    return f"./runs/detect/val", f"./runs/detect/test_results"


def visualize_predictions(
    model_path, data_path, num_samples=10, image_size=640, device="0"
):
    """
    Visualiza predicciones en algunas imágenes aleatorias del conjunto de prueba.

    Args:
        model_path (str): Ruta al archivo .pt del modelo
        data_path (str): Ruta al archivo data.yaml
        num_samples (int): Número de muestras aleatorias para visualizar
        image_size (int): Tamaño de las imágenes
        device (str): Dispositivo para inferencia
    """
    # Obtener ruta al directorio de imágenes de prueba desde el archivo data.yaml
    import yaml

    with open(data_path, "r") as f:
        data_config = yaml.safe_load(f)

    test_images_dir = os.path.join(os.path.dirname(data_path), data_config["test"])

    # Verificar si el directorio existe
    if not os.path.exists(test_images_dir):
        print(
            f"ERROR: No se encontró el directorio de imágenes de prueba: {test_images_dir}"
        )
        print(
            "Saltando visualización de predicciones ya que las imágenes no están disponibles localmente."
        )
        return None

    # Listar todas las imágenes y seleccionar muestras aleatorias
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_files.extend(glob.glob(os.path.join(test_images_dir, ext)))

    if not image_files:
        print(f"No se encontraron imágenes en {test_images_dir}")
        return None

    # Seleccionar muestras aleatorias
    import random

    random.shuffle(image_files)
    samples = image_files[:num_samples]

    # Crear directorio para guardar las visualizaciones
    output_dir = f"./predictions_{os.path.basename(model_path).split('.')[0]}"
    os.makedirs(output_dir, exist_ok=True)

    # Ejecutar predicciones en muestras seleccionadas
    predict_cmd = [
        "yolo",
        "task=detect",
        "mode=predict",
        f"model={model_path}",
        f"source={','.join(samples)}",
        f"imgsz={image_size}",
        f"device={device}",
        f"save=True",
        f"save_txt=True",
        f"save_conf=True",
        f"project={output_dir}",
        "name=samples",
    ]

    print(
        f"\n=== Generando visualizaciones de predicciones para {len(samples)} imágenes ==="
    )
    print(f"Ejecutando: {' '.join(predict_cmd)}")

    try:
        subprocess.run(predict_cmd, check=True)
        print(f"Visualizaciones guardadas en: {os.path.join(output_dir, 'samples')}")
        return os.path.join(output_dir, "samples")
    except subprocess.CalledProcessError as e:
        print(f"Error al generar visualizaciones: {e}")
        print(
            "Esto puede ocurrir si las imágenes de prueba no están disponibles localmente."
        )
        return None


def analyze_results(val_results_dir, test_results_dir):
    """
    Analiza y muestra un resumen de los resultados de evaluación.

    Args:
        val_results_dir (str): Directorio con resultados de validación
        test_results_dir (str): Directorio con resultados de prueba
    """
    # Buscar archivos de resultados JSON
    result_files = {
        "val": os.path.join(val_results_dir, "results.json"),
        "test": os.path.join(test_results_dir, "results.json"),
    }

    print("\n=== ANÁLISIS DE RESULTADOS ===")

    metrics = {}
    found_results = False

    for split, result_file in result_files.items():
        if os.path.exists(result_file):
            found_results = True
            try:
                with open(result_file, "r") as f:
                    results = json.load(f)

                metrics[split] = {
                    "mAP50": results.get("metrics", {}).get("mAP50", 0),
                    "mAP50-95": results.get("metrics", {}).get("mAP50-95", 0),
                    "precision": results.get("metrics", {}).get("precision", 0),
                    "recall": results.get("metrics", {}).get("recall", 0),
                }

                print(f"\nResultados en conjunto de {split.upper()}:")
                print(f"- mAP@0.5: {metrics[split]['mAP50']:.4f}")
                print(f"- mAP@0.5-0.95: {metrics[split]['mAP50-95']:.4f}")
                print(f"- Precision: {metrics[split]['precision']:.4f}")
                print(f"- Recall: {metrics[split]['recall']:.4f}")
            except Exception as e:
                print(f"Error al leer los resultados para {split}: {e}")
        else:
            print(
                f"No se encontró el archivo de resultados para {split}: {result_file}"
            )

    if not found_results:
        print(
            "No se encontraron archivos de resultados. Esto puede deberse a que los datos no están disponibles localmente."
        )
        print(
            "Asegúrese de que los datos estén descargados y en la ubicación correcta para obtener resultados."
        )

    return metrics


def continue_training(
    model_path, data_path, epochs=20, batch_size=16, image_size=640, device="0"
):
    """
    Continúa el entrenamiento desde un modelo previamente entrenado.

    Args:
        model_path (str): Ruta al modelo pre-entrenado para continuar
        data_path (str): Ruta al archivo data.yaml
        epochs (int): Número de épocas adicionales
        batch_size (int): Tamaño del batch para entrenamiento
        image_size (int): Tamaño de las imágenes
        device (str): Dispositivo para entrenar
    """
    # Directorio de resultados basado en el nombre del modelo
    model_name = os.path.basename(model_path).split(".")[0]
    output_name = f"continue_{model_name}"

    # Obtener una lista de directorios existentes antes del entrenamiento
    existing_dirs = set(glob.glob("./runs/detect/*"))

    # Comando para continuar el entrenamiento
    train_cmd = [
        "yolo",
        "task=detect",
        "mode=train",
        f"model={model_path}",  # Usar el modelo existente como punto de partida
        f"data={data_path}",
        f"device={device}",
        "save=True",
        f"epochs={epochs}",
        f"batch={batch_size}",
        "val=True",
        "plots=True",
        f"imgsz={image_size}",
        f"name={output_name}",
    ]

    print(
        f"\n=== Continuando entrenamiento desde {model_path} por {epochs} épocas adicionales ==="
    )
    print(f"Ejecutando: {' '.join(train_cmd)}")

    result = subprocess.run(train_cmd, capture_output=True, text=True)

    # Analizamos la salida para encontrar el directorio donde se guardaron los resultados
    output_lines = result.stdout.split("\n") if result.stdout else []

    # Buscamos la línea que contiene "save_dir="
    save_dir = None
    for line in output_lines:
        if "save_dir=" in line:
            parts = line.strip().split()
            for part in parts:
                if part.startswith("save_dir="):
                    save_dir = part.split("=")[1]
                    break

    # Si no encontramos el directorio en la salida, buscamos comparando los directorios antes y después
    if not save_dir:
        new_dirs = set(glob.glob("./runs/detect/*"))
        added_dirs = new_dirs - existing_dirs

        # Buscar directorios que coincidan con el patrón "continue_*"
        continue_dirs = [
            d for d in added_dirs if os.path.basename(d).startswith("continue_")
        ]

        if continue_dirs:
            # Tomar el directorio más reciente
            save_dir = max(continue_dirs, key=os.path.getmtime)
        else:
            # Si no encontramos nada, usar el nombre esperado
            save_dir = f"./runs/detect/{output_name}"

    # Ruta completa al modelo
    best_weights = os.path.join(save_dir, "weights/best.pt")

    if os.path.exists(best_weights):
        print(
            f"\nEntrenamiento adicional completado. Mejor modelo guardado en: {best_weights}"
        )
        return best_weights
    else:
        print(
            f"\nNo se encontró el archivo de pesos después del entrenamiento adicional en: {best_weights}"
        )
        # Buscar en cualquier directorio de continue_ reciente
        continue_models = glob.glob("./runs/detect/continue_*/weights/best.pt")
        if continue_models:
            newest_model = max(continue_models, key=os.path.getmtime)
            print(f"Sin embargo, se encontró un modelo reciente en: {newest_model}")
            return newest_model
        return None


def plot_metrics_comparison(metrics):
    """
    Genera gráficas comparativas de las métricas entre los conjuntos de validación y prueba.

    Args:
        metrics (dict): Diccionario con las métricas para cada conjunto
    """
    if not metrics or len(metrics) < 1:
        print("No hay suficientes datos para generar gráficas comparativas")
        return

    # Verificar si tenemos métricas tanto para validación como para prueba
    if "val" not in metrics or "test" not in metrics:
        print(
            "Faltan datos para validación o prueba, no se pueden generar gráficas comparativas"
        )
        return

    # Verificar que tenemos todas las métricas necesarias
    required_metrics = ["mAP50", "mAP50-95", "precision", "recall"]
    for split in ["val", "test"]:
        for metric in required_metrics:
            if metric not in metrics[split]:
                print(
                    f"Falta la métrica {metric} para {split}, no se pueden generar gráficas comparativas"
                )
                return

    # Crear una figura con 4 subplots (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Comparación de Métricas entre Validación y Prueba", fontsize=16)

    # Configurar colores y etiquetas
    colors = ["#3498db", "#2ecc71"]  # Azul y verde
    splits = ["Validación", "Prueba"]

    # Graficar mAP@0.5
    axs[0, 0].bar(
        splits, [metrics["val"]["mAP50"], metrics["test"]["mAP50"]], color=colors
    )
    axs[0, 0].set_title("mAP@0.5")
    axs[0, 0].set_ylim(0, 1)

    # Graficar mAP@0.5-0.95
    axs[0, 1].bar(
        splits, [metrics["val"]["mAP50-95"], metrics["test"]["mAP50-95"]], color=colors
    )
    axs[0, 1].set_title("mAP@0.5-0.95")
    axs[0, 1].set_ylim(0, 1)

    # Graficar Precision
    axs[1, 0].bar(
        splits,
        [metrics["val"]["precision"], metrics["test"]["precision"]],
        color=colors,
    )
    axs[1, 0].set_title("Precision")
    axs[1, 0].set_ylim(0, 1)

    # Graficar Recall
    axs[1, 1].bar(
        splits, [metrics["val"]["recall"], metrics["test"]["recall"]], color=colors
    )
    axs[1, 1].set_title("Recall")
    axs[1, 1].set_ylim(0, 1)

    # Ajustar layout y guardar
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("./metrics_comparison.png")
    print("\nGráfica comparativa guardada como: metrics_comparison.png")


def find_best_models():
    """
    Busca y lista los mejores modelos disponibles en el directorio runs/detect.
    """
    model_dirs = glob.glob("./runs/detect/train*")
    model_dirs.extend(glob.glob("./runs/detect/continue_*"))

    models = []

    for model_dir in model_dirs:
        best_model = os.path.join(model_dir, "weights/best.pt")
        if os.path.exists(best_model):
            # Intentar obtener el rendimiento del modelo
            results_file = os.path.join(model_dir, "results.json")
            model_info = {
                "path": best_model,
                "name": os.path.basename(model_dir),
                "mAP50": "N/A",
                "mAP50-95": "N/A",
            }

            if os.path.exists(results_file):
                try:
                    with open(results_file, "r") as f:
                        results = json.load(f)
                    model_info["mAP50"] = results.get("metrics", {}).get("mAP50", "N/A")
                    model_info["mAP50-95"] = results.get("metrics", {}).get(
                        "mAP50-95", "N/A"
                    )
                except:
                    pass

            models.append(model_info)

    # Ordenar por mAP50-95 si está disponible
    try:
        models.sort(
            key=lambda x: float(x["mAP50-95"]) if x["mAP50-95"] != "N/A" else -1,
            reverse=True,
        )
    except:
        pass

    print("\n=== MODELOS DISPONIBLES ===")
    print(f"{'Modelo':<20} {'mAP@0.5':<10} {'mAP@0.5-0.95':<12} {'Ruta'}")
    print("-" * 80)

    for model in models:
        print(
            f"{model['name']:<20} {model['mAP50']:<10} {model['mAP50-95']:<12} {model['path']}"
        )

    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluar y analizar modelos YOLOv11 entrenados"
    )

    parser.add_argument("--model", type=str, help="Ruta al modelo .pt para evaluar")
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.yaml",
        help="Ruta al archivo data.yaml",
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Tamaño del batch para evaluación"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de las imágenes")
    parser.add_argument(
        "--device", type=str, default="0", help="Dispositivo para evaluación"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Número de imágenes de muestra para visualizar",
    )
    parser.add_argument(
        "--continue-epochs",
        type=int,
        default=0,
        help="Número de épocas adicionales para continuar entrenando (0 para omitir)",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="Listar modelos disponibles"
    )

    args = parser.parse_args()

    if args.list_models:
        best_models = find_best_models()
        sys.exit(0)

    if not args.model:
        print("Error: Debe especificar la ruta a un modelo con --model")
        parser.print_help()
        sys.exit(1)

    print("\n=== EVALUACIÓN Y ANÁLISIS DE MODELO ===")
    print(f"Modelo: {args.model}")
    print(f"Archivo de datos: {args.data}")
    print("-" * 50)

    try:
        # Evaluar el modelo
        val_dir, test_dir = evaluate_model(
            model_path=args.model,
            data_path=args.data,
            batch_size=args.batch,
            image_size=args.imgsz,
            device=args.device,
        )

        # Visualizar algunas predicciones
        vis_dir = visualize_predictions(
            model_path=args.model,
            data_path=args.data,
            num_samples=args.samples,
            image_size=args.imgsz,
            device=args.device,
        )

        # Analizar resultados
        metrics = analyze_results(val_dir, test_dir)

        # Generar gráficas comparativas si hay métricas disponibles
        if metrics and len(metrics) > 0:
            plot_metrics_comparison(metrics)
        else:
            print("\nNo hay suficientes datos para generar gráficas comparativas")

        # Continuar entrenamiento si se solicitó
        if args.continue_epochs > 0:
            new_model = continue_training(
                model_path=args.model,
                data_path=args.data,
                epochs=args.continue_epochs,
                batch_size=args.batch,
                image_size=args.imgsz,
                device=args.device,
            )

            if new_model:
                print("\nEvaluar el modelo mejorado:")
                print(
                    f"python {os.path.basename(__file__)} --model {new_model} --data {args.data}"
                )

        print("\n=== PROCESO COMPLETO ===")
        print(f"Resultados de validación: {val_dir}")
        print(f"Resultados de prueba: {test_dir}")
        print(f"Visualizaciones: {vis_dir if vis_dir else 'No se pudieron generar'}")
        print(
            "\nPara comparar con otros modelos o continuar entrenando, vuelva a ejecutar este script."
        )

    except Exception as e:
        print(f"\nError durante la evaluación: {e}")
        print("\nRecomendaciones:")
        print(
            "1. Verifique que el dataset esté disponible localmente en la ruta especificada."
        )
        print(
            "2. Si no tiene los datos localmente, descargue o genere el dataset con combineDatasets.py"
        )
        print("   python combineDatasets.py")
        print("3. Asegúrese de tener instalado ultralytics:")
        print("   pip install ultralytics")
        print(
            "4. Vuelva a intentar la evaluación con la ruta absoluta al directorio data:"
        )
        print(
            f"   python evaluateModel.py --model {args.model} --data $(pwd)/data/data.yaml"
        )
