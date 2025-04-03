#!/usr/bin/env python
import os
import argparse
import glob
import matplotlib.pyplot as plt
import sys
import tempfile
import shutil
import datetime
import random
import yaml
import traceback

# Intentar importar YOLO una sola vez
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def evaluate_model(model_path, data_path, batch_size=16, image_size=640, device="0"):
    """
    Evalúa un modelo YOLOv11 entrenado usando los conjuntos de validación y prueba.
    Utiliza la API de Python de Ultralytics para obtener métricas precisas.

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

    # Verificar que YOLO está disponible
    if not YOLO_AVAILABLE:
        print("⚠️ Error: El módulo 'ultralytics' no está instalado.")
        print("Instale ultralytics con: pip install ultralytics")
        raise ImportError("Se requiere el módulo 'ultralytics' para evaluar el modelo")

    # Usar la API de Python de Ultralytics
    try:
        print("\n=== Evaluando modelo usando la API de Ultralytics ===")
        print(f"Cargando modelo desde {model_path}...")
        model = YOLO(model_path)

        # Evaluar en conjunto de validación
        print(f"\n=== Evaluando en conjunto de VALIDACIÓN ===")
        val_metrics = model.val(
            data=data_path,
            batch=batch_size,
            imgsz=image_size,
            device=device,
            save_json=True,
            save_txt=True,
            save_conf=True,
            plots=True,
        )

        # Extraer las métricas del objeto retornado
        val_results = {
            "mAP50": float(val_metrics.box.map50),
            "mAP50-95": float(val_metrics.box.map),
            "precision": float(val_metrics.box.mp),
            "recall": float(val_metrics.box.mr),
        }

        # Obtener directorio de resultados
        val_dir = (
            val_metrics.save_dir
            if hasattr(val_metrics, "save_dir")
            else "./runs/detect/val"
        )

        print("\nMétricas de VALIDACIÓN:")
        print(f"- mAP@0.5: {val_results['mAP50']:.4f}")
        print(f"- mAP@0.5-0.95: {val_results['mAP50-95']:.4f}")
        print(f"- Precision: {val_results['precision']:.4f}")
        print(f"- Recall: {val_results['recall']:.4f}")
        print(f"- Resultados guardados en: {val_dir}")

        # Evaluar en conjunto de prueba
        print(f"\n=== Evaluando en conjunto de PRUEBA ===")
        test_metrics = model.val(
            data=data_path,
            split="test",
            batch=batch_size,
            imgsz=image_size,
            device=device,
            save_json=True,
            save_txt=True,
            save_conf=True,
            name="test_results",
            plots=True,
        )

        # Extraer las métricas del objeto retornado
        test_results = {
            "mAP50": float(test_metrics.box.map50),
            "mAP50-95": float(test_metrics.box.map),
            "precision": float(test_metrics.box.mp),
            "recall": float(test_metrics.box.mr),
        }

        # Obtener directorio de resultados
        test_dir = (
            test_metrics.save_dir
            if hasattr(test_metrics, "save_dir")
            else "./runs/detect/test_results"
        )

        print("\nMétricas de PRUEBA:")
        print(f"- mAP@0.5: {test_results['mAP50']:.4f}")
        print(f"- mAP@0.5-0.95: {test_results['mAP50-95']:.4f}")
        print(f"- Precision: {test_results['precision']:.4f}")
        print(f"- Recall: {test_results['recall']:.4f}")
        print(f"- Resultados guardados en: {test_dir}")

        return val_dir, test_dir, val_results, test_results

    except Exception as e:
        print(f"❌ Error durante la evaluación con API: {e}")
        traceback.print_exc()
        raise


def visualize_predictions(
    model_path,
    data_path,
    num_samples=10,
    image_size=640,
    device="0",
    metrics=None,
    output_dir=None,
):
    """
    Visualiza predicciones en algunas imágenes aleatorias del conjunto de prueba
    usando la API de Ultralytics.

    Args:
        model_path (str): Ruta al archivo .pt del modelo
        data_path (str): Ruta al archivo data.yaml
        num_samples (int): Número de muestras aleatorias para visualizar
        image_size (int): Tamaño de las imágenes
        device (str): Dispositivo para inferencia
        metrics (dict): Diccionario con métricas para incluir en el README
        output_dir (str): Directorio de salida (requerido)
    """
    # Verificar que YOLO está disponible
    if not YOLO_AVAILABLE:
        print("⚠️ Error: El módulo 'ultralytics' no está instalado.")
        print("Instale ultralytics con: pip install ultralytics")
        return None

    # Verificar que se proporcionó un directorio de salida
    if output_dir is None:
        print("⚠️ Error: Se requiere un directorio de salida (output_dir)")
        return None

    try:
        print("\n=== Visualizando predicciones con API de Ultralytics ===")

        # Cargar el modelo
        model = YOLO(model_path)

        # Obtener ruta al directorio de imágenes de prueba
        with open(data_path, "r") as f:
            data_config = yaml.safe_load(f)

        # Normalizar la ruta del directorio de test
        test_path = data_config["test"]
        test_images_dir = os.path.normpath(
            os.path.join(os.path.dirname(data_path), test_path)
        )

        # Listar todas las imágenes y seleccionar muestras aleatorias
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_files.extend(glob.glob(os.path.join(test_images_dir, ext)))

        if not image_files:
            print(f"No se encontraron imágenes en {test_images_dir}")
            return None

        # Seleccionar muestras aleatorias
        random.shuffle(image_files)
        samples = image_files[:num_samples]

        # Asegurarse de que el directorio de salida existe
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n=== Generando visualizaciones para {len(samples)} imágenes ===")

        # Usar un directorio temporal para las predicciones
        with tempfile.TemporaryDirectory() as temp_images_dir:
            # Copiar las imágenes al directorio temporal
            for i, img_path in enumerate(samples):
                temp_file = os.path.join(temp_images_dir, f"sample_{i}.jpg")
                shutil.copy(img_path, temp_file)

            # Crear un directorio temporal para los resultados
            with tempfile.TemporaryDirectory() as temp_results_dir:
                # Realizar predicciones usando la API y guardar en el directorio temporal
                results = model.predict(
                    source=temp_images_dir,
                    imgsz=image_size,
                    device=device,
                    save=True,
                    save_txt=True,
                    save_conf=True,
                    project=temp_results_dir,
                    name="temp_predictions",
                )

                # Directorio donde YOLO guardó los resultados
                yolo_results_dir = os.path.join(temp_results_dir, "temp_predictions")

                # Copiar todos los archivos generados al directorio de salida
                for item in os.listdir(yolo_results_dir):
                    src_path = os.path.join(yolo_results_dir, item)
                    dst_path = os.path.join(output_dir, item)

                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                    elif os.path.isdir(src_path):
                        # Si el destino ya existe, eliminarlo primero para evitar errores
                        if os.path.exists(dst_path):
                            shutil.rmtree(dst_path)
                        shutil.copytree(src_path, dst_path)

            print(f"Visualizaciones guardadas en: {output_dir}")

            # Crear un archivo README con métricas e información
            with open(os.path.join(output_dir, "README.txt"), "w") as f:
                f.write(f"# Resultados de evaluación del modelo\n\n")
                f.write(f"Modelo evaluado: {model_path}\n")
                f.write(
                    f"Fecha de evaluación: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"Tamaño de imagen: {image_size}x{image_size}\n")
                f.write(f"Número de muestras visualizadas: {num_samples}\n\n")

                # Incluir métricas si están disponibles
                if metrics and "val" in metrics and "test" in metrics:
                    f.write(f"## Métricas\n\n")
                    f.write(f"### Validación\n")
                    f.write(f"- mAP@0.5: {metrics['val']['mAP50']:.4f}\n")
                    f.write(f"- mAP@0.5-0.95: {metrics['val']['mAP50-95']:.4f}\n")
                    f.write(f"- Precision: {metrics['val']['precision']:.4f}\n")
                    f.write(f"- Recall: {metrics['val']['recall']:.4f}\n\n")

                    f.write(f"### Prueba\n")
                    f.write(f"- mAP@0.5: {metrics['test']['mAP50']:.4f}\n")
                    f.write(f"- mAP@0.5-0.95: {metrics['test']['mAP50-95']:.4f}\n")
                    f.write(f"- Precision: {metrics['test']['precision']:.4f}\n")
                    f.write(f"- Recall: {metrics['test']['recall']:.4f}\n\n")

                    # Calcular promedios
                    avg_map50 = (metrics["val"]["mAP50"] + metrics["test"]["mAP50"]) / 2
                    avg_map50_95 = (
                        metrics["val"]["mAP50-95"] + metrics["test"]["mAP50-95"]
                    ) / 2

                    f.write(f"### Evaluación de calidad\n")
                    if avg_map50 > 0.95 and avg_map50_95 > 0.7:
                        f.write("Calidad del modelo: EXCELENTE\n")
                        f.write("- Alta precisión en detección de placas\n")
                        f.write("- Adecuado para implementación en producción\n")
                    elif avg_map50 > 0.90 and avg_map50_95 > 0.65:
                        f.write("Calidad del modelo: MUY BUENA\n")
                        f.write("- Buen rendimiento general\n")
                        f.write(
                            "- Puede considerar entrenamiento adicional para perfeccionar\n"
                        )
                    elif avg_map50 > 0.85 and avg_map50_95 > 0.6:
                        f.write("Calidad del modelo: BUENA\n")
                        f.write("- Rendimiento aceptable\n")
                        f.write("- Recomendado continuar entrenamiento para mejorar\n")
                    else:
                        f.write("Calidad del modelo: NECESITA MEJORA\n")
                        f.write(
                            "- Considere entrenamiento adicional o ajuste de hiperparámetros\n"
                        )
                        f.write("- Evalúe si el conjunto de datos necesita mejoras\n")

            return output_dir

    except Exception as e:
        print(f"❌ Error al generar visualizaciones: {e}")
        return None


def plot_metrics_comparison(metrics, output_dir=None):
    """
    Genera gráficas comparativas de las métricas entre los conjuntos de validación y prueba.

    Args:
        metrics (dict): Diccionario con las métricas para cada conjunto
        output_dir (str): Directorio donde guardar la imagen
    """
    if not metrics or len(metrics) < 2:
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

    # Guardar solo en el directorio de resultados
    if output_dir:
        plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
        print(f"\nGráfica comparativa guardada en: {output_dir}/metrics_comparison.png")
    else:
        print(
            "\nNo se pudo guardar la gráfica comparativa, no se proporcionó un directorio"
        )


def continue_training(
    model_path, data_path, epochs=20, batch_size=16, image_size=640, device="0"
):
    """
    Continúa el entrenamiento desde un modelo previamente entrenado
    usando la API de Ultralytics.

    Args:
        model_path (str): Ruta al modelo pre-entrenado para continuar
        data_path (str): Ruta al archivo data.yaml
        epochs (int): Número de épocas adicionales
        batch_size (int): Tamaño del batch para entrenamiento
        image_size (int): Tamaño de las imágenes
        device (str): Dispositivo para entrenar
    """
    # Verificar que YOLO está disponible
    if not YOLO_AVAILABLE:
        print("⚠️ Error: El módulo 'ultralytics' no está instalado.")
        print("Instale ultralytics con: pip install ultralytics")
        return None

    try:
        print(
            f"\n=== Continuando entrenamiento desde {model_path} por {epochs} épocas adicionales ==="
        )

        # Cargar el modelo
        model = YOLO(model_path)

        # Continuar entrenamiento
        results = model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            device=device,
            resume=True,
        )

        # Obtener la ruta al mejor modelo
        if hasattr(results, "best") and os.path.exists(results.best):
            best_model = results.best
        else:
            # Buscar el mejor modelo
            model_dir = (
                results.save_dir
                if hasattr(results, "save_dir")
                else f"./runs/detect/train{epochs}"
            )
            best_model = os.path.join(model_dir, "weights/best.pt")

        if os.path.exists(best_model):
            print(
                f"\nEntrenamiento adicional completado. Mejor modelo guardado en: {best_model}"
            )
            return best_model
        else:
            print(f"\nNo se encontró el mejor modelo en la ruta esperada: {best_model}")
            return None

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        return None


def find_best_models():
    """
    Busca y lista los mejores modelos disponibles en el directorio runs/detect.
    Usa la API de Ultralytics para evaluar los modelos si está disponible.
    """
    model_dirs = glob.glob("./runs/detect/train*")
    model_dirs.extend(glob.glob("./runs/detect/continue_*"))

    models = []

    for model_dir in model_dirs:
        best_model = os.path.join(model_dir, "weights/best.pt")
        if os.path.exists(best_model):
            # Crear la información del modelo
            model_info = {
                "path": best_model,
                "name": os.path.basename(model_dir),
                "mAP50": "N/A",
                "mAP50-95": "N/A",
            }

            # Intentar evaluar el modelo para obtener métricas si YOLO está disponible
            if YOLO_AVAILABLE:
                try:
                    # Cargar modelo
                    model = YOLO(best_model)

                    # Evaluar brevemente para obtener métricas
                    metrics = model.val(verbose=False)
                    model_info["mAP50"] = f"{float(metrics.box.map50):.4f}"
                    model_info["mAP50-95"] = f"{float(metrics.box.map):.4f}"
                except Exception as e:
                    print(f"No se pudo evaluar el modelo {best_model}: {e}")
            else:
                print(
                    "⚠️ Ultralytics no está instalado. No se pueden evaluar los modelos automáticamente."
                )

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
        val_dir, test_dir, val_metrics, test_metrics = evaluate_model(
            model_path=args.model,
            data_path=args.data,
            batch_size=args.batch,
            image_size=args.imgsz,
            device=args.device,
        )

        # Crear un diccionario de métricas para análisis
        metrics = {"val": val_metrics, "test": test_metrics}

        # Preparar directorio único para todos los resultados (UN SOLO TIMESTAMP)
        model_dir = os.path.dirname(os.path.dirname(args.model))
        train_name = os.path.basename(model_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_base_dir = "./resultados"
        results_dir = os.path.join(results_base_dir, train_name, timestamp)
        os.makedirs(results_dir, exist_ok=True)

        # Generar gráficas comparativas y guardarlas solo en el directorio de resultados
        if metrics and len(metrics) > 0:
            plot_metrics_comparison(metrics, results_dir)
        else:
            print("\nNo hay suficientes datos para generar gráficas comparativas")

        # Visualizar predicciones y guardar en el mismo directorio de resultados
        vis_dir = visualize_predictions(
            model_path=args.model,
            data_path=args.data,
            num_samples=args.samples,
            image_size=args.imgsz,
            device=args.device,
            metrics=metrics,
            output_dir=results_dir,  # Usar el mismo directorio creado arriba
        )

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

        # Obtener la información más actualizada sobre directorios
        val_dirs = sorted(glob.glob(f"{val_dir}*"), key=os.path.getmtime)
        test_dirs = sorted(glob.glob(f"{test_dir}*"), key=os.path.getmtime)

        # Usar los directorios más recientes si existen
        recent_val_dir = val_dirs[-1] if val_dirs else val_dir
        recent_test_dir = test_dirs[-1] if test_dirs else test_dir

        print("\n=== PROCESO COMPLETO ===")
        print(f"Resultados de validación: {recent_val_dir}")
        print(f"Resultados de prueba: {recent_test_dir}")
        print(f"Visualizaciones y métricas: {results_dir}")

        # Resumen de métricas obtenidas
        if metrics and "val" in metrics and "test" in metrics:
            print("\n=== RESUMEN DE MÉTRICAS ===")
            print(
                f"Validación: mAP@0.5={metrics['val']['mAP50']:.4f}, mAP@0.5-0.95={metrics['val']['mAP50-95']:.4f}"
            )
            print(
                f"Prueba: mAP@0.5={metrics['test']['mAP50']:.4f}, mAP@0.5-0.95={metrics['test']['mAP50-95']:.4f}"
            )

            # Evaluación de calidad del modelo
            avg_map50 = (metrics["val"]["mAP50"] + metrics["test"]["mAP50"]) / 2
            avg_map50_95 = (
                metrics["val"]["mAP50-95"] + metrics["test"]["mAP50-95"]
            ) / 2

            if avg_map50 > 0.95 and avg_map50_95 > 0.7:
                print("\nCalidad del modelo: EXCELENTE")
                print("- Alta precisión en detección de placas")
                print("- Adecuado para implementación en producción")
            elif avg_map50 > 0.90 and avg_map50_95 > 0.65:
                print("\nCalidad del modelo: MUY BUENA")
                print("- Buen rendimiento general")
                print("- Puede considerar entrenamiento adicional para perfeccionar")
            elif avg_map50 > 0.85 and avg_map50_95 > 0.6:
                print("\nCalidad del modelo: BUENA")
                print("- Rendimiento aceptable")
                print("- Recomendado continuar entrenamiento para mejorar")
            else:
                print("\nCalidad del modelo: NECESITA MEJORA")
                print("- Considere entrenamiento adicional o ajuste de hiperparámetros")
                print("- Evalúe si el conjunto de datos necesita mejoras")

        print("\nPara comparar con otros modelos o continuar entrenando:")
        print(f"  python {os.path.basename(__file__)} --list-models")
        print(
            f"  python {os.path.basename(__file__)} --model [ruta_al_modelo] --data {args.data} --continue-epochs 20"
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
