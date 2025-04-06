#!/usr/bin/env python
"""
Flujo de trabajo completo para detección de placas de matrícula con YOLOv8/YOLOv11
Este script permite ejecutar el proceso completo desde preparación de datos hasta evaluación.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path


# Colores para terminal
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_step(step, description):
    """Imprime un paso del flujo de trabajo con formato"""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}PASO {step}: {description}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")


def run_command(cmd, description=None):
    """Ejecuta un comando y muestra el resultado"""
    if description:
        print(f"{Colors.YELLOW}{description}{Colors.ENDC}")

    print(f"{Colors.GREEN}Ejecutando: {' '.join(cmd)}{Colors.ENDC}")

    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed_time = time.time() - start_time

        print(
            f"{Colors.GREEN}Comando ejecutado correctamente en {elapsed_time:.2f} segundos{Colors.ENDC}"
        )
        print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Error al ejecutar el comando:{Colors.ENDC}")
        print(f"{Colors.RED}{e}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Salida del comando:{Colors.ENDC}")
        print(e.stdout)
        print(f"{Colors.RED}Error del comando:{Colors.ENDC}")
        print(e.stderr)
        return False, None


def prepare_local_data(import_from=None, output_dir="data"):
    """Prepara datos locales para evaluación"""
    print_step(1, "Preparación de datos locales")

    cmd = ["python", "prepareLocalData.py", "--output", output_dir]
    if import_from:
        cmd.extend(["--import-from", import_from])

    description = "Creando estructura mínima de datos local"
    if import_from:
        description = f"Importando datos desde {import_from}"

    return run_command(cmd, description)


def combine_datasets(output_dir="data"):
    """Combina los datasets de placas de matrícula"""
    print_step(2, "Combinación de datasets")

    cmd = ["python", "combineDatasets.py"]
    description = "Combinando datasets en una estructura unificada"

    return run_command(cmd, description)


def train_model(epochs=20, batch=16, device="0", data_path="data/data.yaml", imgsz=640):
    """Entrena el modelo YOLOv11s"""
    print_step(3, "Entrenamiento del modelo")

    if not os.path.exists(data_path):
        print(
            f"{Colors.RED}Error: No se encontró el archivo de datos en {data_path}{Colors.ENDC}"
        )
        print(
            f"{Colors.YELLOW}Ejecute primero 'prepare_local_data' o 'combine_datasets'{Colors.ENDC}"
        )
        return False, None

    # Obtener la ruta absoluta para data.yaml
    abs_data_path = os.path.abspath(data_path)

    cmd = [
        "python",
        "trainYolov11s.py",
        "--epochs",
        str(epochs),
        "--batch",
        str(batch),
        "--device",
        device,
        "--data",
        abs_data_path,
        "--imgsz",
        str(imgsz),
    ]

    description = (
        f"Entrenando YOLOv11s por {epochs} épocas con batch={batch} en device={device}"
    )

    return run_command(cmd, description)


def evaluate_model(
    model_path, data_path="data/data.yaml", device="0", samples=10, continue_epochs=0
):
    """Evalúa un modelo entrenado"""
    print_step(4, "Evaluación del modelo")

    if not os.path.exists(model_path):
        print(
            f"{Colors.RED}Error: No se encontró el modelo en {model_path}{Colors.ENDC}"
        )
        print(
            f"{Colors.YELLOW}Asegúrese de que el entrenamiento se completó correctamente{Colors.ENDC}"
        )
        return False, None

    if not os.path.exists(data_path):
        print(
            f"{Colors.RED}Error: No se encontró el archivo de datos en {data_path}{Colors.ENDC}"
        )
        print(
            f"{Colors.YELLOW}Ejecute primero 'prepare_local_data' o 'combine_datasets'{Colors.ENDC}"
        )
        return False, None

    # Obtener la ruta absoluta para data.yaml
    abs_data_path = os.path.abspath(data_path)

    cmd = [
        "python",
        "evaluateModel.py",
        "--model",
        model_path,
        "--data",
        abs_data_path,
        "--device",
        device,
        "--samples",
        str(samples),
    ]

    if continue_epochs > 0:
        cmd.extend(["--continue-epochs", str(continue_epochs)])

    description = f"Evaluando el modelo {model_path} con {samples} muestras"
    if continue_epochs > 0:
        description += f" y continuando el entrenamiento por {continue_epochs} épocas"

    return run_command(cmd, description)


def list_models():
    """Lista los modelos disponibles"""
    print_step(5, "Listado de modelos disponibles")

    cmd = ["python", "evaluateModel.py", "--list-models"]
    description = "Listando y comparando todos los modelos entrenados"

    return run_command(cmd, description)


def complete_workflow(
    use_local=False, import_from=None, epochs=20, batch=16, device="0", samples=10
):
    """Ejecuta el flujo de trabajo completo desde la preparación de datos hasta la evaluación"""
    print(
        f"{Colors.BOLD}{Colors.HEADER}FLUJO DE TRABAJO COMPLETO PARA DETECCIÓN DE PLACAS DE MATRÍCULA{Colors.ENDC}"
    )
    print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}")

    # Paso 1: Preparación de datos
    if use_local:
        success, _ = prepare_local_data(import_from)
    else:
        success, _ = combine_datasets()

    if not success:
        print(
            f"{Colors.RED}Error en la preparación de datos. Abortando flujo de trabajo.{Colors.ENDC}"
        )
        return False

    # Paso 2: Entrenamiento
    success, _ = train_model(epochs=epochs, batch=batch, device=device)

    if not success:
        print(
            f"{Colors.RED}Error en el entrenamiento. Abortando flujo de trabajo.{Colors.ENDC}"
        )
        return False

    # Paso 3: Encontrar el último modelo entrenado
    train_dirs = sorted(Path("runs/detect").glob("train*"))
    if not train_dirs:
        print(
            f"{Colors.RED}No se encontraron directorios de entrenamiento. Abortando flujo de trabajo.{Colors.ENDC}"
        )
        return False

    latest_train_dir = str(train_dirs[-1])
    model_path = os.path.join(latest_train_dir, "weights/best.pt")

    if not os.path.exists(model_path):
        print(
            f"{Colors.RED}No se encontró el modelo en {model_path}. Abortando flujo de trabajo.{Colors.ENDC}"
        )
        return False

    # Paso 4: Evaluación
    success, _ = evaluate_model(model_path=model_path, device=device, samples=samples)

    if not success:
        print(
            f"{Colors.RED}Error en la evaluación. El flujo de trabajo puede estar incompleto.{Colors.ENDC}"
        )

    # Paso 5: Listar modelos
    list_models()

    print(f"\n{Colors.BOLD}{Colors.GREEN}FLUJO DE TRABAJO COMPLETADO{Colors.ENDC}")
    print(f"{Colors.YELLOW}Próximos pasos recomendados:{Colors.ENDC}")
    print(
        f"  1. Revise los resultados de evaluación en las carpetas 'runs/detect/val' y 'runs/detect/test_results'"
    )
    print(f"  2. Examine las visualizaciones en 'predictions_best/samples'")
    print(f"  3. Considere continuar el entrenamiento:")
    print(
        f"     python licensePlateWorkflow.py --continue-training {model_path} --continue-epochs 20"
    )

    return True


def continue_training_workflow(model_path, continue_epochs=20, device="0", samples=10):
    """Continúa el entrenamiento de un modelo existente y lo evalúa"""
    print(
        f"{Colors.BOLD}{Colors.HEADER}CONTINUACIÓN DE ENTRENAMIENTO Y EVALUACIÓN{Colors.ENDC}"
    )
    print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}")

    # Verificar que el modelo existe
    if not os.path.exists(model_path):
        print(
            f"{Colors.RED}Error: No se encontró el modelo en {model_path}{Colors.ENDC}"
        )
        return False

    # Paso 1: Continuar entrenamiento y evaluar
    success, _ = evaluate_model(
        model_path=model_path,
        device=device,
        samples=samples,
        continue_epochs=continue_epochs,
    )

    if not success:
        print(
            f"{Colors.RED}Error en la continuación del entrenamiento. El proceso puede estar incompleto.{Colors.ENDC}"
        )

    # Paso 2: Listar modelos
    list_models()

    print(
        f"\n{Colors.BOLD}{Colors.GREEN}PROCESO DE CONTINUACIÓN COMPLETADO{Colors.ENDC}"
    )
    print(f"{Colors.YELLOW}Próximos pasos recomendados:{Colors.ENDC}")
    print(f"  1. Compare el rendimiento del modelo mejorado con el original")
    print(f"  2. Considere ejecutar más iteraciones de entrenamiento si es necesario")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flujo de trabajo completo para detección de placas de matrícula con YOLOv11"
    )

    # Subparsers para diferentes comandos
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # Parser para preparación de datos local
    prepare_parser = subparsers.add_parser(
        "prepare-local", help="Preparar datos locales para evaluación"
    )
    prepare_parser.add_argument(
        "--import-from", type=str, help="Importar datos desde un directorio existente"
    )
    prepare_parser.add_argument(
        "--output", type=str, default="data", help="Directorio de salida para los datos"
    )

    # Parser para combinar datasets
    combine_parser = subparsers.add_parser("combine-datasets", help="Combinar datasets")

    # Parser para entrenamiento
    train_parser = subparsers.add_parser("train", help="Entrenar modelo YOLOv11s")
    train_parser.add_argument(
        "--epochs", type=int, default=20, help="Número de épocas de entrenamiento"
    )
    train_parser.add_argument("--batch", type=int, default=16, help="Tamaño del batch")
    train_parser.add_argument(
        "--device", type=str, default="0", help="Dispositivo (0, 1, 2, etc. o 'cpu')"
    )
    train_parser.add_argument(
        "--data", type=str, default="data/data.yaml", help="Ruta al archivo data.yaml"
    )
    train_parser.add_argument(
        "--imgsz", type=int, default=640, help="Tamaño de las imágenes"
    )

    # Parser para evaluación
    eval_parser = subparsers.add_parser("evaluate", help="Evaluar modelo entrenado")
    eval_parser.add_argument(
        "--model", type=str, required=True, help="Ruta al modelo .pt"
    )
    eval_parser.add_argument(
        "--data", type=str, default="data/data.yaml", help="Ruta al archivo data.yaml"
    )
    eval_parser.add_argument(
        "--device", type=str, default="0", help="Dispositivo (0, 1, 2, etc. o 'cpu')"
    )
    eval_parser.add_argument(
        "--samples", type=int, default=10, help="Número de muestras para visualización"
    )
    eval_parser.add_argument(
        "--continue-epochs",
        type=int,
        default=0,
        help="Continuar entrenamiento con este número de épocas (0 para omitir)",
    )

    # Parser para listar modelos
    list_parser = subparsers.add_parser(
        "list-models", help="Listar modelos disponibles"
    )

    # Parser para flujo de trabajo completo
    workflow_parser = subparsers.add_parser(
        "workflow", help="Ejecutar flujo de trabajo completo"
    )
    workflow_parser.add_argument(
        "--local",
        action="store_true",
        help="Usar preparación de datos local en lugar de combinación de datasets",
    )
    workflow_parser.add_argument(
        "--import-from",
        type=str,
        help="Importar datos desde un directorio existente (solo con --local)",
    )
    workflow_parser.add_argument(
        "--epochs", type=int, default=20, help="Número de épocas de entrenamiento"
    )
    workflow_parser.add_argument(
        "--batch", type=int, default=16, help="Tamaño del batch"
    )
    workflow_parser.add_argument(
        "--device", type=str, default="0", help="Dispositivo (0, 1, 2, etc. o 'cpu')"
    )
    workflow_parser.add_argument(
        "--samples", type=int, default=10, help="Número de muestras para visualización"
    )

    # Parser para continuación de entrenamiento
    continue_parser = subparsers.add_parser(
        "continue-training", help="Continuar entrenamiento de un modelo existente"
    )
    continue_parser.add_argument(
        "model_path", type=str, help="Ruta al modelo .pt para continuar"
    )
    continue_parser.add_argument(
        "--continue-epochs", type=int, default=20, help="Número de épocas adicionales"
    )
    continue_parser.add_argument(
        "--device", type=str, default="0", help="Dispositivo (0, 1, 2, etc. o 'cpu')"
    )
    continue_parser.add_argument(
        "--samples", type=int, default=10, help="Número de muestras para visualización"
    )

    args = parser.parse_args()

    # Ejecutar el comando correspondiente
    if args.command == "prepare-local":
        prepare_local_data(args.import_from, args.output)
    elif args.command == "combine-datasets":
        combine_datasets()
    elif args.command == "train":
        train_model(args.epochs, args.batch, args.device, args.data, args.imgsz)
    elif args.command == "evaluate":
        evaluate_model(
            args.model, args.data, args.device, args.samples, args.continue_epochs
        )
    elif args.command == "list-models":
        list_models()
    elif args.command == "workflow":
        complete_workflow(
            args.local,
            args.import_from,
            args.epochs,
            args.batch,
            args.device,
            args.samples,
        )
    elif args.command == "continue-training":
        continue_training_workflow(
            args.model_path, args.continue_epochs, args.device, args.samples
        )
    else:
        parser.print_help()
        sys.exit(1)
