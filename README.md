# Sistema de Reconocimiento de Placas de Matrícula

## **Descripción General**

Este proyecto presenta un **Sistema de Reconocimiento de Placas de Matrícula (LPR)** que utiliza **modelos YOLO** para la detección de objetos y **EasyOCR** para el reconocimiento de texto, diseñado para lograr una detección precisa y en tiempo real de placas de matrícula. Entrenado con un conjunto de datos híbrido y diverso de más de **52.000 imágenes etiquetadas**, el sistema demuestra un rendimiento sólido con alta precisión y recall, haciéndolo adecuado para aplicaciones como monitorización de tráfico, peajes automatizados y gestión de estacionamientos.

## **Características**

- **Detección de Objetos basada en YOLO**: Utiliza **modelos YOLO** para detectar placas de matrícula en tiempo real.
- **EasyOCR para Reconocimiento de Texto**: Extrae caracteres alfanuméricos de las placas de matrícula detectadas.
- **Conjunto de Datos Híbrido**: Combina tres conjuntos de datos diferentes que contienen una variedad de escenarios desafiantes como oclusiones, inconsistencias de iluminación y variaciones de pose.
- **Aumento de Datos**: Incluye transformaciones como volteo, recorte, rotación y ajustes de color para simular condiciones del mundo real y mejorar la robustez del modelo.

## **Conjunto de Datos**

El conjunto de datos consiste en imágenes procedentes de múltiples proyectos:

- **Conjunto de Datos 1**: 21.175 imágenes de un proyecto de visión por computador de reconocimiento de placas de matrícula.
- **Conjunto de Datos 2**: 6.784 imágenes del proyecto de placas de matrícula de vehículos en Turquía.
- **Conjunto de Datos 3**: 24.242 imágenes de un proyecto de placas de matrícula de vehículos.

Este conjunto de datos híbrido contiene un total de **52.201 imágenes etiquetadas**, divididas en conjuntos de entrenamiento, validación y prueba:

- **Entrenamiento**: 46.278 imágenes (87%)
- **Validación**: 3.957 imágenes (8%)
- **Prueba**: 1.966 imágenes (4%)

### **Preprocesamiento**

- **Redimensionamiento**: Todas las imágenes fueron redimensionadas a **640x640** píxeles.
- **Aumento de Datos**: Incluye volteo horizontal, rotación, recorte, conversión a escala de grises y ajustes en brillo, tono, saturación y contraste.

## **Arquitectura del Modelo**

Evaluamos varios **modelos YOLO**, incluyendo:

- **YOLOv8** y **YOLOv11** (con varias configuraciones: **n**, **s**, **m**, **l**, **x**).
- **EasyOCR** para el reconocimiento óptico de caracteres de las placas de matrícula detectadas.

### **Hallazgos Clave**

- **YOLOv11x** logró la mayor Precisión media (**mAP**) de **0.98466 en IoU 50%** y **0.71605 en IoU 50-95%** después de 20 épocas.
- **YOLOv11n**, un modelo ligero, logró un alto rendimiento con valores similares de precisión y recall después de 100 épocas.
- El modelo **YOLOv11m** logró el mejor equilibrio entre rendimiento y precisión, con el **mAP@50-95** más alto de **0.71743**.

## **Evaluación de Rendimiento**

El sistema fue evaluado utilizando **Precisión**, **Recall** y **Precisión media (mAP)** tanto en **IoU 50%** como en **IoU 50-95%**. Los resultados mostraron mejoras en recall y mAP con entrenamiento extendido, particularmente en modelos **YOLOv8** y **YOLOv11** entrenados durante **100 épocas**.

### **Resultados de Ejemplo:**

- **YOLOv8** (10 épocas) logró:

  - **Precisión (P)**: 0.974
  - **Recall (R)**: 0.954
  - **mAP@50**: 0.978
  - **mAP@50-95**: 0.682

- **YOLOv11** (10 épocas) logró:
  - **Precisión (P)**: 0.981
  - **Recall (R)**: 0.951
  - **mAP@50**: 0.981
  - **mAP@50-95**: 0.682

El entrenamiento durante **100 épocas** mejoró significativamente las capacidades de detección, aumentando el **mAP@50-95** hasta en un **4.25%**.

## **Estudio de Ablación**

Se realizó un **Estudio de Ablación** para evaluar el rendimiento de diferentes variantes de modelos **YOLOv11** entrenados en el conjunto de datos híbrido. El estudio tenía como objetivo evaluar las compensaciones entre la precisión del modelo, el recall y la eficiencia computacional.

### **Variantes del Modelo YOLOv11**

Se evaluaron las siguientes variantes de **YOLOv11**:

- **YOLOv11n** (ligero)
- **YOLOv11s**
- **YOLOv11m**
- **YOLOv11l**
- **YOLOv11x** (modelo más grande)

### **Comparación de Rendimiento**

Los resultados del estudio de ablación se resumen en la tabla a continuación:

| **Modelo**   | **Precisión (B)** | **Recall (B)** | **mAP@50 (B)** | **mAP@50-95 (B)** |
| ------------ | ----------------- | -------------- | -------------- | ----------------- |
| **YOLOv11x** | 0.97384           | 0.95932        | 0.98387        | 0.71605           |
| **YOLOv11l** | 0.96733           | 0.96591        | 0.98497        | 0.71454           |
| **YOLOv11m** | 0.97035           | 0.96761        | 0.98582        | 0.71743           |
| **YOLOv11s** | 0.97312           | 0.96170        | 0.98477        | 0.71344           |
| **YOLOv11n** | 0.97125           | 0.95849        | 0.98235        | 0.71064           |

### **Hallazgos Clave**

- **YOLOv11m** logró el **mAP@50-95** más alto (**0.71743**), convirtiéndolo en la mejor opción para aplicaciones que requieren alta precisión.
- **YOLOv11x** logró la **Precisión (B)** más alta (**0.97384**), adecuado para aplicaciones que priorizan la precisión de detección.
- **YOLOv11l** logró el **Recall (B)** más alto (**0.96591**), indicando su capacidad para minimizar detecciones perdidas.
- **YOLOv11n** y **YOLOv11s** son modelos ligeros que equilibran la eficiencia computacional con métricas de detección satisfactorias, haciéndolos ideales para implementación en dispositivos edge.

Estos resultados proporcionan información sobre las compensaciones entre precisión, recall y requisitos computacionales para cada variante, permitiendo la selección del modelo adaptado a necesidades específicas de la aplicación.

## **Guía Rápida para el Entrenamiento**

Esta sección proporciona una guía simplificada para entrenar tu propio modelo de detección de placas de matrícula. Para instrucciones detalladas, ejemplos y solución de problemas, consulta el archivo [README_TRAINING.md](README_TRAINING.md).

### Importante: Uso de Rutas Absolutas

Al ejecutar comandos de entrenamiento o evaluación, debes especificar rutas absolutas para los archivos de datos. Esto es porque YOLOv busca los archivos relativos a su directorio de conjuntos de datos configurado, no al directorio de trabajo actual.

```bash
# Usa $(pwd) para obtener la ruta absoluta al directorio actual
python trainYolov11s.py --epochs 20 --batch 16 --device 0 --data $(pwd)/data/data.yaml

# Lo mismo aplica para la evaluación
python evaluateModel.py --model runs/detect/train/weights/best.pt --data $(pwd)/data/data.yaml
```

Esta solución evita el error "Dataset images not found" que ocurre cuando se utilizan rutas relativas.

### Nota sobre Directorios de Entrenamiento Incrementales

YOLO crea directorios con nombres incrementales para cada nueva ejecución de entrenamiento (train, train2, train3, etc.) para evitar sobrescribir resultados anteriores. Los scripts del sistema están actualizados para detectar automáticamente estos directorios incrementales, por lo que siempre buscarán en el directorio más reciente.

### Visión General del Proceso de Entrenamiento

1. **Combinar conjuntos de datos**:

   ```bash
   python combineDatasets.py
   ```

2. **Entrenar el modelo**:

   ```bash
   python trainYolov11s.py --epochs 20 --batch 16 --device 0 --data $(pwd)/data/data.yaml
   ```

3. **Evaluar el modelo**:

   ```bash
   python evaluateModel.py --model runs/detect/train/weights/best.pt --data $(pwd)/data/data.yaml
   ```

4. **Continuar el entrenamiento (si es necesario)**:

   ```bash
   python evaluateModel.py --model runs/detect/train/weights/best.pt --data $(pwd)/data/data.yaml --continue-epochs 20
   ```

5. **Comparar modelos**:
   ```bash
   python evaluateModel.py --list-models
   ```

## **Estructura de la Documentación**

Este proyecto incluye dos archivos README:

1. **README.md** (este archivo): Proporciona una visión general del proyecto, características, información del conjunto de datos, arquitectura del modelo, resultados de rendimiento y una guía rápida para el entrenamiento.

2. **[README_TRAINING.md](README_TRAINING.md)**: Contiene instrucciones detalladas para el proceso completo de entrenamiento, incluyendo:
   - Guía paso a paso para combinar conjuntos de datos, entrenamiento y evaluación
   - Explicaciones en profundidad de los parámetros de entrenamiento y sus efectos
   - Información detallada sobre cómo interpretar métricas y visualizaciones del modelo
   - Ejemplos de modelos buenos, sobreajustados y subajustados con acciones sugeridas
   - Un flujo de trabajo iterativo para obtener el mejor modelo posible
   - Solución de problemas comunes durante el entrenamiento

Si estás buscando comprender la arquitectura y resultados del proyecto, este README proporciona toda la información esencial. Si quieres entrenar tus propios modelos o mejorar los existentes, consulta el [README_TRAINING.md](README_TRAINING.md) para instrucciones completas.
