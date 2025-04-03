# Entrenamiento de YOLOv11s para Detección de Placas de Matrícula

Este proyecto contiene scripts para combinar tres datasets diferentes de placas de matrícula y entrenar un modelo YOLOv11s para detectarlas.

## Estructura del Proyecto

- **combineDatasets.py**: Script para combinar los tres datasets en una estructura unificada
- **trainYolov11s.py**: Script para entrenar el modelo YOLOv11s con el dataset combinado
- **evaluateModel.py**: Script para evaluar, visualizar resultados y continuar la iteración de modelos
- **data/**: Directorio donde se almacenará el dataset combinado (generado automáticamente)
- **runs/**: Directorio donde se guardarán los resultados del entrenamiento (generado automáticamente)

## Datasets Utilizados

El proyecto utiliza tres datasets diferentes para el entrenamiento:

1. License-Plate-Recognition-4 (21,175 imágenes)
2. License-Plates-of-Vehicles-in-Turkey-150 (6,784 imágenes)
3. Vehicle-Registration-Plates-2 (24,242 imágenes)

Estos datasets se encuentran en el directorio `content/`.

## Requisitos Previos

1. Python 3.8 o superior
2. Ultralytics YOLO (v8.2.70 o superior)
3. PyTorch
4. Otros paquetes: PyYAML, matplotlib, shutil, argparse

Para instalar los requisitos, puedes ejecutar:

```bash
pip install ultralytics pyyaml matplotlib
```

## Guía Completa del Proceso Iterativo

### Paso 1: Combinar los Datasets

El primer paso es combinar los tres datasets en una estructura unificada:

```bash
python combineDatasets.py
```

**Resultado esperado:**

```
Iniciando la combinación de datasets...
Procesando dataset: License-Plate-Recognition-4
Procesando dataset: License-Plates-of-Vehicles-in-Turkey-150
Procesando dataset: Vehicle-Registration-Plates-2

Resumen de imágenes copiadas:
- Train: 46278 imágenes
- Validation: 3957 imágenes
- Test: 1966 imágenes
Total: 52201 imágenes

Se ha creado el archivo data.yaml en data/data.yaml

Proceso completado. Ahora puedes entrenar YOLOv11s con el dataset combinado.
```

Este proceso creará una estructura como esta:

```

 data/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml
```

El archivo `data.yaml` generado contendrá:

```yaml
train: ./train/images
val: ./valid/images
test: ./test/images
nc: 1
names:
  - license_plate
```

### Paso 2: Entrenar el Modelo Inicial

Una vez combinados los datasets, podemos entrenar el modelo YOLOv11s inicial:

```bash
python trainYolov11s.py --epochs 20 --batch 16 --device 0
```

Este comando:

- Entrena un modelo YOLOv11s por 20 épocas
- Usa un tamaño de batch de 16
- Utiliza la primera GPU (device 0)

**Resultado esperado:**

```
Iniciando entrenamiento de YOLOv11s para detección de placas de matrícula
Archivo de datos: data/data.yaml
Épocas: 20
Batch size: 16
Tamaño de imagen: 640x640
Dispositivo: 0
--------------------------------------------------
Ejecutando: yolo task=detect mode=train model=yolo11s.pt data=data/data.yaml device=0 save=True epochs=20 batch=16 val=True plots=True imgsz=640

[YOLO] Ultralytics YOLOv8.2.70 🚀 Python-3.8.10 torch-2.0.1+cu118
[YOLO] [1/20] Scanning 'data/train/labels'... 46278 images, 0 corrupt: 100% 46278/46278 [00:03<00:00]
[YOLO] [1/20] Training: 100% 2893/2893 [05:51<00:00]
[YOLO] [1/20] Validation: 100% 248/248 [00:47<00:00]
[YOLO] Epoch 1/20 completed (mAP@0.5 = 0.837, mAP@0.5:0.95 = 0.603)
...
[YOLO] [20/20] Training: 100% 2893/2893 [05:51<00:00]
[YOLO] [20/20] Validation: 100% 248/248 [00:47<00:00]
[YOLO] Epoch 20/20 completed (mAP@0.5 = 0.957, mAP@0.5:0.95 = 0.689)

Entrenamiento completado. Mejor modelo guardado en: ./runs/detect/train/weights/best.pt

Evaluando modelo: yolo task=detect mode=val model=./runs/detect/train/weights/best.pt data=data/data.yaml save_json=True plots=True

Proceso completo. Revisa los resultados en la carpeta 'runs/detect'.
```

Durante el entrenamiento, podrás observar:

- El progreso por época
- Las métricas de rendimiento (mAP, precisión, recall)
- La pérdida de entrenamiento

### Paso 3: Evaluar el Modelo en Detalle

Para una evaluación más detallada:

```bash
python evaluateModel.py --model runs/detect/train/weights/best.pt
```

Este comando:

- Evalúa el modelo en los conjuntos de validación y prueba
- Genera visualizaciones de predicciones
- Analiza las métricas en detalle
- Crea gráficas comparativas

**Resultado esperado:**

```
=== EVALUACIÓN Y ANÁLISIS DE MODELO ===
Modelo: runs/detect/train/weights/best.pt
Archivo de datos: data/data.yaml
--------------------------------------------------

=== Evaluando best en el conjunto de VALIDACIÓN ===
Ejecutando: yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data/data.yaml batch=16 imgsz=640 device=0 save_json=True save_txt=True save_conf=True plots=True

=== Evaluando best en el conjunto de PRUEBA ===
Ejecutando: yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data/data.yaml split=test batch=16 imgsz=640 device=0 save_json=True save_txt=True save_conf=True name=test_results plots=True

=== Generando visualizaciones de predicciones para 10 imágenes ===
Ejecutando: yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=img1.jpg,img2.jpg,... imgsz=640 device=0 save=True save_txt=True save_conf=True project=./predictions_best name=samples

=== ANÁLISIS DE RESULTADOS ===

Resultados en conjunto de VALIDACIÓN:
- mAP@0.5: 0.9576
- mAP@0.5-0.95: 0.6890
- Precision: 0.9423
- Recall: 0.9312

Resultados en conjunto de PRUEBA:
- mAP@0.5: 0.9443
- mAP@0.5-0.95: 0.6735
- Precision: 0.9351
- Recall: 0.9148

Gráfica comparativa guardada como: metrics_comparison.png

=== PROCESO COMPLETO ===
Resultados de validación: ./runs/detect/val
Resultados de prueba: ./runs/detect/test_results
Visualizaciones: ./predictions_best/samples
```

### Paso 4: Interpretar los Resultados

#### Métricas Clave

1. **mAP@0.5**:

   - Mean Average Precision con un umbral de IoU (Intersección sobre Unión) de 0.5
   - Valor óptimo: 1.0 (cuanto más cercano a 1, mejor)
   - Interpretación: Un valor alto (>0.9) indica que el modelo detecta correctamente las placas de matrícula con un buen solapamiento

2. **mAP@0.5-0.95**:

   - Media de mAP calculada en diferentes umbrales de IoU (0.5 a 0.95)
   - Más riguroso que mAP@0.5
   - Un modelo con mAP@0.5-0.95 de 0.67-0.70 es considerado bueno para detección de placas

3. **Precision**:

   - Proporción de detecciones correctas entre todas las detecciones hechas
   - Mide si el modelo hace predicciones precisas
   - Un valor >0.93 es excelente para este tipo de tarea

4. **Recall**:
   - Proporción de objetos correctamente detectados del total de objetos reales
   - Mide si el modelo encuentra todas las placas
   - Un valor >0.90 indica que el modelo rara vez pierde detección de placas

#### Analizando Gráficas

La gráfica `metrics_comparison.png` muestra la comparación de métricas entre validación y prueba:

- Si las barras son similares entre val/test, el modelo generaliza bien
- Si el rendimiento en prueba es significativamente menor, puede indicar sobreajuste

En el directorio de resultados (`runs/detect/train/`), encontrarás:

1. **Curva de Precisión-Recall (`PR_curve.png`)**:

   - Muestra la relación entre precisión y recall para diferentes umbrales
   - Una curva más alta indica mejor rendimiento
   - El área bajo la curva (PR AUC) debe ser alta

2. **Matriz de Confusión (`confusion_matrix.png`)**:

   - Muestra falsos positivos, falsos negativos, verdaderos positivos
   - Idealmente, la mayoría de las predicciones deben estar en la diagonal

3. **Gráfica de Pérdida (`results.png`)**:
   - Muestra la evolución de las pérdidas durante el entrenamiento
   - La tendencia debe ser decreciente

#### Visualizaciones

En el directorio `predictions_best/samples` encontrarás:

- Imágenes con las detecciones dibujadas (cuadros delimitadores)
- Archivos de texto con las coordenadas y confianza de cada detección

Esto te permite evaluar cualitativamente el rendimiento del modelo:

- ¿Está detectando correctamente todas las placas?
- ¿Hay placas que se pierden (falsos negativos)?
- ¿Hay detecciones incorrectas (falsos positivos)?

### Paso 5: Continuar el Entrenamiento

Si los resultados no son óptimos, puedes continuar entrenando desde el modelo actual:

```bash
python evaluateModel.py --model runs/detect/train/weights/best.pt --continue-epochs 20
```

Este comando:

- Evalúa el modelo actual
- Continúa entrenando por 20 épocas más
- Guarda el nuevo modelo mejorado

**Resultado esperado:**

```
=== Continuando entrenamiento desde runs/detect/train/weights/best.pt por 20 épocas adicionales ===
Ejecutando: yolo task=detect mode=train model=runs/detect/train/weights/best.pt data=data/data.yaml device=0 save=True epochs=20 batch=16 val=True plots=True imgsz=640 name=continue_best

[YOLO] Ultralytics YOLOv8.2.70 🚀 Python-3.8.10 torch-2.0.1+cu118
[YOLO] [1/20] Training: 100% 2893/2893 [05:51<00:00]
[YOLO] [1/20] Validation: 100% 248/248 [00:47<00:00]
...
[YOLO] [20/20] Training: 100% 2893/2893 [05:51<00:00]
[YOLO] [20/20] Validation: 100% 248/248 [00:47<00:00]
[YOLO] Epoch 20/20 completed (mAP@0.5 = 0.982, mAP@0.5:0.95 = 0.714)

Entrenamiento adicional completado. Mejor modelo guardado en: ./runs/detect/continue_best/weights/best.pt

Evaluar el modelo mejorado:
python evaluateModel.py --model ./runs/detect/continue_best/weights/best.pt --data data/data.yaml
```

### Paso 6: Encontrar el Mejor Modelo

Para listar y comparar todos los modelos disponibles:

```bash
python evaluateModel.py --list-models
```

**Resultado esperado:**

```
=== MODELOS DISPONIBLES ===
Modelo               mAP@0.5    mAP@0.5-0.95  Ruta
--------------------------------------------------------------------------------
continue_best        0.9824     0.7143        ./runs/detect/continue_best/weights/best.pt
train                0.9576     0.6890        ./runs/detect/train/weights/best.pt
```

Aquí puedes ver que el modelo continuado (continue_best) tiene un mejor rendimiento que el inicial (train).

### Paso 7: Evaluar el Mejor Modelo

Una vez identificado el mejor modelo, podemos evaluarlo a fondo:

```bash
python evaluateModel.py --model ./runs/detect/continue_best/weights/best.pt --samples 20
```

Este comando:

- Evalúa exhaustivamente el mejor modelo
- Genera 20 visualizaciones de muestra (en lugar de 10 predeterminadas)

## Interpretar Resultados: Casos de Ejemplo

### Ejemplo 1: Modelo con Buen Rendimiento

```
Resultados en conjunto de VALIDACIÓN:
- mAP@0.5: 0.9824
- mAP@0.5-0.95: 0.7143
- Precision: 0.9623
- Recall: 0.9512

Resultados en conjunto de PRUEBA:
- mAP@0.5: 0.9743
- mAP@0.5-0.95: 0.7025
- Precision: 0.9542
- Recall: 0.9417
```

**Interpretación:**

- El modelo tiene excelente rendimiento (mAP > 0.97)
- Generaliza bien (resultados similares en validación y prueba)
- Alta precisión y recall (raramente comete errores)
- Probablemente listo para implementación

### Ejemplo 2: Modelo con Sobreajuste

```
Resultados en conjunto de VALIDACIÓN:
- mAP@0.5: 0.9856
- mAP@0.5-0.95: 0.7243
- Precision: 0.9723
- Recall: 0.9612

Resultados en conjunto de PRUEBA:
- mAP@0.5: 0.9123
- mAP@0.5-0.95: 0.6425
- Precision: 0.9142
- Recall: 0.8917
```

**Interpretación:**

- El modelo rinde muy bien en validación pero peor en prueba
- Indica sobreajuste (no generaliza bien)
- Posibles soluciones:
  - Aumentar la regularización (dropout)
  - Usar más datos de entrenamiento
  - Aplicar técnicas de aumento de datos más agresivas

### Ejemplo 3: Modelo con Precisión Alta pero Recall Bajo

```
Resultados en conjunto de PRUEBA:
- mAP@0.5: 0.9324
- mAP@0.5-0.95: 0.6734
- Precision: 0.9723
- Recall: 0.8512
```

**Interpretación:**

- El modelo es muy preciso cuando detecta una placa (pocos falsos positivos)
- Pero pierde varias placas (falsos negativos altos)
- Útil cuando es crítico evitar detecciones incorrectas
- Mejorar aumentando el umbral de confianza

## Flujo de Trabajo Iterativo Recomendado

Para obtener el mejor modelo posible, se recomienda este flujo de trabajo iterativo:

1. **Preparación de datos**:

   ```bash
   python combineDatasets.py
   ```

2. **Entrenamiento inicial**:

   ```bash
   python trainYolov11s.py --epochs 20
   ```

3. **Evaluación detallada**:

   ```bash
   python evaluateModel.py --model runs/detect/train/weights/best.pt
   ```

4. **Análisis de resultados**:

   - Examinar métricas (mAP, precision, recall)
   - Revisar visualizaciones
   - Identificar problemas (sobreajuste, subajuste)

5. **Iteración**:

   - Si mAP@0.5 < 0.94 o mAP@0.5-0.95 < 0.68:
     ```bash
     python evaluateModel.py --model runs/detect/train/weights/best.pt --continue-epochs 20
     ```
   - Si hay sobreajuste (diferencia grande entre validación y prueba):
     ```bash
     # Ajustar hiperparámetros como batch y learning rate
     python trainYolov11s.py --epochs 30 --batch 32 --imgsz 640
     ```
   - Si hay subajuste (rendimiento bajo en validación y prueba):
     ```bash
     # Entrenar más épocas
     python trainYolov11s.py --epochs 50 --batch 16
     ```

6. **Selección del mejor modelo**:

   ```bash
   python evaluateModel.py --list-models
   ```

7. **Evaluación final**:
   ```bash
   python evaluateModel.py --model <ruta_al_mejor_modelo> --samples 30
   ```

## Notas Adicionales

- **Tiempo de Entrenamiento**: El entrenamiento inicial (20 épocas) toma aproximadamente 2-3 horas en una GPU moderna (NVIDIA RTX 3080 o similar)
- **Rendimiento Esperado**: Un modelo YOLOv11s bien entrenado debe alcanzar:
  - mAP@0.5 > 0.95
  - mAP@0.5-0.95 > 0.70
  - Precision > 0.94
  - Recall > 0.93
- **Optimización de Memoria**: Si experimentas problemas de memoria en GPU:
  - Reduce el tamaño del batch (`--batch 8` o menor)
  - Usa un modelo más ligero (YOLOv11n en lugar de YOLOv11s)
- **Solución de Problemas**:
  - Si las métricas no mejoran después de 20-30 épocas, prueba ajustar la tasa de aprendizaje
  - Si hay muchos falsos positivos, verifica la calidad de las anotaciones
  - Si hay muchos falsos negativos, asegúrate de que el dataset incluye suficientes variaciones de placas
