# Documentación de Entrenamiento de Modelo de Reconocimiento de Caracteres (Kanji)

Este documento describe el flujo de trabajo, la arquitectura y los resultados del experimento de entrenamiento registrado en el cuaderno adjunto.

## Descripción General
El cuaderno `kanji_classificator_model_training.ipynb` (antes `train.ipynb`) implementa un ciclo completo de aprendizaje profundo (Deep Learning) para la clasificación de caracteres manuscritos. El flujo ha sido actualizado para utilizar el dataset **ETL9B** (binarizado) y abarca desde la carga de datos hasta la inferencia con imágenes externas, incorporando umbralizado Otsu.

## Etapas del Cuaderno

El cuaderno se estructura en las siguientes secciones lógicas:

1.  **Configuración y Constantes**:
    * Se establecen las rutas a los directorios de datos y salida.
    * Se definen los parámetros globales que gobernarán el entrenamiento (hiperparámetros).
    * **Renombrado**: El cuaderno principal ahora es `kanji_classificator_model_training.ipynb`.
    * Se incluye `MAX_CLASSES_LIMIT` para permitir el entrenamiento con un número reducido de clases y se establece una semilla fija para reproducibilidad.
    * **HSK-1**: Se incluye el cuaderno `hsk1_with_my_architecture.ipynb` para pruebas con datasets externos.
    * Se configura el dispositivo de cómputo (GPU/CPU).

2.  **Preparación de Datos y Preprocesamiento**:
    * Soporte para el dataset **ETL9B** (binarizado), reemplazando al ETL9G.
    * Implementación de una clase para la gestión del dataset, encargada de leer las imágenes y sus etiquetas, con soporte para filtrar el número máximo de clases a utilizar.
    * Definición de **transformaciones y aumento de datos (Data Augmentation)** para el conjunto de entrenamiento, incluyendo:
        * Redimensionado de imágenes.
        * Conversión a escala de grises (replicando canales para compatibilidad con la red).
        * Transformaciones geométricas (perspectiva, rotación, traslación).
        * Ajustes de color y desenfoque.
        * Inversión aleatoria de colores y adición de ruido gaussiano.
    * División del dataset en subconjuntos de **Entrenamiento (80%)**, **Validación (10%)** y **Test (10%)**.

3.  **Definición de la Arquitectura**:
    * Carga de un modelo base pre-entrenado.
    * Modificación de la última capa completamente conectada para adaptar la salida al número total de clases (caracteres únicos) presentes en el dataset.

4.  **Bucle de Entrenamiento**:
    * Ejecución del entrenamiento durante un número definido de épocas.
    * Cálculo de la pérdida y la precisión tanto en entrenamiento como en validación.
    * Guardado automático del **mejor modelo** basado en la precisión de validación.
    * Almacenamiento del historial de métricas y el mapeo de clases.

5.  **Análisis de Resultados**:
    * Visualización de las curvas de aprendizaje (Pérdida y Precisión) para diagnosticar el comportamiento del modelo (e.g., overfitting).

6.  **Evaluación e Inferencia**:
    * Medición del rendimiento final utilizando el subconjunto de Test (no visto durante el entrenamiento).
    * Pruebas de predicción con imágenes externas situadas en una carpeta específica.
    * **Preprocesamiento**: Se incorpora umbralizado **Otsu** para binarizar las entradas de inferencia, asegurando compatibilidad con el entrenamiento sobre ETL9B.

## Arquitectura de la Red Neuronal

* **Modelo Base**: ResNet18 (Red Residual de 18 capas).
* **Pesos**: Pre-entrenados (Transfer Learning).
* **Adaptación**: Se sustituye la capa de clasificación original por una capa lineal que proyecta las características extraídas al número de clases Kanji del dataset (aprox. 2965 clases).
* **Entrada**: Imágenes redimensionadas a 128x128 píxeles. Aunque son en escala de grises, se tratan como 3 canales para cumplir con los requisitos de la red pre-entrenada.

## Hiperparámetros

A continuación se listan los hiperparámetros utilizados en esta versión del experimento:

| Parámetro | Valor | Descripción |
| :--- | :--- | :--- |
| **Learning Rate** | 0.002 | Tasa de aprendizaje inicial para el optimizador. |
| **Batch Size** | 128 | Número de muestras procesadas antes de actualizar el modelo. |
| **Epochs** | 30 | Número total de pasadas completas por el dataset de entrenamiento. |
| **Image Size** | 128 x 128 | Resolución a la que se redimensionan las imágenes de entrada. |
| **Optimizador** | Adam | Algoritmo de optimización utilizado. |
| **Función de Pérdida** | CrossEntropyLoss | Función de coste para clasificación multiclase. |

## Resultados Generales

En la ejecución registrada en este cuaderno, el modelo mostró una convergencia estable y rápida gracias al uso de Transfer Learning.

* **Precisión en Validación (Final)**: ~99.5%
* **Pérdida en Validación (Final)**: ~0.018
* **Observaciones**: El modelo alcanza una precisión muy alta en pocas épocas. Sería interesante implementar un "Early Stopping".

### Experimento HSK-1 (HSK1-resnet18-model-v1)

En este experimento se utilizó la arquitectura desarrollada para Kanji (ResNet18 adaptada) con un dataset de caracteres chinos HSK-1.

*   **Precisión en Validación**: 98.50%
*   **Precisión en Test**: 97.82%
*   **Pérdida en Validación**: 0.0645
*   **Observaciones**: Los resultados confirman que la arquitectura es altamente efectiva también para otros datasets de caracteres similares, manteniendo una precisión superior al 97%.