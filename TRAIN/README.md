# Documentación de Entrenamiento de Modelo de Reconocimiento de Caracteres (Kanji)

Este documento describe el flujo de trabajo, la arquitectura y los resultados del experimento de entrenamiento registrado en el cuaderno adjunto.

## Descripción General
El cuaderno implementa un ciclo completo de aprendizaje profundo (Deep Learning) para la clasificación de caracteres manuscritos (específicamente los Kanji correspondientes al dataset ETL9G). El flujo abarca desde la carga de datos hasta la inferencia con imágenes externas, pasando por un preprocesamiento robusto y el entrenamiento de una red neuronal convolucional.

## Etapas del Cuaderno

El cuaderno se estructura en las siguientes secciones lógicas:

1.  **Configuración y Constantes**:
    * Se establecen las rutas a los directorios de datos y salida.
    * Se definen los parámetros globales que gobernarán el entrenamiento (hiperparámetros).
    * **Nuevo**: Se incluye `MAX_CLASSES_LIMIT` para permitir el entrenamiento con un número reducido de clases y se establece una semilla fija para reproducibilidad.
    * Se configura el dispositivo de cómputo (GPU/CPU).

2.  **Preparación de Datos y Preprocesamiento**:
    * Implementación de una clase para la gestión del dataset (`ETL9GDataset`), encargada de leer las imágenes y sus etiquetas, ahora con soporte para filtrar el número máximo de clases a utilizar.
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
    * Pruebas de predicción con imágenes externas situadas en una carpeta específica para verificar el funcionamiento en escenarios reales.

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