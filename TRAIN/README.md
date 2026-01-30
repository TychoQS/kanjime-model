# Documentación de Entrenamiento de Modelo de Reconocimiento de Caracteres (Kanji)

Este documento describe el flujo de trabajo, la arquitectura y los resultados del experimento de entrenamiento registrado en el cuaderno adjunto.

## Descripción General
El cuaderno `kanji_classificator_model_training.ipynb` (antes `train.ipynb`) implementa un ciclo completo de aprendizaje profundo (Deep Learning) para la clasificación de caracteres manuscritos. El flujo ha sido actualizado para utilizar el dataset **ETL9B** (binarizado) y abarca desde la carga de datos hasta la inferencia con imágenes externas, incorporando umbralizado Otsu.

## Tabla de Contenidos

| Archivo / Directorio | Tipo | Descripción |
| :--- | :--- | :--- |
| `kanji_classificator_model_training.ipynb` | Cuaderno | Pipeline principal de entrenamiento y evaluación. |
| `hsk1_with_my_architecture.ipynb` | Cuaderno | Pruebas de arquitectura con el dataset HSK. |
| `modules/` | Carpeta | Módulos Python refactorizados del pipeline de entrenamiento. |
| `Training_Output/` | Carpeta | Resultados del entrenamiento (modelo, historia, clases). |
| `Training_Output/last_checkpoint.pth` | Archivo | Estado del último entrenamiento para reanudación. |
| `HSK_Training_Output/` | Carpeta | Resultados del entrenamiento HSK. |
| `README.md` | Doc | Esta documentación. |


## Etapas del Cuaderno

El cuaderno se estructura en las siguientes secciones lógicas:

1.  **Configuración y Constantes**:
    * Se establecen las rutas a los directorios de datos y salida.
    * Se definen los parámetros globales que gobernarán el entrenamiento (hiperparámetros).
    * **Renombrado**: El cuaderno principal ahora es `kanji_classificator_model_training.ipynb`.
    * Se incluye `MAX_CLASSES_LIMIT` para permitir el entrenamiento con un número reducido de clases y se establece una semilla fija para reproducibilidad.
    * **HSK-1**: Se incluye el cuaderno `hsk1_with_my_architecture.ipynb` para pruebas con datasets externos.
    * Se configura el dispositivo de cómputo (GPU/CPU).
    * **CASIA**: Se usa el dataset CASIA para pruebas con datasets externos en el modelo de kanjis.

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
    * **Verificación de Equilibrio**: Se incluye una celda que utiliza `Counter` para mostrar la frecuencia de cada clase y verificar si el dataset está equilibrado.

3.  **Definición de la Arquitectura**:
    * Carga de un modelo base pre-entrenado.
    * Modificación de la última capa completamente conectada para adaptar la salida al número total de clases (caracteres únicos) presentes en el dataset.

4.  **Bucle de Entrenamiento**:
    * Ejecución del entrenamiento durante un número definido de épocas.
    * **Sistema de Checkpoints**: Guardado automático del estado del entrenamiento (`last_checkpoint.pth`) en cada época para permitir la reanudación ante interrupciones.
    * Cálculo de la pérdida y la precisión tanto en entrenamiento como en validación.
    * Guardado automático del **mejor modelo** basado en la precisión de validación (`best_kanji_model.pth`).
    * Almacenamiento del historial de métricas y el mapeo de clases.

5.  **Análisis de Resultados**:
    * Visualización de las curvas de aprendizaje (Pérdida y Precisión) para diagnosticar el comportamiento del modelo (e.g., overfitting).

6.  **Evaluación e Inferencia**:
    * Medición del rendimiento final utilizando el subconjunto de Test (no visto durante el entrenamiento).
    * Pruebas de predicción con imágenes externas situadas en una carpeta específica.
    * **Preprocesamiento**: Se incorpora umbralizado **Otsu** para binarizar las entradas de inferencia, asegurando compatibilidad con el entrenamiento sobre ETL9B.
    * **Mejoras en Inferencia**: La función `predict_and_evaluate` ahora soporta dos tipos de formatos para la carga de imágenes:
        * Nombre de archivo como etiqueta (e.g., `あ.png`).
        * Carpeta contenedora como etiqueta (e.g., `あ/001.png`).
    * **Monte Carlo Dropout**: Se ha implementado la capacidad de aplicar Monte Carlo Dropout durante la inferencia para estimar la incertidumbre del modelo. Esto permite realizar múltiples pasadas con el dropout activado y calcular la media de las probabilidades y la desviación estándar (incertidumbre).

7.  **Optimización de Hiperparámetros (Optuna)**:
    * Se ha integrado **Optuna** para la búsqueda automática de los mejores hiperparámetros.
    * **Optimización de Velocidad**: Para acelerar la etapa de Optuna, se utiliza únicamente un **2%** de las clases del dataset original.
    * Definición de una función objetivo (`objective`) que entrena el modelo con diferentes configuraciones sugeridas por el "trial".
    * Optimización de variables como: `learning_rate`, `batch_size`, `optimizer` (Adam/SGD), etc.
    * Visualización de la importancia de los hiperparámetros y el historial de optimización.

## Arquitectura de la Red Neuronal

* **Modelo Base**: MobileNetV3 Large (Optimizado para eficiencia).
* **Pesos**: Pre-entrenados (Transfer Learning).
* **Adaptación**: Se sustituye la capa de clasificación original por una capa lineal que proyecta las características extraídas al número de clases Kanji del dataset (aprox. 2965 clases).
* **Entrada**: Imágenes redimensionadas a 128x128 píxeles. Aunque son en escala de grises, se tratan como 3 canales para cumplir con los requisitos de la red pre-entrenada.

## Hiperparámetros

A continuación se listan los hiperparámetros utilizados en esta versión del experimento:

| Parámetro | Valor | Descripción |
| :--- | :--- | :--- |
| **Learning Rate** | 0.0033408 | Tasa de aprendizaje sugerida por Optuna. |
| **Batch Size** | 64 | Tamaño de lote sugerido por Optuna. |
| **Weight Decay** | 2.3089e-04 | Regularización L2 sugerida por Optuna. |
| **Epochs** | 30 | Límite máximo de épocas (Early Stopping aplicado). |
| **Image Size** | 128 x 128 | Resolución de entrada. |
| **Optimizador** | AdamW | Variante avanzada del optimizador Adam. |
| **Early Stopping** | Paciencia 5 | Detención si la precisión de validación no mejora. |

## Resultados Generales

En la ejecución registrada en este cuaderno utilizando la arquitectura **MobileNetV3 Large**, se obtuvieron los siguientes resultados:

* **ID del Experimento**: `mobilenet_v3-model-v2`
* **Precisión en Validación (Mejor)**: 24.84% (Época 18)
* **Pérdida en Validación (Mejor)**: 3.3931
* **Precisión en Entrenamiento (Final)**: 45.37% (Época 23)
* **Precisión en Test (Final)**: 22.16%
* **Observaciones**: Se observó una caída significativa en la precisión con respecto a la versión anterior, a pesar de utilizar los mejores parámetros sugeridos por Optuna (`lr: 0.00334`, `batch_size: 64`). El entrenamiento se detuvo por Early Stopping en la época 23. Se mantiene la eliminación del factor `delta` en el Early Stopping. En este entrenamiento se congelaron todas las capas menos la última.

## Modularización (Refactorización)

El código del cuaderno principal ha sido **refactorizado** para separar las responsabilidades en módulos independientes ubicados en la carpeta `modules/`. Esta estructura permite:

- **Mejor organización**: Cada módulo tiene una responsabilidad específica.
- **Reutilización**: Los componentes pueden importarse en otros experimentos o cuadernos.
- **Mantenibilidad**: Cambios en una funcionalidad no afectan al resto del código.
- **Testing**: Facilita la creación de pruebas unitarias para cada componente.

Los módulos principales son:
- `dataset.py`: Gestión del dataset ETL9
- `train_model.py`: Bucle de entrenamiento con checkpoints
- `evaluation.py`: Inferencia y evaluación con Monte Carlo Dropout
- `optuna.py`: Optimización de hiperparámetros
- `transforms.py`: Transformaciones de Data Augmentation

Para más detalles, consultar el archivo `modules/README.md`.