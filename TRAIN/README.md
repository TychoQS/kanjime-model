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
        * Redimensionado de imágenes (96x96).
        * Conversión a escala de grises (1 canal).
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
    * **Incertidumbre**: Se puede extender el modelo para estimar la incertidumbre mediante técnicas como Monte Carlo Dropout si se añaden capas de dropout al entrenamiento.

7.  **Optimización de Hiperparámetros (Optuna)**:
    * Se ha integrado **Optuna** para la búsqueda automática de los mejores hiperparámetros.
    * **Optimización de Velocidad**: Para acelerar la etapa de Optuna, se utiliza únicamente un **2%** de las clases del dataset original.
    * Definición de una función objetivo (`objective`) que entrena el modelo con diferentes configuraciones sugeridas por el "trial".
    * Optimización de variables como: `learning_rate`, `batch_size`, `optimizer` (Adam/SGD), etc.
    * Visualización de la importancia de los hiperparámetros y el historial de optimización.

## Arquitectura de la Red Neuronal

* **Modelo Base**: GhostNet.
* **Cabezales (Multi-Head)**:
    * **Kanji Head**: Clasificación de los caracteres Kanji (clases principales).
    * **Component Head**: Clasificación auxiliar de componentes/radicales para mejorar la representación interna.
* **Entrada**: Imágenes redimensionadas a 128x128 píxeles en 3 canales.
* **Estrategia**: El backbone extrae features que se comparten entre ambos cabezales, forzando al modelo a aprender características estructurales (trazos/radicales) robustas.

## Hiperparámetros

A continuación se listan los hiperparámetros utilizados en esta versión del experimento:

| Parámetro | Valor | Descripción |
| :--- | :--- | :--- |
| **Learning Rate** | 0.0008189 | Tasa de aprendizaje (Optuna). |
| **Batch Size** | 128 | Tamaño de lote. |
| **Epochs** | 30 | Épocas ejecutadas. |
| **Image Size** | 128 x 128 | Resolución de entrada (3 canales). |
| **Optimizador** | AdamW | Optimizador AdamW (con Weight Decay). |
| **Arquitectura** | GhostNet | Arquitectura orientada a eficiencia con módulos Ghost. |

## Resultados Generales

En la ejecución registrada en este cuaderno utilizando la arquitectura **GhostNet**, se obtuvieron los siguientes resultados:

* **ID del Experimento**: `ghostnet-model-v3`
* **Precisión en Validación (Mejor)**: 99.54%
* **Pérdida en Validación (Mejor)**: 0.0233
* **Precisión en Entrenamiento (Final)**: 96.72%
* **Precisión en Test (Final)**: 99.73%
* **Top-5 Precisión en Test**: 100.00%
* **Evaluación CASIA (Train Top-5)**: 93.50%
* **Observaciones**: El modelo utiliza una arquitectura multi-head (Kanji + Componentes) sobre un backbone GhostNet. Se han utilizado los mejores hiperparámetros encontrados por Optuna para maximizar el rendimiento.

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
- `transforms.py`: Transformaciones de Data Augmentation.
- `models.py`: Definición de arquitecturas de modelos de red neuronal.

Para más detalles, consultar el archivo `modules/README.md`.