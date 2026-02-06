# Kanji Recognition - Reconocimiento de Caracteres Japoneses

Sistema de reconocimiento de caracteres Kanji manuscritos mediante Deep Learning

## Descripción General

Este proyecto implementa un pipeline completo de Machine Learning para clasificar caracteres Kanji del dataset ETL9B. Incluye preprocesamiento de datos, entrenamiento con data augmentation, optimización de hiperparámetros con Optuna y evaluación con imágenes externas.

## Estructura del Repositorio

| Directorio | Descripción |
| :--- | :--- |
| `TRAIN/` | Entrenamiento, y modelos (`.pth`). |
| `TESTS/` | Imágenes manuscritas externas para pruebas de inferencia. |
| `DATA/` | Datasets (no incluido en el repositorio por tamaño). |
| `Noto_Sans_JP/` | Fuente tipográfica para generación de datos sintéticos. |
| `.antigravity/` | Configuración y reglas del agente de IA. |
| `training_log.csv` | Registro histórico de experimentos y modelos (Versionado). |


# Errores conocidos y solventados

## Discrepancia en Accuracy de Test (Commit 6216312) [SOLVED]
Los resultados de test reportados en commits anteriores a `6216312` pueden diferir de los valores mostrados en los notebooks debido a un error en la evaluación.

**Problema identificado:**
- La función `train_model()` y `train_kaggle()` devolvía el modelo de la **última época** en lugar del **mejor modelo** guardado en disco.
- Las evaluaciones en test se realizaban sobre el modelo incorrecto, causando variaciones en accuracy de ±2-4%.

**Solución aplicada:**
- Modificadas `train_model()` y `train_kaggle()` para cargar y devolver el mejor modelo automáticamente.
- Todas las evaluaciones posteriores usan el modelo con mejor validation accuracy.

**Impacto:**
- Resultados pre-fix: Test accuracy variable (ej: 90-94%).
- Resultados post-fix: Test accuracy estable y reproducible (ej: 92.4%).

**Nota:** Este problema afecta únicamente a los resultados de test. Los resultados de evaluación con otros datasets no se ven afectados ya que cargaban el modelo antes de evaluar.

## Inestabilidad en Curvas de Validación y Overfitting (Commit 3444174) [SOLVED]
A medida que se añadieron transformaciones de Data Augmentation (DA) más complejas, las curvas de validación empezaron a mostrar un comportamiento muy errático y una brecha creciente respecto a la curva de entrenamiento (overfitting).

**Problema identificado:**
- En los primeros modelos simples el entrenamiento era estable, pero al introducir `Erode`, `Dilate` y `Elastic` de forma agresiva, la validación se volvió inestable.
- Se descubrió que ciertas transformaciones estaban alterando las distribuciones de los caracteres de tal manera que el modelo aprendía artefactos irrelevantes para el dataset ETL9B (que es binarizado).

**Solución aplicada:**
- Se identificaron y eliminaron explícitamente las transformaciones que generaban mayor ruido en el dominio binario y alejaban la distribución de validación de la de entrenamiento:
    - `ColorJitter` (Incompatible con la naturaleza binarizada del dataset).
    - `GaussianNoise` (Generaba distribuciones de píxeles inconsistentes con el fondo limpio).
- Se ajustaron los parámetros de `ElasticTransform` para ser menos agresivos.

**Impacto:**
- **Estabilización:** Las curvas de validación pasaron de ser erráticas a seguir de cerca la tendencia de entrenamiento.
- **Mejora en Generalización:** Los modelos entrenados tras este ajuste presentan métricas mucho más consistentes tanto en el conjunto de test propio como en evaluaciones externas (CASIA).