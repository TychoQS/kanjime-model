# Documentación de Módulos (TRAIN/modules)

Este directorio contiene la lógica refactorizada del pipeline de entrenamiento, separada en módulos específicos para mejorar la mantenibilidad y reutilización del código.

## Tabla de Contenidos

| Archivo | Descripción |
| :--- | :--- |
| `config.py` | **Configuración**: Contiene constantes globales, rutas de archivos y parámetros por defecto (Batch Size, LR, etc.). |
| `data_loaders.py` | **Carga de Datos**: Funciones para crear DataLoaders (`get_dataloaders`) y dividir el dataset (`create_splits`). |
| `dataset.py` | **Clase Dataset**: Implementación de `ETL9Dataset` para la gestión de imágenes ETL9B binarizadas. |
| `evaluation.py` | **Inferencia**: Funciones para evaluación (`predict_and_evaluate`), visualización de errores y soporte para Monte Carlo Dropout. |
| `fonts.py` | **Fuentes**: Utilidades para cargar fuentes japonesas necesarias para la visualización (Matplotlib). |
| `image_processing.py` | **Procesado**: Funciones de normalización/desnormalización y preprocesamiento de imágenes (Otsu). |
| `models.py` | **Arquitecturas**: Definición de modelos PyTorch. Incluye `MultiHeadKanjiClassificator` (MobileNetV3 con cabezal de componentes). |
| `optuna.py` | **Optimización**: Función `objective` para la búsqueda de hiperparámetros con Optuna. |
| `train_model.py` | **Entrenamiento**: Bucles principales de entrenamiento (`train_model`) y validación. |
| `train_utils.py` | **Utilidades de Entrenamiento**: Clases auxiliares como `EarlyStopping` y configuración de optimizadores/schedulers. |
| `transforms.py` | **Augmentation**: Definición de pipelines de transformación y Data Augmentation con `torchvision` y `albumentations`. |
| `utils.py` | **General**: Funciones de utilidad general como configuración de semilla (`set_seed`) y detección de dispositivo. |
| `visualization.py` | **Visualización**: Clase `TrainingPlotter` para la generación de gráficas de entrenamiento (pérdida y precisión). |

## Uso General

Para utilizar estos módulos desde un cuaderno o script en la carpeta superior (`TRAIN/`), asegúrese de que el paquete sea accesible (por defecto Python añade el directorio actual al path):

```python
from modules.config import *
from modules.models import build_multi_head_model
# ...
```

## Notas de Desarrollo

- **Nuevas Arquitecturas**: Si se añaden nuevos modelos, deben registrarse en `models.py`.
- **Configuración**: Para cambiar rutas o parámetros por defecto, editar `config.py`.
