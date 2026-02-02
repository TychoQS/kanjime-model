# Módulos de Entrenamiento

Este directorio contiene los módulos Python que implementan las distintas responsabilidades del pipeline de entrenamiento, extraídas del cuaderno principal para mejorar la organización y reutilización del código.

## Descripción General

Los módulos han sido diseñados siguiendo el principio de **separación de responsabilidades**, permitiendo:
- Mayor mantenibilidad del código
- Reutilización de componentes entre diferentes experimentos
- Testing más sencillo de componentes individuales
- Mejor organización del proyecto

## Contenido

| Archivo | Tipo | Descripción |
| :--- | :--- | :--- |
| `dataset.py` | Módulo | Clase `ETL9Dataset` para la carga y gestión del dataset ETL9. Soporta filtrado por número máximo de clases. |
| `evaluation.py` | Módulo | Funciones de evaluación e inferencia. Incluye `predict_and_evaluate` con soporte para **Monte Carlo Dropout**. |
| `train_model.py` | Módulo | Función principal `train_model` que ejecuta el bucle de entrenamiento con soporte para **checkpoints** y reanudación. |
| `train_utils.py` | Módulo | Utilidades de entrenamiento. Incluye la clase `EarlyStopping` para detener el entrenamiento cuando no hay mejora. |
| `transforms.py` | Módulo | Transformaciones personalizadas de Data Augmentation: `GaussianNoise` y `MorphologicalTransform` (erosión/dilatación). |
| `optuna.py` | Módulo | Función `objective` para la optimización de hiperparámetros con Optuna. |
| `fonts.py` | Módulo | Utilidades para la carga de fuentes japonesas (NotoSansJP) para visualización de caracteres. |
| `image_processing.py` | Módulo | Funciones de procesamiento de imágenes: binarización Otsu, preprocesamiento y desnormalización. |
| `models.py` | Módulo | Definición centralizada de arquitecturas (MobileViT v2 de la librería timm). |
| `config.py` | Módulo | Configuración global y rutas del proyecto centralizadas. |
| `data_loaders.py` | Módulo | Funciones para la creación de DataLoaders de entrenamiento y validación. |
| `utils.py` | Módulo | Utilidades generales (semillas, configuración de dispositivo, etc.). |



## Uso

Los módulos se importan desde el cuaderno principal de la siguiente forma:

```python
from modules.dataset import ETL9Dataset
from modules.train_model import train_model
from modules.train_utils import EarlyStopping
from modules.evaluation import predict_and_evaluate
from modules.transforms import GaussianNoise, MorphologicalTransform
from modules.optuna import objective
from modules.fonts import load_font
from modules import image_processing as ip
from modules.models import build_model
```

## Dependencias

Los módulos requieren las siguientes librerías:
- `torch` / `torchvision` - Framework de Deep Learning
- `numpy` - Operaciones numéricas
- `opencv-python` (cv2) - Procesamiento de imágenes
- `Pillow` (PIL) - Manipulación de imágenes
- `pandas` - Lectura de archivos CSV del dataset
- `matplotlib` - Visualización
- `optuna` - Optimización de hiperparámetros
- `tqdm` - Barras de progreso
- `timm` - Modelos pre-entrenados y bloques Transformer
- `torchinfo` - Visualización de la arquitectura (opcional)
