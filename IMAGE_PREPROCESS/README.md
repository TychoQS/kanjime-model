# Preprocesamiento de Imágenes

Pipeline de preprocesamiento para imágenes de caracteres Kanji, orientado a mejorar la calidad de entrada antes de la inferencia del modelo entrenado.

## Descripción General

Este directorio contiene un script de preprocesamiento que aplica una serie de transformaciones a imágenes. El pipeline incluye conversión a escala de grises, mejora de contraste adaptativa (CLAHE), filtro bilateral y umbralizado Otsu. Genera una imagen comparativa con todas las etapas del proceso.

## Contenido

| Archivo / Directorio | Tipo | Descripción |
| :--- | :--- | :--- |
| `preprocess.py` | Script | Pipeline principal de preprocesamiento de imágenes. |
| `samples/` | Carpeta | Imágenes de ejemplo para probar el pipeline. |
| `output/` | Carpeta | Resultados generados por el script (comparativas de etapas). |
| `README.md` | Doc | Esta documentación. |

## Etapas del Pipeline

El script `preprocess.py` aplica las siguientes transformaciones de forma secuencial:

1. **Original**: Carga de la imagen de entrada.
2. **Escala de grises**: Conversión a un solo canal.
3. **CLAHE** *(condicional)*: Mejora de contraste adaptativa, aplicada únicamente si la imagen se detecta como de bajo contraste.
4. **Suavizado Gaussiano**: Reducción de ruido con kernel 5×5.
5. **Umbralizado Otsu**: Binarización automática con umbral óptimo.

## Uso

```bash
python preprocess.py <ruta_imagen>
```

El resultado se guarda en la carpeta `output/` con el mismo nombre que la imagen de entrada.

**Ejemplo:**
```bash
python preprocess.py samples/1.jpeg
# → Saved: output/1.jpeg
```
