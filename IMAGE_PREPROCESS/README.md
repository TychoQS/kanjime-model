# Preprocesamiento de Imágenes

Pipeline de preprocesamiento para imágenes de caracteres Kanji, orientado a mejorar la calidad de entrada antes de la inferencia del modelo entrenado.

## Descripción General

Este directorio contiene un script de preprocesamiento que aplica una serie de transformaciones a imágenes. El pipeline incluye conversión a escala de grises, mejora de contraste adaptativa (CLAHE), filtro bilateral, umbralizado Otsu y operación de cierre morfológico. Genera una imagen comparativa con todas las etapas del proceso.

## Tabla de Contenidos

| Archivo / Directorio | Tipo | Descripción |
| :--- | :--- | :--- |
| `custom_binarization.py` | Script | Pipeline principal de binarización y preprocesamiento (reemplaza a `preprocess.py`). |
| `preprocess_utils.py` | Módulo | Clases de utilidad para el guardado de resultados con soporte para mosaicos de etapas. |
| `samples/` | Carpeta | Colección de imágenes de caracteres Kanji para pruebas. |
| `output/` | Carpeta | Resultados del procesamiento organizados por script. |
| `README.md` | Doc | Esta documentación. |

## Etapas del Pipeline

El script `preprocess.py` aplica las siguientes transformaciones de forma secuencial:

1. **Original**: Carga de la imagen de entrada.
2. **Escala de grises**: Conversión a un solo canal.
3. **CLAHE** *(condicional)*: Mejora de contraste adaptativa, aplicada únicamente si la imagen se detecta como de bajo contraste.
4. **Filtrado Bilateral**: Reducción de ruido con mejor resultado en bordes.
5. **Umbralizado Otsu**: Binarización automática con umbral óptimo.
6. **Operación Morfológica**: Cierre morfológico para eliminar pequeños ruidos en el interior de los trazos.

## Uso

Para procesar una imagen y generar la comparativa de etapas:

```bash
python custom_binarization.py <ruta_a_la_imagen>
```

**Ejemplo:**
```bash
python custom_binarization.py samples/1.jpeg
```

El script genera una imagen comparativa en `output/custom_binarization/` que incluye las etapas de: Grayscale, CLAHE (si es necesario), Filtro Bilateral, Otsu y Operación Morfológica.
