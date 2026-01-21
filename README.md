# Kanji Recognition - Reconocimiento de Caracteres Japoneses

Sistema de reconocimiento de caracteres Kanji manuscritos mediante Deep Learning

## Descripción General

Este proyecto implementa un pipeline completo de Machine Learning para clasificar caracteres Kanji del dataset ETL9G. Incluye preprocesamiento de datos, entrenamiento con data augmentation, y evaluación con imágenes externas.

## Estructura del Repositorio

| Directorio | Descripción |
| :--- | :--- |
| `TRAIN/` | Cuaderno de entrenamiento, modelos guardados y métricas. |
| `TESTS/` | Imágenes manuscritas externas para pruebas de inferencia. |
| `DATA/` | Dataset ETL9G (no incluido en el repositorio por tamaño). |
| `Noto_Sans_JP/` | Fuente tipográfica para generación de datos sintéticos. |
| `.antigravity/` | Configuración y reglas del agente de IA. |