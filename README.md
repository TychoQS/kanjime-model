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
