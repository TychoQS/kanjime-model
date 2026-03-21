# Benchmarks de Detectores de Texto

## Descripción General

Este directorio contiene los scripts utilizados para evaluar el rendimiento temporal y computacional de los detectores de texto integrados en el pipeline de preprocesamiento.

## Tabla de Contenidos

| Archivo / Directorio | Tipo | Descripción |
| :--- | :--- | :--- |
| `benchmark_fast.py` | Script | Evaluación de rendimiento para el modelo FAST. |
| `benchmark_mixnet.py` | Script | Evaluación de rendimiento para el modelo MixNet. |
| `benchmark_pan_pp.py` | Script | Evaluación de rendimiento para el modelo PAN++. |
| `README.md` | Doc | Esta documentación. |

## Uso

Los scripts de este directorio son ejecutados generalmente a través del script principal ubicado en el directorio superior (`run_text_detectors_benchmark.sh`). Cada script inicializa el modelo correspondiente y mide los tiempos de inferencia.
