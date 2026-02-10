# Documentación de Reuniones y Seguimiento

Este directorio contiene actas, resúmenes de trabajo y guías preparadas para las reuniones con el tutor del Trabajo de Fin de Grado (TFG).

## Descripción General

El objetivo de este espacio es centralizar toda la documentación que sirve de apoyo para las tutorías, permitiendo una trazabilidad clara del progreso del proyecto y facilitando la explicación de los hitos técnicos alcanzados.

## Tabla de Contenidos

| Archivo | Tipo | Descripción |
| :--- | :--- | :--- |
| `walkthrough.md` | Guía | Resumen completo de cambios y experimentos del 21/01 al 10/02. |
| `plots.R` | Script | Script de R base para generar análisis visual desde `training_log.csv`. |
| `performance_plots.pdf` | Informe | Gráficos de rendimiento (Accuracy, Top-5) por experimento y arquitectura. |
| `README.md` | Doc | Esta documentación. |

## Uso

Los archivos de este directorio están diseñados para ser consultados durante las reuniones. 
- Para generar o actualizar los gráficos de rendimiento, ejecutar: `Rscript plots.R` desde este directorio.
- Se recomienda usar visores de Markdown que soporten diagramas Mermaid para `walkthrough.md`.

## Dependencias

- **R Base**: Necesario para ejecutar el script de análisis de datos.
- **Visor Markdown**: Compatible con Github Flavored Markdown.
- **Mermaid.js**: Necesario para renderizar diagramas en `walkthrough.md`.
