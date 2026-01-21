# Configuración del Agente

Directorio que contiene archivos de configuración para el agente de IA (Antigravity/Gemini).

## Descripción General

Este directorio almacena las reglas y contratos que definen el comportamiento del agente al interactuar con el repositorio. Permite estandarizar commits, documentación y flujos de trabajo.

## Contenido

| Archivo | Tipo | Descripción |
| :--- | :--- | :--- |
| `rules.md` | Configuración | Contrato de reglas para commits y documentación. |

## Reglas Definidas

El archivo `rules.md` establece:

1. **Formato de Commits**: Etiquetas semánticas (`[MODEL]`, `[ARCH]`, `[DOCS]`, etc.).
2. **Auto-Tagging**: Creación automática de tags Git para commits de modelos.
3. **Idioma Híbrido**: Código en inglés, documentación en español.
4. **Estándar de READMEs**: Estructura basada en `TRAIN/README.md`.

## Uso

El agente lee automáticamente estas reglas al iniciar una sesión (si se configura en `.gemini/rules.md`) o puede referenciarse manualmente:

```
Lee las reglas en .antigravity/rules.md y síguelas.
```
