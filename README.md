# Proyecto Final - Análisis de Accidentes en Madrid 🚦

Este proyecto analiza los accidentes de tráfico en Madrid con el objetivo de:
- Identificar franjas horarias y condiciones que influyen en la gravedad.
- Construir un modelo predictivo (Regresión Logística) para predecir si un implicado da positivo en alcohol.

## Contenido
- `ProyectoFinal1.ipynb`: Notebook con el EDA y análisis.
- `models.py`: Clase para el modelo de Regresión Logística.
- `testing_models.py` (opcional): script de testing del modelo.

## Tecnologías utilizadas
- Python (pandas, scikit-learn, matplotlib, seaborn, geopandas)
- Jupyter Notebook

## Resultados principales
- La mayoría de accidentes graves ocurren en horas de la noche/madrugada.
- El modelo de regresión logística logra alta exactitud, aunque con dificultades para detectar casos positivos de alcohol.
