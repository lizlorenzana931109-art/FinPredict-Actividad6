# Notebooks

Esta carpeta contiene los notebooks utilizados para el desarrollo de la **Actividad 6 de FinPredict**, enfocada en la evaluación, validación y comparación de modelos de riesgo de incumplimiento hipotecario.

## Objetivo

Los notebooks documentan de forma reproducible el proceso completo de evaluación de modelos a partir del dataset V2 preparado específicamente para esta actividad.

La finalidad es analizar el desempeño de distintos modelos de clasificación y seleccionar la alternativa más adecuada considerando tanto criterios técnicos como impacto de negocio.

## Fuente de datos

Los notebooks utilizan los datasets almacenados en la carpeta:

`Data/`

Principalmente:

- `finpredict_desarrollo_v2.parquet`
- `finpredict_evaluacion_v2.parquet`
- `finpredict_preparado_v2.parquet`
- `manifiesto_v2.json`

La separación entre desarrollo y evaluación final es temporal, por lo que el conjunto de evaluación se mantiene reservado hasta la etapa final del análisis.

## Notebook principal

### `01_evaluacion_modelos_actividad6.ipynb`

Notebook principal de la Actividad 6.

Contiene el proceso de:

1. Configuración del entorno.
2. Carga y validación de los datos V2.
3. Definición de variables predictoras y variable objetivo.
4. Construcción de un modelo baseline.
5. Entrenamiento de modelos candidatos.
6. Evaluación mediante matriz de confusión.
7. Cálculo de métricas de clasificación.
8. Análisis de ROC y AUC.
9. Análisis de Precision-Recall.
10. Validación cruzada temporal.
11. Comparación estadística del desempeño.
12. Ajuste del threshold de clasificación.
13. Comparación de estrategias operativas.
14. Simulación de pruebas A/B.
15. Evaluación del impacto de negocio.
16. Selección y justificación del modelo final.
17. Generación de evidencias para el reporte técnico.

## Modelos considerados

La evaluación contempla inicialmente:

- `DummyClassifier` como línea base.
- Regresión Logística.
- Random Forest.

El modelo final no se define de manera anticipada. La selección se realiza a partir de los resultados obtenidos en la evaluación.

## Métricas

Debido al desbalance natural de la variable objetivo, la evaluación no se basa únicamente en accuracy.

Se consideran principalmente:

- Precision.
- Recall.
- F1-score.
- Specificity.
- ROC-AUC.
- PR-AUC.
- Matriz de confusión.

También se analiza el efecto de modificar el threshold de clasificación sobre los falsos positivos y falsos negativos.

## Validación

La validación conserva el orden temporal de los datos para evitar fuga de información.

El conjunto de evaluación final corresponde al corte del:

**30 de septiembre de 2025**

con una ventana de predicción entre:

**1 de octubre y 31 de diciembre de 2025**

Este conjunto se utiliza únicamente después de finalizar el entrenamiento, la validación cruzada y la selección de hiperparámetros.

## Reproducibilidad

Los notebooks deben ejecutarse en orden y mantener una semilla fija cuando exista algún proceso aleatorio.

Las principales decisiones metodológicas, métricas y resultados deben quedar documentados dentro de los propios notebooks mediante bloques Markdown y comentarios en código.

## Consideración académica

El proyecto FinPredict utiliza datos sintéticos con fines exclusivamente académicos.

La metodología busca aproximarse a un flujo de trabajo empresarial de machine learning, incorporando trazabilidad, separación temporal, validación estadística y criterios de negocio para la selección del modelo.
