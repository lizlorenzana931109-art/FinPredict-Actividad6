# Resultados — FinPredict Actividad 6

Esta carpeta contiene las salidas generadas durante la evaluación y validación de los modelos de riesgo hipotecario desarrollados para **FinPredict**.

Los archivos fueron exportados directamente desde el notebook principal de la Actividad 6 con el objetivo de mantener trazabilidad entre el análisis, las métricas obtenidas y los entregables finales.

## Contenido

| Archivo | Descripción |
|---|---|
| `comparacion_modelos.csv` | Comparación inicial entre los modelos evaluados durante la fase de desarrollo. |
| `validacion_cruzada_temporal.csv` | Resultados de la validación cruzada temporal utilizada para evaluar estabilidad y generalización en diferentes periodos. |
| `comparacion_estadistica_bootstrap.csv` | Resultados del bootstrap agrupado por crédito para comparar estadísticamente Regresión Logística y Random Forest. |
| `analisis_threshold.csv` | Evaluación de diferentes umbrales de clasificación, incluyendo precision, recall, specificity, F1 y volumen de cartera priorizada. |
| `metricas_holdout_final.csv` | Métricas obtenidas en el conjunto de evaluación final no utilizado durante la selección del modelo. |
| `predicciones_holdout_final.csv` | Probabilidades de riesgo y clasificación final de cada crédito del holdout utilizando el threshold operativo seleccionado. |
| `simulacion_ab.csv` | Resultados de la simulación académica de una prueba A/B para evaluar una posible intervención preventiva. |
| `impacto_operativo_simulado.csv` | Indicadores derivados de la simulación A/B, incluyendo reducción absoluta, reducción relativa, NNT e incumplimientos evitados estimados. |
| `resumen_ejecutivo.csv` | Consolidado de los principales resultados técnicos y operativos de la Actividad 6. |

## Modelo seleccionado

El modelo seleccionado fue una **Regresión Logística**, después de comparar su desempeño contra Random Forest mediante validación cruzada temporal y análisis estadístico.

Principales resultados en el holdout final:

- Threshold operativo: **0.75**
- Precision: **25.47%**
- Recall: **75.64%**
- F1-score: **38.11%**
- Specificity: **91.41%**
- ROC-AUC: **0.9248**
- PR-AUC: **0.3469**
- Cartera priorizada: **11.10%**

## Validación estadística

La comparación mediante bootstrap agrupado por crédito mostró una diferencia media de **PR-AUC de 0.0104** a favor de la Regresión Logística, con un intervalo de confianza del 95% de **0.0039 a 0.0174**.

Para ROC-AUC, el intervalo de confianza incluyó cero, por lo que no se identificó una diferencia concluyente entre ambos modelos en esa métrica.

## Simulación A/B

La prueba A/B incluida en esta actividad corresponde a una **simulación académica** y no a una intervención real en producción.

Bajo el supuesto establecido para el experimento:

- Tasa de incumplimiento del grupo control: **26.09%**
- Tasa de incumplimiento del grupo FinPredict: **20.62%**
- Reducción relativa simulada: **20.95%**
- NNT estimado: **18.3**

Estos valores sirven para ilustrar cómo podría evaluarse el impacto operativo del modelo mediante un experimento controlado real.

## Consideraciones

Los datos utilizados en FinPredict son **sintéticos y de uso académico**.

Los resultados de esta carpeta deben interpretarse conjuntamente con el notebook principal, donde se documentan el proceso de evaluación, las decisiones metodológicas y la justificación del modelo seleccionado.
