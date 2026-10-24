# Data

Esta carpeta contiene los datasets utilizados en la **Actividad 6 de FinPredict**, orientada a la evaluación y validación de modelos para identificar riesgo de incumplimiento en créditos hipotecarios.

## Origen de los datos

Los archivos de esta carpeta corresponden a una versión ajustada del dataset desarrollado previamente en la Actividad 5.

Durante la exploración inicial se identificó que la versión anterior presentaba una relación demasiado determinista entre los días de atraso al corte y el incumplimiento futuro, especialmente en créditos cercanos a los 90 días de mora.

Por esta razón se desarrolló una **versión V2**, manteniendo la lógica temporal original del proyecto, pero incorporando mayor heterogeneidad en el comportamiento de pago.

## Ajustes realizados en V2

De forma general, el proceso incluyó:

- Simulación de nuevos movimientos de pago para la cartera hipotecaria.
- Incorporación de pagos completos, parciales y regularizaciones.
- Mayor variabilidad entre perfiles de comportamiento.
- Conservación de la definición temporal del target `incumplimiento_90d`.
- Reconstrucción histórica de la morosidad por contrato y periodo.
- Segmentación operativa de la cartera:
  - **0 a 60 días de atraso:** población del modelo preventivo.
  - **61 a 89 días:** cobranza intensiva.
  - **90 días o más:** incumplimiento ya materializado.
- Construcción de variables transaccionales de los tres meses previos al corte.
- Separación temporal entre datos de desarrollo y evaluación final.

## Archivos

### `finpredict_preparado_v2.parquet`

Dataset analítico completo utilizado como base de la Actividad 6.

Contiene:

- **336,837 observaciones contrato-periodo**
- **44,445 créditos distintos**
- **30 variables**
- **12,898 incumplimientos**
- Prevalencia global aproximada de **3.83%**

### `finpredict_desarrollo_v2.parquet`

Conjunto destinado al desarrollo, entrenamiento y validación de modelos.

- **303,234 observaciones**
- **11,642 incumplimientos**
- Prevalencia aproximada de **3.84%**

Incluye únicamente ventanas históricas anteriores al periodo reservado para evaluación final.

### `finpredict_evaluacion_v2.parquet`

Conjunto reservado como **holdout temporal final**.

- **33,603 observaciones**
- **1,256 incumplimientos**
- Prevalencia aproximada de **3.74%**
- Fecha de corte: **30 de septiembre de 2025**
- Ventana de predicción: **1 de octubre al 31 de diciembre de 2025**

Este conjunto no debe utilizarse para entrenamiento ni selección de hiperparámetros.

### `manifiesto_v2.json`

Archivo de trazabilidad que documenta los principales criterios utilizados para construir V2, incluyendo:

- tamaño de los datasets;
- prevalencia;
- ventanas temporales;
- definición del target;
- segmentación operativa;
- criterios metodológicos de V2;
- semilla utilizada para reproducibilidad.

## Variable objetivo

La variable objetivo es:

`incumplimiento_90d`

Representa si un crédito mantiene saldo materialmente pendiente cuando alguna obligación alcanza **90 días de atraso dentro de la ventana futura de predicción**.

## Consideración metodológica

La versión V2 no fue creada con el propósito de reducir artificialmente el desempeño de los modelos.

Su objetivo fue obtener una representación más realista del problema de negocio, reduciendo relaciones excesivamente deterministas y permitiendo evaluar de forma más adecuada aspectos como:

- capacidad de generalización;
- precisión y recall;
- ROC-AUC y PR-AUC;
- matrices de confusión;
- ajuste de umbral;
- validación cruzada temporal;
- comparación estadística entre modelos;
- impacto operativo de las decisiones.

Los datos utilizados en FinPredict son de carácter sintético y fueron generados exclusivamente con fines académicos.
