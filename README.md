# FinPredict · Actividad 6

### Evaluación y validación de modelos para riesgo hipotecario

[![Streamlit App](https://img.shields.io/badge/Streamlit-Abrir%20Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://finpredict-actividad6-m8fgl8qd4gduyk4kgs3mwk.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Estado-Actividad%206%20completada-1FB5AD)

---

## Descripción

**FinPredict** es un proyecto académico de inteligencia artificial orientado a la identificación preventiva de riesgo de incumplimiento en créditos hipotecarios.

En esta Actividad 6 se evalúan y validan modelos supervisados mediante un enfoque que combina:

- baseline de referencia;
- comparación de modelos;
- validación cruzada temporal;
- métricas adaptadas al desbalance de clases;
- análisis estadístico mediante bootstrap agrupado por crédito;
- ajuste de threshold;
- evaluación final sobre holdout independiente;
- curvas ROC y Precision-Recall;
- simulación académica de prueba A/B;
- análisis de impacto operativo;
- visualización ejecutiva mediante Streamlit.

El objetivo no es automatizar decisiones crediticias adversas, sino explorar el uso de modelos predictivos para **priorizar intervenciones preventivas con supervisión humana**.

---

## Dashboard interactivo

La evidencia principal de la Actividad 6 puede explorarse mediante el tablero ejecutivo desarrollado en Streamlit:

### 👉 [Abrir FinPredict Dashboard](https://finpredict-actividad6-m8fgl8qd4gduyk4kgs3mwk.streamlit.app/)

El dashboard consume directamente los resultados generados durante el proceso analítico y permite consultar:

- métricas finales del modelo;
- comparación entre Regresión Logística y Random Forest;
- validación estadística;
- análisis interactivo de threshold;
- matrices de confusión OOF y holdout;
- curvas ROC y Precision-Recall;
- simulación A/B;
- impacto operativo estimado;
- evidencia técnica;
- recomendación ejecutiva.

---

## Problema de negocio

FinPredict busca anticipar si un crédito hipotecario de la población preventiva puede alcanzar un **incumplimiento de 90 días** durante el horizonte futuro de predicción.

La segmentación operativa utilizada es:

| Segmento | Tratamiento |
|---|---|
| 0–60 días de atraso | Población preventiva evaluada por FinPredict |
| 61–89 días | Gestión intensiva mediante regla operativa |
| ≥90 días | Incumplimiento ya materializado; fuera del modelo predictivo |

Esta separación evita utilizar inteligencia artificial en casos donde el deterioro ya es prácticamente evidente y concentra el modelo en una etapa donde todavía existe oportunidad preventiva.

---

## Datos

El proyecto utiliza un **dataset sintético con fines exclusivamente académicos**.

La versión utilizada para la Actividad 6 contiene:

| Dataset | Observaciones | Incumplimientos | Prevalencia |
|---|---:|---:|---:|
| Desarrollo | 303,234 | 11,642 | 3.84% |
| Holdout final | 33,603 | 1,256 | 3.74% |

El conjunto final de evaluación corresponde a un periodo temporal posterior y permaneció reservado durante la selección del modelo y del threshold.

---

## Modelos evaluados

Se utilizaron tres niveles de referencia:

1. **DummyClassifier** como baseline.
2. **Regresión Logística**.
3. **Random Forest**.

El baseline obtuvo una accuracy de **96.16%**, pero no detectó ningún incumplimiento.

Este resultado demuestra por qué la accuracy aislada no es adecuada para evaluar FinPredict: la clase positiva representa una proporción reducida de la cartera.

Por esta razón se priorizaron:

- Precision
- Recall
- Specificity
- F1-score
- ROC-AUC
- PR-AUC

---

## Validación temporal

La comparación entre modelos se realizó mediante **validación cruzada temporal expansiva**.

Resultados promedio:

| Modelo | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Regresión Logística | 20.03% | 85.09% | 86.17% | 32.41% | 0.9227 | 0.3409 |
| Random Forest | 21.99% | 82.28% | 88.11% | 34.68% | 0.9219 | 0.3318 |

Los resultados muestran modelos competitivos con diferentes compensaciones operativas.

---

## Comparación estadística

Para complementar la comparación descriptiva se utilizó **bootstrap agrupado por `id_credito`**, preservando la dependencia de las observaciones longitudinales de un mismo contrato.

### PR-AUC

- Diferencia Logística − Random Forest: **+0.0104**
- IC95%: **[0.0039, 0.0174]**

El intervalo no contiene cero, por lo que la ventaja de Regresión Logística en PR-AUC fue consistente dentro del análisis realizado.

### ROC-AUC

- Diferencia: **+0.0008**
- IC95%: **[-0.0003, 0.0018]**

En este caso el intervalo incluye cero, por lo que no se identificó una superioridad concluyente entre modelos.

---

## Modelo seleccionado

### Regresión Logística

La selección final consideró conjuntamente:

- desempeño en PR-AUC;
- estabilidad temporal;
- análisis estadístico;
- interpretabilidad;
- comportamiento sobre la clase positiva;
- capacidad de traducir el score a una estrategia operativa.

El modelo no se seleccionó únicamente por una métrica aislada.

---

## Threshold operativo

El threshold se analizó utilizando las predicciones **out-of-fold**, antes de abrir el holdout final.

Se seleccionó:

### **Threshold = 0.75**

Comparación:

| Threshold | Precision | Recall | F1 | Cartera priorizada |
|---|---:|---:|---:|---:|
| 0.50 | 20.00% | 85.11% | 32.39% | 16.61% |
| **0.75** | **25.28%** | **74.79%** | **37.79%** | **11.54%** |
| 0.85 | 31.71% | 58.40% | 41.10% | 7.19% |

Aunque 0.85 maximiza F1, reduce considerablemente la cobertura del evento.

El threshold **0.75** se adoptó como compromiso entre detección preventiva y capacidad operativa.

---

## Evaluación final

Una vez congelados el modelo y el threshold, se realizó una única evaluación sobre el holdout independiente.

### Métricas finales

| Métrica | Resultado |
|---|---:|
| ROC-AUC | **0.9248** |
| PR-AUC | **0.3469** |
| Precision | **25.47%** |
| Recall | **75.64%** |
| Specificity | **91.41%** |
| F1-score | **38.11%** |
| Cartera priorizada | **11.10%** |

### Matriz de confusión

| | Predicción: No incumplimiento | Predicción: Incumplimiento |
|---|---:|---:|
| **Real: No incumplimiento** | TN = **29,567** | FP = **2,780** |
| **Real: Incumplimiento** | FN = **306** | TP = **950** |

FinPredict detectó **950 de 1,256 incumplimientos futuros**, priorizando **3,730 créditos de 33,603**.

---

## Curvas ROC y Precision-Recall

En el holdout final:

- **ROC-AUC:** 0.9248
- **PR-AUC:** 0.3469
- **Baseline PR-AUC:** 0.0374

La curva Precision-Recall es especialmente relevante debido a la baja prevalencia de la clase positiva.

---

## Simulación A/B

Para ilustrar cómo podría evaluarse el impacto operativo del modelo se desarrolló una **simulación académica de prueba A/B**.

| Grupo | Créditos | Incumplimientos | Tasa |
|---|---:|---:|---:|
| Control | 1,863 | 486 | 26.09% |
| FinPredict | 1,867 | 385 | 20.62% |

Resultados de la simulación:

- Chi-cuadrada: **15.2599**
- p-value: **0.0001**
- Reducción absoluta simulada: **5.47 pp**
- Reducción relativa simulada: **20.95%**
- NNT estimado: **18.3**
- Incumplimientos evitados estimados: **102**

> **Importante:** esta prueba es una simulación académica construida bajo un supuesto de efectividad de intervención. Estos resultados no demuestran impacto causal real de FinPredict en producción.

---

## Recomendación ejecutiva

### GO condicionado para piloto controlado

La evidencia obtenida permite recomendar una siguiente fase de validación operativa controlada.

FinPredict presenta:

- desempeño superior al baseline;
- estabilidad entre validación temporal y holdout;
- evidencia estadística favorable en PR-AUC;
- un threshold operacionalmente defendible;
- capacidad de concentrar acciones preventivas en una fracción reducida de la cartera.

La solución debe utilizarse como herramienta de **priorización preventiva con supervisión humana**.

No se propone su uso para:

- rechazo automático de solicitudes;
- modificación automática de condiciones contractuales;
- decisiones adversas sobre clientes sin revisión humana.

---

## Estructura del repositorio

```text
FinPredict-Actividad6/
│
├── Data/
│   ├── finpredict_preparado_v2.parquet
│   ├── finpredict_desarrollo_v2.parquet
│   ├── finpredict_evaluacion_v2.parquet
│   └── manifiesto_v2.json
│
├── notebooks/
    └── 00_exploracion_dataset_actividad6.ipynb
│   └── 01_evaluacion_modelos_actividad6.ipynb
│
├── resultados/
│   ├── analisis_threshold.csv
│   ├── comparacion_estadistica_bootstrap.csv
│   ├── comparacion_modelos.csv
│   ├── impacto_operativo_simulado.csv
│   ├── metricas_holdout_final.csv
│   ├── predicciones_holdout_final.csv
│   ├── resumen_ejecutivo.csv
│   ├── simulacion_ab.csv
│   └── validacion_cruzada_temporal.csv
│
├── dashboard/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── README.md
│
└── README.md
```

---

## Flujo metodológico

```text
Datos sintéticos V2
        ↓
Preparación analítica
        ↓
DummyClassifier
        ↓
Regresión Logística
        ↓
Random Forest
        ↓
Validación cruzada temporal
        ↓
Predicciones OOF
        ↓
Bootstrap agrupado
        ↓
Selección de modelo
        ↓
Análisis de threshold
        ↓
Threshold 0.75
        ↓
Holdout independiente
        ↓
Simulación A/B
        ↓
Impacto operativo
        ↓
Dashboard ejecutivo
```

---

## Tecnologías utilizadas

- Python
- pandas
- NumPy
- scikit-learn
- SciPy
- Plotly
- Streamlit
- Google Colab
- GitHub

---

## Reproducibilidad

Los resultados presentados en el dashboard no se capturan manualmente.

El notebook genera archivos estructurados en la carpeta:

```text
resultados/
```

La aplicación Streamlit consume directamente estos archivos, manteniendo trazabilidad entre:

```text
Notebook
   ↓
Resultados CSV
   ↓
Dashboard
```

Esto permite separar la **capa analítica** de la **capa de consumo y visualización**.

---

## Uso de inteligencia artificial generativa

Se utilizó **ChatGPT** como herramienta de apoyo durante el desarrollo del proyecto para:

- estructuración y depuración de código;
- revisión metodológica;
- interpretación de resultados;
- exploración de alternativas de modelado;
- documentación del repositorio;
- desarrollo del dashboard;
- apoyo en el uso de GitHub y Streamlit;
- redacción y estructuración del informe técnico.

Las métricas, predicciones y resultados experimentales utilizados en el proyecto proceden de ejecuciones realizadas dentro del entorno de trabajo y fueron revisados antes de incorporarse a los entregables.

La inteligencia artificial generativa se utilizó como herramienta de asistencia y no como sustituto de la validación técnica ni de las decisiones de la autora.

---

## Repositorio

**FinPredict · Actividad 6**

https://github.com/lizlorenzana931109-art/FinPredict-Actividad6

---

## Aplicación

### 👉 [Abrir dashboard FinPredict](https://finpredict-actividad6-m8fgl8qd4gduyk4kgs3mwk.streamlit.app/)

---

## Consideraciones

- Los datos utilizados son sintéticos.
- El proyecto tiene fines académicos.
- Las métricas no constituyen evidencia de desempeño sobre una cartera real.
- La simulación A/B no representa una intervención real.
- Cualquier implementación productiva requeriría validación con datos autorizados, gobierno, seguridad, monitoreo, explicabilidad y supervisión humana.

---

## Autora

**Lizbeth Bruno Lorenzana**  
Maestría en Inteligencia Artificial  
Gestión de Proyectos de Inteligencia Artificial  
2026
