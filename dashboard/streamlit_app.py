import streamlit as st
import pandas as pd
from pathlib import Path

# ============================================================
# FINPREDICT | TABLERO EJECUTIVO
# PASO 1 - CARGA DE RESULTADOS
# ============================================================

st.set_page_config(
    page_title="FinPredict | Actividad 6",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------
# Rutas del proyecto
# ------------------------------------------------------------

RUTA_DASHBOARD = Path(__file__).resolve().parent
RUTA_PROYECTO = RUTA_DASHBOARD.parent
RUTA_RESULTADOS = RUTA_PROYECTO / "resultados"

# ------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------

st.title("FinPredict")
st.subheader("Tablero ejecutivo de evaluación y validación")

st.write(
    "Resultados de la Actividad 6 para la evaluación "
    "de modelos de riesgo hipotecario."
)

# ------------------------------------------------------------
# Cargar resultados reales
# ------------------------------------------------------------

df_resumen = pd.read_csv(
    RUTA_RESULTADOS / "resumen_ejecutivo.csv"
)

df_holdout = pd.read_csv(
    RUTA_RESULTADOS / "metricas_holdout_final.csv"
)

df_threshold = pd.read_csv(
    RUTA_RESULTADOS / "analisis_threshold.csv"
)

# ------------------------------------------------------------
# Verificación visual
# ------------------------------------------------------------

st.success(
    "Datos cargados correctamente desde el repositorio."
)

# ------------------------------------------------------------
# KPIs principales
# ------------------------------------------------------------

st.markdown("## Resumen ejecutivo")

fila_holdout = df_holdout.iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "ROC-AUC",
        f"{fila_holdout['roc_auc']:.4f}"
    )

with col2:
    st.metric(
        "PR-AUC",
        f"{fila_holdout['pr_auc']:.4f}"
    )

with col3:
    st.metric(
        "Recall",
        f"{fila_holdout['recall']:.2%}"
    )

col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "Precision",
        f"{fila_holdout['precision']:.2%}"
    )

with col5:
    st.metric(
        "F1-score",
        f"{fila_holdout['f1']:.2%}"
    )

with col6:
    st.metric(
        "Cartera priorizada",
        f"{fila_holdout['porcentaje_priorizado']:.2%}"
    )

# ------------------------------------------------------------
# Decisión del modelo
# ------------------------------------------------------------

st.markdown("## Decisión del modelo")

st.info(
    """
    **Modelo seleccionado:** Regresión Logística  
    **Threshold operativo:** 0.75  

    El modelo fue seleccionado considerando desempeño en PR-AUC,
    estabilidad temporal, interpretabilidad y equilibrio entre
    detección de riesgo y capacidad operativa.
    """
)

# ------------------------------------------------------------
# Comparación de modelos
# ------------------------------------------------------------

# ------------------------------------------------------------
# Comparación de modelos - Validación cruzada temporal
# ------------------------------------------------------------

st.markdown("## Comparación de modelos")

df_cv = pd.read_csv(
    RUTA_RESULTADOS / "validacion_cruzada_temporal.csv"
)

df_bootstrap = pd.read_csv(
    RUTA_RESULTADOS / "comparacion_estadistica_bootstrap.csv"
)

st.write(
    "Desempeño promedio de Regresión Logística y Random Forest "
    "durante la validación cruzada temporal."
)

# ------------------------------------------------------------
# Resumen promedio por modelo
# ------------------------------------------------------------

columnas_metricas = [
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc"
]

df_comparacion_cv = (
    df_cv
    .groupby("modelo")[columnas_metricas]
    .mean()
    .reset_index()
)

# Formato visual
df_comparacion_visual = df_comparacion_cv.copy()

for columna in [
    "precision",
    "recall",
    "specificity",
    "f1"
]:
    df_comparacion_visual[columna] = (
        df_comparacion_visual[columna] * 100
    ).map("{:.2f}%".format)

for columna in [
    "roc_auc",
    "pr_auc"
]:
    df_comparacion_visual[columna] = (
        df_comparacion_visual[columna]
        .map("{:.4f}".format)
    )

df_comparacion_visual.columns = [
    "Modelo",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "ROC-AUC",
    "PR-AUC"
]

st.dataframe(
    df_comparacion_visual,
    use_container_width=True,
    hide_index=True
)

# ------------------------------------------------------------
# Comparación estadística
# ------------------------------------------------------------

st.markdown("### Comparación estadística")

fila_pr = df_bootstrap[
    df_bootstrap["metrica"] == "PR-AUC"
].iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Δ PR-AUC",
        f"{fila_pr['diferencia_media']:.4f}"
    )

with col2:
    st.metric(
        "IC 95% inferior",
        f"{fila_pr['IC_95_inf']:.4f}"
    )

with col3:
    st.metric(
        "IC 95% superior",
        f"{fila_pr['IC_95_sup']:.4f}"
    )

st.success(
    """
    **Resultado:** la Regresión Logística presentó una ventaja
    consistente en PR-AUC frente a Random Forest.

    El intervalo de confianza del 95% de la diferencia no incluye
    cero, lo que respalda la selección del modelo en una métrica
    especialmente relevante para una clase positiva poco frecuente.
    """
)

# ------------------------------------------------------------
# Evidencia estadística
# ------------------------------------------------------------

st.markdown("### Comparación estadística")

fila_pr = df_bootstrap[
    df_bootstrap["metrica"] == "PR-AUC"
].iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Δ PR-AUC",
        f"{fila_pr['diferencia_media']:.4f}"
    )

with col2:
    st.metric(
        "IC 95% inferior",
        f"{fila_pr['IC_95_inf']:.4f}"
    )

with col3:
    st.metric(
        "IC 95% superior",
        f"{fila_pr['IC_95_sup']:.4f}"
    )

st.success(
    """
    La Regresión Logística mostró una ventaja consistente en PR-AUC.
    El intervalo de confianza del 95% no incluye cero, lo que respalda
    su selección frente a Random Forest para este problema.
    """
)

# ------------------------------------------------------------
# Datos fuente
# ------------------------------------------------------------

with st.expander("Ver resultados técnicos completos"):
    
    st.write("### Resumen ejecutivo")
    st.dataframe(
        df_resumen,
        use_container_width=True
    )

    st.write("### Métricas del holdout final")
    st.dataframe(
        df_holdout,
        use_container_width=True
    )

    st.write("### Análisis de threshold")
    st.dataframe(
        df_threshold,
        use_container_width=True
    )
