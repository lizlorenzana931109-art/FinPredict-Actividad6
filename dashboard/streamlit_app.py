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
