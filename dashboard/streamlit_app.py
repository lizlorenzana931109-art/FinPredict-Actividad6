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
    "Archivos cargados correctamente desde el repositorio."
)

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
