# ============================================================
# FINPREDICT | ACTIVIDAD 6
# TABLERO EJECUTIVO DE EVALUACIÓN Y VALIDACIÓN
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy.stats import chi2_contingency
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="FinPredict | Actividad 6",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. BRANDING FINPREDICT
# ============================================================

NAVY = "#102A43"
NAVY_2 = "#173F5F"
TEAL = "#1FB5AD"
TEAL_LIGHT = "#E7F7F5"
LIGHT = "#F5F8FA"
GRAY = "#6B7C93"
BORDER = "#D9E2EC"
WHITE = "#FFFFFF"


st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {WHITE};
    }}

    .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }}

    h1, h2, h3 {{
        color: {NAVY};
        font-family: Arial, Helvetica, sans-serif;
    }}

    p, div, span {{
        font-family: Arial, Helvetica, sans-serif;
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    .fin-header {{
        border-top: 4px solid {TEAL};
        border-bottom: 1px solid {BORDER};
        padding: 18px 0 22px 0;
        margin-bottom: 22px;
    }}

    .fin-kicker {{
        font-size: 13px;
        font-weight: 700;
        letter-spacing: .8px;
        color: {TEAL};
        text-transform: uppercase;
        margin-bottom: 6px;
    }}

    .fin-title {{
        font-size: 38px;
        line-height: 1.1;
        font-weight: 750;
        color: {NAVY};
        margin-bottom: 8px;
    }}

    .fin-subtitle {{
        font-size: 17px;
        color: {GRAY};
        line-height: 1.5;
    }}

    .executive-box {{
        background-color: {TEAL_LIGHT};
        border-left: 7px solid {TEAL};
        padding: 18px 20px;
        border-radius: 5px;
        margin: 14px 0 22px 0;
        color: {NAVY};
    }}

    .method-box {{
        background-color: #F0F4F8;
        border: 1px solid {BORDER};
        padding: 15px 18px;
        border-radius: 8px;
        color: {NAVY};
        margin-top: 10px;
    }}

    [data-testid="stMetric"] {{
        background-color: {LIGHT};
        border: 1px solid {BORDER};
        padding: 16px 18px;
        border-radius: 8px;
        min-height: 120px;
    }}

    [data-testid="stMetricLabel"] {{
        color: {GRAY};
        font-weight: 600;
    }}

    [data-testid="stMetricValue"] {{
        color: {NAVY};
        font-weight: 750;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}

    hr {{
        border: none;
        height: 1px;
        background: {BORDER};
        margin: 32px 0;
    }}

    .fin-footer {{
        margin-top: 35px;
        padding-top: 14px;
        border-top: 1px solid {BORDER};
        color: {GRAY};
        font-size: 12px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. RUTAS DEL PROYECTO
# ============================================================

RUTA_DASHBOARD = Path(__file__).resolve().parent
RUTA_PROYECTO = RUTA_DASHBOARD.parent
RUTA_RESULTADOS = RUTA_PROYECTO / "resultados"


# ============================================================
# 4. FUNCIONES AUXILIARES
# ============================================================

@st.cache_data
def cargar_csv(nombre_archivo):
    ruta = RUTA_RESULTADOS / nombre_archivo

    if not ruta.exists():
        st.error(
            f"No se encontró el archivo: resultados/{nombre_archivo}"
        )
        st.stop()

    return pd.read_csv(ruta)


def porcentaje(valor):
    return f"{valor:.2%}"


def numero(valor):
    return f"{int(valor):,}"


def crear_matriz_confusion(
    tn,
    fp,
    fn,
    tp,
    titulo
):

    matriz = np.array([
        [tn, fp],
        [fn, tp]
    ])

    textos = np.array([
        [
            f"TN<br>{tn:,}",
            f"FP<br>{fp:,}"
        ],
        [
            f"FN<br>{fn:,}",
            f"TP<br>{tp:,}"
        ]
    ])

    fig = go.Figure(
        data=go.Heatmap(
            z=matriz,
            x=[
                "Predicción: No incumplimiento",
                "Predicción: Incumplimiento"
            ],
            y=[
                "Real: No incumplimiento",
                "Real: Incumplimiento"
            ],
            text=textos,
            texttemplate="%{text}",
            textfont=dict(
                size=22
            ),
            colorscale=[
                [0, "#EEF3F8"],
                [1, TEAL]
            ],
            showscale=False
        )
    )

    fig.update_layout(
        title=titulo,
        height=450,
        template="plotly_white",
        margin=dict(
            l=30,
            r=30,
            t=70,
            b=30
        )
    )

    fig.update_yaxes(
        autorange="reversed"
    )

    return fig


# ============================================================
# 5. CARGA DE RESULTADOS
# ============================================================

df_resumen = cargar_csv(
    "resumen_ejecutivo.csv"
)

df_holdout = cargar_csv(
    "metricas_holdout_final.csv"
)

df_threshold = cargar_csv(
    "analisis_threshold.csv"
)

df_cv = cargar_csv(
    "validacion_cruzada_temporal.csv"
)

df_bootstrap = cargar_csv(
    "comparacion_estadistica_bootstrap.csv"
)

df_predicciones = cargar_csv(
    "predicciones_holdout_final.csv"
)

df_ab = cargar_csv(
    "simulacion_ab.csv"
)

df_impacto = cargar_csv(
    "impacto_operativo_simulado.csv"
)


# ============================================================
# 6. ENCABEZADO
# ============================================================

header_html = f"""
<div class="fin-header">
    <div class="fin-kicker">
        FINCOMERCIAL, S.A. · FINPREDICT · ACTIVIDAD 6
    </div>

    <div class="fin-title">
        Tablero ejecutivo de evaluación y validación
    </div>

    <div class="fin-subtitle">
        Evaluación del modelo de riesgo hipotecario mediante
        validación temporal, análisis estadístico, ajuste de
        threshold, holdout independiente y simulación A/B.
    </div>
</div>
"""

st.markdown(
    header_html,
    unsafe_allow_html=True
)


# ============================================================
# 7. DECISIÓN EJECUTIVA
# ============================================================

st.markdown(
    """
    <div class="executive-box">

    <strong>Decisión ejecutiva · Regresión Logística</strong><br><br>

    Modelo seleccionado mediante validación cruzada temporal y
    comparación estadística.

    <strong>Threshold operativo: 0.75.</strong>

    La selección busca equilibrar capacidad de detección,
    interpretabilidad y volumen operativo de alertas preventivas.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 8. RESUMEN EJECUTIVO
# ============================================================

st.markdown("## 1. Resumen ejecutivo")

fila_holdout = df_holdout.iloc[0]

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "ROC-AUC",
        f"{fila_holdout['roc_auc']:.4f}"
    )

with c2:
    st.metric(
        "PR-AUC",
        f"{fila_holdout['pr_auc']:.4f}"
    )

with c3:
    st.metric(
        "Recall",
        porcentaje(
            fila_holdout["recall"]
        )
    )

c4, c5, c6 = st.columns(3)

with c4:
    st.metric(
        "Precision",
        porcentaje(
            fila_holdout["precision"]
        )
    )

with c5:
    st.metric(
        "F1-score",
        porcentaje(
            fila_holdout["f1"]
        )
    )

with c6:
    st.metric(
        "Cartera priorizada",
        porcentaje(
            fila_holdout[
                "porcentaje_priorizado"
            ]
        )
    )


st.markdown(
    f"""
    <div class="method-box">

    En el holdout final, FinPredict prioriza
    <strong>
    {numero(fila_holdout['casos_priorizados'])} créditos
    </strong>
    de una cartera de 33,603 observaciones.

    El modelo identifica
    <strong>
    {numero(fila_holdout['tp'])} incumplimientos futuros
    </strong>
    y mantiene una especificidad de
    <strong>
    {porcentaje(fila_holdout['specificity'])}
    </strong>.

    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


# ============================================================
# 9. COMPARACIÓN DE MODELOS
# ============================================================

st.markdown("## 2. Comparación de modelos")

st.write(
    "Desempeño promedio de Regresión Logística y Random Forest "
    "durante la validación cruzada temporal."
)

metricas_cv = [
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc"
]

df_comparacion_cv = (
    df_cv
    .groupby("modelo")[metricas_cv]
    .mean()
    .reset_index()
)

df_cv_tabla = df_comparacion_cv.copy()

for columna in [
    "precision",
    "recall",
    "specificity",
    "f1"
]:
    df_cv_tabla[columna] = (
        df_cv_tabla[columna]
        .map(
            lambda x: f"{x:.2%}"
        )
    )

for columna in [
    "roc_auc",
    "pr_auc"
]:
    df_cv_tabla[columna] = (
        df_cv_tabla[columna]
        .map(
            lambda x: f"{x:.4f}"
        )
    )

df_cv_tabla.columns = [
    "Modelo",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "ROC-AUC",
    "PR-AUC"
]

st.dataframe(
    df_cv_tabla,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# Gráfico comparativo
# ------------------------------------------------------------

df_plot_modelos = (
    df_comparacion_cv[
        [
            "modelo",
            "recall",
            "f1",
            "pr_auc"
        ]
    ]
    .melt(
        id_vars="modelo",
        var_name="Métrica",
        value_name="Valor"
    )
)

nombres_metricas = {
    "recall": "Recall",
    "f1": "F1",
    "pr_auc": "PR-AUC"
}

df_plot_modelos[
    "Métrica"
] = (
    df_plot_modelos[
        "Métrica"
    ]
    .map(
        nombres_metricas
    )
)

fig_modelos = px.bar(
    df_plot_modelos,
    x="Métrica",
    y="Valor",
    color="modelo",
    barmode="group",
    text_auto=".3f",
    color_discrete_sequence=[
        NAVY,
        TEAL
    ]
)

fig_modelos.update_layout(
    title=(
        "Comparación de desempeño "
        "en validación temporal"
    ),
    xaxis_title="",
    yaxis_title="Valor",
    legend_title="Modelo",
    height=430,
    template="plotly_white"
)

st.plotly_chart(
    fig_modelos,
    use_container_width=True
)


# ============================================================
# 10. COMPARACIÓN ESTADÍSTICA
# ============================================================

st.markdown(
    "### Comparación estadística"
)

fila_pr = df_bootstrap[
    df_bootstrap[
        "metrica"
    ] == "PR-AUC"
].iloc[0]

fila_roc = df_bootstrap[
    df_bootstrap[
        "metrica"
    ] == "ROC-AUC"
].iloc[0]


b1, b2, b3 = st.columns(3)

with b1:
    st.metric(
        "Δ PR-AUC",
        f"{fila_pr['diferencia_media']:.4f}"
    )

with b2:
    st.metric(
        "IC 95% inferior",
        f"{fila_pr['IC_95_inf']:.4f}"
    )

with b3:
    st.metric(
        "IC 95% superior",
        f"{fila_pr['IC_95_sup']:.4f}"
    )


st.markdown(
    f"""
    <div class="executive-box">

    <strong>Resultado estadístico</strong><br><br>

    La diferencia de PR-AUC se calculó como:

    <strong>
    Regresión Logística − Random Forest.
    </strong>

    La diferencia media fue
    <strong>
    {fila_pr['diferencia_media']:.4f}
    </strong>,
    con IC95%
    <strong>
    [{fila_pr['IC_95_inf']:.4f},
    {fila_pr['IC_95_sup']:.4f}]
    </strong>.

    Como el intervalo no contiene cero,
    la ventaja en PR-AUC es consistente
    a favor de Regresión Logística.

    Para ROC-AUC, el IC95% fue
    <strong>
    [{fila_roc['IC_95_inf']:.4f},
    {fila_roc['IC_95_sup']:.4f}]
    </strong>,
    por lo que no se identifica una
    superioridad concluyente.

    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


# ============================================================
# 11. ANÁLISIS INTERACTIVO DE THRESHOLD
# ============================================================

st.markdown(
    "## 3. Threshold y capacidad operativa"
)

st.write(
    "El threshold modifica el equilibrio entre detección "
    "de incumplimientos y volumen de créditos enviados "
    "a intervención."
)

thresholds_disponibles = sorted(
    df_threshold[
        "threshold"
    ].unique()
)

threshold_seleccionado = st.select_slider(
    "Selecciona un threshold",
    options=thresholds_disponibles,
    value=0.75
)

fila_threshold = df_threshold[
    np.isclose(
        df_threshold[
            "threshold"
        ],
        threshold_seleccionado
    )
].iloc[0]


t1, t2, t3, t4 = st.columns(4)

with t1:
    st.metric(
        "Precision",
        porcentaje(
            fila_threshold[
                "precision"
            ]
        )
    )

with t2:
    st.metric(
        "Recall",
        porcentaje(
            fila_threshold[
                "recall"
            ]
        )
    )

with t3:
    st.metric(
        "F1-score",
        porcentaje(
            fila_threshold[
                "f1"
            ]
        )
    )

with t4:
    st.metric(
        "Cartera priorizada",
        porcentaje(
            fila_threshold[
                "porcentaje_priorizado"
            ]
        )
    )


# ------------------------------------------------------------
# Curva threshold
# ------------------------------------------------------------

fig_threshold = go.Figure()

fig_threshold.add_trace(
    go.Scatter(
        x=df_threshold[
            "threshold"
        ],
        y=df_threshold[
            "precision"
        ],
        mode="lines+markers",
        name="Precision",
        line=dict(
            color=NAVY
        )
    )
)

fig_threshold.add_trace(
    go.Scatter(
        x=df_threshold[
            "threshold"
        ],
        y=df_threshold[
            "recall"
        ],
        mode="lines+markers",
        name="Recall",
        line=dict(
            color=TEAL
        )
    )
)

fig_threshold.add_trace(
    go.Scatter(
        x=df_threshold[
            "threshold"
        ],
        y=df_threshold[
            "f1"
        ],
        mode="lines+markers",
        name="F1",
        line=dict(
            color="#7A6FF0"
        )
    )
)

fig_threshold.add_vline(
    x=0.75,
    line_dash="dash",
    line_color=TEAL,
    annotation_text=(
        "Threshold operativo 0.75"
    ),
    annotation_position="top right"
)

fig_threshold.update_layout(
    title=(
        "Trade-off entre métricas "
        "según threshold"
    ),
    xaxis_title="Threshold",
    yaxis_title="Valor",
    yaxis=dict(
        tickformat=".0%",
        range=[0, 1]
    ),
    height=440,
    template="plotly_white"
)

st.plotly_chart(
    fig_threshold,
    use_container_width=True
)


# ============================================================
# 12. MATRIZ OOF PARA ANÁLISIS DE THRESHOLD
# ============================================================

st.markdown(
    "### Matriz de confusión OOF · "
    "Análisis de threshold"
)

st.caption(
    "Esta matriz corresponde a las predicciones "
    "out-of-fold utilizadas para seleccionar el threshold. "
    "La evaluación final independiente se presenta "
    "posteriormente sobre el holdout."
)

tn_oof = int(
    fila_threshold[
        "tn"
    ]
)

fp_oof = int(
    fila_threshold[
        "fp"
    ]
)

fn_oof = int(
    fila_threshold[
        "fn"
    ]
)

tp_oof = int(
    fila_threshold[
        "tp"
    ]
)


fig_cm_oof = crear_matriz_confusion(
    tn=tn_oof,
    fp=fp_oof,
    fn=fn_oof,
    tp=tp_oof,
    titulo=(
        "Matriz de confusión OOF · "
        f"Threshold {threshold_seleccionado:.2f}"
    )
)

st.plotly_chart(
    fig_cm_oof,
    use_container_width=True
)


st.markdown(
    f"""
    <div class="method-box">

    Con threshold
    <strong>
    {threshold_seleccionado:.2f}
    </strong>,
    el modelo identifica
    <strong>
    {tp_oof:,}
    </strong>
    incumplimientos y deja sin detectar
    <strong>
    {fn_oof:,}
    </strong>.

    Se generan
    <strong>
    {fp_oof:,}
    </strong>
    alertas preventivas adicionales
    y se priorizan
    <strong>
    {int(fila_threshold['casos_priorizados']):,}
    </strong>
    créditos, equivalentes al
    <strong>
    {fila_threshold['porcentaje_priorizado']:.2%}
    </strong>
    del universo OOF analizado.

    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


# ============================================================
# 13. HOLDOUT FINAL
# ============================================================

st.markdown(
    "## 4. Evaluación final independiente"
)

st.write(
    "El modelo y el threshold fueron congelados "
    "antes de utilizar el conjunto holdout."
)


# ------------------------------------------------------------
# KPIs finales del holdout
# ------------------------------------------------------------

h1, h2, h3, h4 = st.columns(4)

with h1:
    st.metric(
        "TP final",
        numero(
            fila_holdout[
                "tp"
            ]
        )
    )

with h2:
    st.metric(
        "FN final",
        numero(
            fila_holdout[
                "fn"
            ]
        )
    )

with h3:
    st.metric(
        "FP final",
        numero(
            fila_holdout[
                "fp"
            ]
        )
    )

with h4:
    st.metric(
        "TN final",
        numero(
            fila_holdout[
                "tn"
            ]
        )
    )


# ------------------------------------------------------------
# Matriz final del holdout
# ------------------------------------------------------------

st.markdown(
    "### Matriz de confusión final · Holdout"
)

fig_cm_holdout = crear_matriz_confusion(
    tn=int(
        fila_holdout[
            "tn"
        ]
    ),
    fp=int(
        fila_holdout[
            "fp"
        ]
    ),
    fn=int(
        fila_holdout[
            "fn"
        ]
    ),
    tp=int(
        fila_holdout[
            "tp"
        ]
    ),
    titulo=(
        "Matriz de confusión final · "
        "Threshold 0.75"
    )
)

st.plotly_chart(
    fig_cm_holdout,
    use_container_width=True
)


st.markdown(
    f"""
    <div class="method-box">

    Esta es la matriz que representa
    el <strong>resultado final de FinPredict</strong>
    sobre datos no utilizados durante la selección
    del modelo ni del threshold.

    Se identificaron correctamente
    <strong>
    {int(fila_holdout['tp']):,}
    </strong>
    incumplimientos futuros,
    mientras que
    <strong>
    {int(fila_holdout['fn']):,}
    </strong>
    no fueron detectados.

    La cartera priorizada fue de
    <strong>
    {int(fila_holdout['casos_priorizados']):,}
    </strong>
    créditos,
    equivalente al
    <strong>
    {fila_holdout['porcentaje_priorizado']:.2%}
    </strong>
    del holdout.

    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# ROC y Precision-Recall
# ------------------------------------------------------------

y_real = df_predicciones[
    "incumplimiento_90d"
].astype(int)

y_prob = df_predicciones[
    "probabilidad_riesgo"
]


fpr, tpr, _ = roc_curve(
    y_real,
    y_prob
)

roc_auc = roc_auc_score(
    y_real,
    y_prob
)


precision_curve, recall_curve, _ = (
    precision_recall_curve(
        y_real,
        y_prob
    )
)

pr_auc = average_precision_score(
    y_real,
    y_prob
)

prevalencia = y_real.mean()


col_roc, col_pr = st.columns(2)


with col_roc:

    fig_roc = go.Figure()

    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=(
                f"Regresión Logística "
                f"(AUC={roc_auc:.4f})"
            ),
            line=dict(
                color=TEAL,
                width=3
            )
        )
    )

    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Clasificador aleatorio",
            line=dict(
                color=GRAY,
                dash="dash"
            )
        )
    )

    fig_roc.update_layout(
        title=(
            "Curva ROC · Holdout final"
        ),
        xaxis_title=(
            "Tasa de falsos positivos"
        ),
        yaxis_title=(
            "Tasa de verdaderos positivos"
        ),
        template="plotly_white",
        height=450
    )

    st.plotly_chart(
        fig_roc,
        use_container_width=True
    )


with col_pr:

    fig_pr = go.Figure()

    fig_pr.add_trace(
        go.Scatter(
            x=recall_curve,
            y=precision_curve,
            mode="lines",
            name=(
                f"Regresión Logística "
                f"(PR-AUC={pr_auc:.4f})"
            ),
            line=dict(
                color=NAVY,
                width=3
            )
        )
    )

    fig_pr.add_hline(
        y=prevalencia,
        line_dash="dash",
        line_color=GRAY,
        annotation_text=(
            f"Baseline {prevalencia:.4f}"
        )
    )

    fig_pr.update_layout(
        title=(
            "Curva Precision-Recall · "
            "Holdout final"
        ),
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        height=450
    )

    st.plotly_chart(
        fig_pr,
        use_container_width=True
    )


st.markdown(
    f"""
    <div class="executive-box">

    <strong>
    Generalización final
    </strong><br><br>

    ROC-AUC:
    <strong>
    {roc_auc:.4f}
    </strong><br>

    PR-AUC:
    <strong>
    {pr_auc:.4f}
    </strong><br>

    Baseline PR-AUC:
    <strong>
    {prevalencia:.4f}
    </strong><br><br>

    Los resultados del holdout son consistentes
    con la evidencia OOF, lo que indica
    estabilidad temporal en el escenario
    sintético evaluado.

    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


# ============================================================
# 14. SIMULACIÓN A/B
# ============================================================

st.markdown(
    "## 5. Simulación A/B e impacto operativo"
)

st.write(
    "Simulación académica utilizada para ilustrar "
    "cómo podría evaluarse una intervención preventiva "
    "basada en FinPredict."
)


df_ab_plot = df_ab.copy()

if (
    "tasa_incumplimiento_%"
    in df_ab_plot.columns
):

    df_ab_plot[
        "tasa_grafica"
    ] = (
        df_ab_plot[
            "tasa_incumplimiento_%"
        ]
    )

else:

    df_ab_plot[
        "tasa_grafica"
    ] = (
        df_ab_plot[
            "tasa_incumplimiento"
        ] * 100
    )


fig_ab = px.bar(
    df_ab_plot,
    x="grupo",
    y="tasa_grafica",
    text="tasa_grafica",
    color="grupo",
    color_discrete_sequence=[
        NAVY,
        TEAL
    ]
)

fig_ab.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_ab.update_layout(
    title=(
        "Tasa de incumplimiento por grupo"
    ),
    xaxis_title="",
    yaxis_title=(
        "Tasa de incumplimiento"
    ),
    yaxis=dict(
        ticksuffix="%"
    ),
    template="plotly_white",
    height=430,
    showlegend=False
)

st.plotly_chart(
    fig_ab,
    use_container_width=True
)


# ------------------------------------------------------------
# Chi-cuadrada
# ------------------------------------------------------------

tabla_ab = []

for _, fila in df_ab.iterrows():

    incumple = int(
        fila[
            "incumplimientos"
        ]
    )

    no_incumple = int(
        fila[
            "creditos"
        ]
        - fila[
            "incumplimientos"
        ]
    )

    tabla_ab.append(
        [
            no_incumple,
            incumple
        ]
    )


chi2, p_value, _, _ = (
    chi2_contingency(
        tabla_ab
    )
)


a1, a2, a3 = st.columns(3)

with a1:
    st.metric(
        "Chi-cuadrada",
        f"{chi2:.4f}"
    )

with a2:
    st.metric(
        "p-value",
        f"{p_value:.4f}"
    )

with a3:
    st.metric(
        "Créditos simulados",
        f"{int(df_ab['creditos'].sum()):,}"
    )


# ------------------------------------------------------------
# Impacto operativo
# ------------------------------------------------------------

st.markdown(
    "### Impacto operativo simulado"
)

st.dataframe(
    df_impacto,
    hide_index=True,
    use_container_width=True
)


st.markdown(
    """
    <div class="method-box">

    <strong>
    Advertencia metodológica.
    </strong><br><br>

    La prueba A/B es una simulación académica
    construida bajo un supuesto de efectividad
    de intervención.

    El p-value obtenido no constituye evidencia
    de que FinPredict genere este impacto
    en producción.

    El efecto causal deberá validarse mediante
    un experimento controlado real.

    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


# ============================================================
# 15. RECOMENDACIÓN FINAL
# ============================================================

st.markdown(
    "## 6. Recomendación"
)

st.markdown(
    """
    <div class="executive-box">

    <strong>
    GO condicionado para piloto controlado
    </strong><br><br>

    La evidencia técnica respalda avanzar
    a una validación operativa controlada.

    FinPredict presenta:

    <ul>
        <li>
        estabilidad entre validación temporal
        y holdout;
        </li>

        <li>
        desempeño superior al baseline;
        </li>

        <li>
        un threshold operacionalmente defendible;
        </li>

        <li>
        capacidad de priorizar una fracción
        reducida de la cartera;
        </li>

        <li>
        evidencia estadística favorable
        en PR-AUC.
        </li>
    </ul>

    El modelo debe emplearse como herramienta
    de <strong>
    priorización preventiva con supervisión humana
    </strong>.

    No se recomienda utilizarlo para rechazos
    de crédito, modificaciones contractuales
    o decisiones adversas automáticas.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 16. EVIDENCIA Y TRAZABILIDAD
# ============================================================

with st.expander(
    "Ver evidencia técnica y archivos fuente"
):

    st.markdown(
        "### Resumen ejecutivo"
    )

    st.dataframe(
        df_resumen,
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        "### Holdout final"
    )

    st.dataframe(
        df_holdout,
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        "### Validación cruzada temporal"
    )

    st.dataframe(
        df_cv,
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        "### Análisis de threshold"
    )

    st.dataframe(
        df_threshold,
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        "### Comparación estadística"
    )

    st.dataframe(
        df_bootstrap,
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        "### Simulación A/B"
    )

    st.dataframe(
        df_ab,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# 17. PIE DE PÁGINA
# ============================================================

st.markdown(
    """
    <div class="fin-footer">

    FinComercial, S.A. |
    FinPredict |
    Gestión de Proyectos de Inteligencia Artificial |
    Actividad 6<br><br>

    Datos sintéticos con fines académicos.
    Los resultados no representan decisiones reales
    sobre clientes.

    </div>
    """,
    unsafe_allow_html=True
)
