"""
Dashboard de EDA (Exploratory Data Analysis) con Streamlit
Usa: pandas, numpy, matplotlib, seaborn, plotly, scipy, openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats

try:
    from groq import Groq
except ImportError:
    Groq = None

# Configuración de la página
st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo para ocultar menú de Streamlit y mejorar aspecto
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    h1 { color: #1f77b4; }
    h2, h3 { color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Dashboard de Análisis Exploratorio (EDA)")
st.markdown("Carga un archivo CSV o Excel para explorar tus datos.")

# Sidebar: carga de archivos
with st.sidebar:
    st.header("📁 Carga de datos")
    uploaded_file = st.file_uploader(
        "Selecciona un archivo",
        type=["csv", "xlsx", "xls"],
        help="Formatos: CSV, Excel (.xlsx, .xls)"
    )

    st.divider()
    st.header("🤖 Asistente de análisis (Groq)")
    groq_api_key = st.sidebar.text_input(
        "API Key de Groq",
        type="password",
        placeholder="gsk_...",
        help="Obtén tu API key en https://console.groq.com. Se usa solo para el asistente de análisis."
    )

    st.divider()
    st.header("⚙️ Opciones")
    decimal_places = st.slider("Decimales en tablas", 0, 4, 2)
    theme = st.selectbox("Tema de gráficos", ["darkgrid", "whitegrid", "white", "dark"])

# Cargar datos
def load_data(file):
    if file is None:
        return None
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file, engine="openpyxl" if file.name.endswith(".xlsx") else "xlrd")
    except Exception as e:
        st.error(f"Error al cargar: {e}")
        return None
    return None

df = load_data(uploaded_file)

if df is None:
    st.info("👆 Sube un archivo CSV o Excel desde la barra lateral para comenzar.")
    st.stop()

# Barra en sidebar: cantidad de muestras a usar
total_filas = len(df)
n_samples = st.sidebar.slider(
    "Cantidad de muestras",
    min_value=1,
    max_value=total_filas,
    value=total_filas,
    step=1,
    help="Número de filas a tener en cuenta en el análisis (se toman desde el inicio del dataset)."
)
df = df.head(n_samples)
if n_samples < total_filas:
    st.sidebar.caption(f"Usando {n_samples:,} de {total_filas:,} filas.")
else:
    st.sidebar.caption(f"Usando todas las filas ({len(df):,}).")

# Variables útiles para todas las pestañas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# Tabs principales: Cualitativo | Cuantitativo | Gráfico | Asistente
tab_cualitativo, tab_cuantitativo, tab_grafico, tab_asistente = st.tabs([
    "📋 Cualitativo", "📈 Cuantitativo", "📉 Gráfico", "🤖 Asistente de análisis"
])

# ========== CUALITATIVO ==========
with tab_cualitativo:
    st.header("Análisis cualitativo")
    st.caption("Estructura del dataset, tipos de datos y variables categóricas.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filas", f"{len(df):,}")
    with col2:
        st.metric("Columnas", len(df.columns))
    with col3:
        st.metric("Memoria (aprox.)", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    st.subheader("Tipos de datos")
    dtype_df = pd.DataFrame({
        "Columna": df.columns,
        "Tipo": [str(d) for d in df.dtypes],
        "No nulos": df.count().values
    })
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.subheader("Primeras filas")
    st.dataframe(df.head(20), use_container_width=True)

    if cat_cols:
        st.subheader("Frecuencias (variables cualitativas)")
        col_cual = st.selectbox("Columna categórica", cat_cols, key="col_cual")
        frec = df[col_cual].value_counts().reset_index()
        frec.columns = [col_cual, "Frecuencia"]
        frec["%"] = (frec["Frecuencia"] / frec["Frecuencia"].sum() * 100).round(2)
        st.dataframe(frec, use_container_width=True, hide_index=True)
        st.caption(f"Moda: **{df[col_cual].mode().iloc[0]}** (aparece {int((df[col_cual] == df[col_cual].mode().iloc[0]).sum())} veces).")
    else:
        st.info("No hay columnas categóricas (object/category) para frecuencias.")

# ========== CUANTITATIVO ==========
with tab_cuantitativo:
    st.header("Análisis cuantitativo")
    st.caption("Estadísticas numéricas, valores faltantes y tests.")

    st.subheader("Estadísticas descriptivas")
    st.dataframe(
        df.describe(include="all").round(decimal_places),
        use_container_width=True
    )

    st.subheader("Valores faltantes (tabla)")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Columna": df.columns,
        "Faltantes": missing.values,
        "%": missing_pct.values
    })
    missing_df = missing_df[missing_df["Faltantes"] > 0].sort_values("Faltantes", ascending=False)
    if len(missing_df) == 0:
        st.success("No hay valores faltantes.")
    else:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

    if numeric_cols:
        st.subheader("Test de normalidad (Shapiro-Wilk)")
        st.caption("p-value > 0.05 sugiere que los datos podrían seguir una distribución normal.")
        norm_results = []
        for col in numeric_cols[:10]:
            sample = df[col].dropna()
            if len(sample) >= 3 and len(sample) <= 5000:
                stat, pval = stats.shapiro(sample)
                norm_results.append({"Columna": col, "Estadístico": round(stat, 4), "p-value": round(pval, 4)})
        if norm_results:
            st.dataframe(pd.DataFrame(norm_results), use_container_width=True, hide_index=True)
        else:
            st.info("No hay suficientes datos numéricos o muestras válidas para el test.")

        if len(numeric_cols) >= 2:
            st.subheader("Correlación de Pearson (scipy)")
            c1 = st.selectbox("Columna 1", numeric_cols, key="corr1")
            c2 = st.selectbox("Columna 2", numeric_cols, key="corr2")
            if c1 != c2:
                clean = df[[c1, c2]].dropna()
                r, p = stats.pearsonr(clean[c1], clean[c2])
                st.write(f"**Coeficiente r:** {r:.4f}  |  **p-value:** {p:.4f}")

# ========== GRÁFICO ==========
with tab_grafico:
    st.header("Análisis gráfico")
    st.caption("Visualizaciones con Matplotlib, Seaborn y Plotly.")
    sns.set_theme(style=theme)

    # Gráfico de valores faltantes
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Columna": df.columns,
        "Faltantes": missing.values,
        "%": missing_pct.values
    })
    missing_df = missing_df[missing_df["Faltantes"] > 0].sort_values("Faltantes", ascending=False)
    if len(missing_df) > 0:
        st.subheader("Valores faltantes (% por columna)")
        fig_miss, ax = plt.subplots(figsize=(10, max(4, len(missing_df) * 0.3)))
        sns.barplot(data=missing_df, y="Columna", x="%", ax=ax, palette="viridis")
        ax.set_xlabel("% faltantes")
        plt.tight_layout()
        st.pyplot(fig_miss)
        plt.close()

    # Gráficos estáticos (Matplotlib / Seaborn)
    st.subheader("Gráficos estáticos (Matplotlib / Seaborn)")
    if not numeric_cols:
        st.warning("No hay columnas numéricas para graficar.")
    else:
        col_x = st.selectbox("Eje X (distribución)", numeric_cols, key="x_dist")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.hist(df[col_x].dropna(), bins=30, edgecolor="black", alpha=0.7)
        ax1.set_title(f"Distribución de {col_x}")
        ax1.set_xlabel(col_x)
        st.pyplot(fig1)
        plt.close()

        if len(numeric_cols) >= 2:
            col_y = st.selectbox("Eje Y (dispersión)", numeric_cols, key="y_scatter")
            if col_y != col_x:
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax2, alpha=0.6)
                ax2.set_title(f"{col_x} vs {col_y}")
                st.pyplot(fig2)
                plt.close()

        col_box = st.selectbox("Boxplot (columna)", numeric_cols, key="box")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.boxplot(y=df[col_box], ax=ax3)
        ax3.set_ylabel(col_box)
        ax3.set_title(f"Boxplot de {col_box}")
        st.pyplot(fig3)
        plt.close()

    # Gráficos interactivos (Plotly)
    st.subheader("Gráficos interactivos (Plotly)")
    if not numeric_cols:
        st.warning("No hay columnas numéricas.")
    else:
        plot_type = st.radio("Tipo de gráfico", ["Histograma", "Dispersión", "Barras", "Box"], horizontal=True, key="plot_type")

        if plot_type == "Histograma":
            col_hist = st.selectbox("Columna", numeric_cols, key="plotly_hist")
            fig = px.histogram(df, x=col_hist, nbins=40, title=f"Distribución de {col_hist}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Dispersión":
            col_x = st.selectbox("Eje X", numeric_cols, key="px_x")
            col_y = st.selectbox("Eje Y", numeric_cols, key="px_y")
            color_col = st.selectbox("Color (opcional)", [None] + cat_cols + numeric_cols, key="px_color")
            fig = px.scatter(df, x=col_x, y=col_y, color=color_col, hover_data=df.columns.tolist()[:5])
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Barras":
            col_bar = st.selectbox("Columna para barras", numeric_cols + cat_cols, key="px_bar")
            if col_bar in cat_cols or df[col_bar].dtype in ["object", "category"]:
                counts = df[col_bar].value_counts().reset_index()
                counts.columns = [col_bar, "count"]
                fig = px.bar(counts, x=col_bar, y="count", title=f"Frecuencia de {col_bar}")
            else:
                fig = px.histogram(df, x=col_bar, title=f"Distribución de {col_bar}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Box":
            col_box = st.selectbox("Columna", numeric_cols, key="px_box")
            group_col = st.selectbox("Agrupar por (opcional)", [None] + cat_cols, key="px_box_group")
            fig = px.box(df, x=group_col, y=col_box, title=f"Boxplot de {col_box}") if group_col else px.box(df, y=col_box, title=f"Boxplot de {col_box}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Matriz de correlación (gráfico)
    st.subheader("Matriz de correlación (heatmap)")
    if len(numeric_cols) < 2:
        st.warning("Se necesitan al menos 2 columnas numéricas para la correlación.")
    else:
        corr = df[numeric_cols].corr().round(decimal_places)
        fig_corr, ax_corr = plt.subplots(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols) * 0.6)))
        sns.heatmap(corr, annot=True, fmt=f".{decimal_places}f", cmap="Blues", vmin=-1, vmax=1, ax=ax_corr, square=True)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close()

# ========== ASISTENTE DE ANÁLISIS (GROQ + LLAMA 3.3) ==========
def _build_eda_summary(df: pd.DataFrame, numeric_cols: list, cat_cols: list, decimal_places: int = 2) -> str:
    """Construye un resumen en texto del EDA para enviar al LLM."""
    lines = []
    lines.append(f"Dataset: {len(df)} filas, {len(df.columns)} columnas.")
    lines.append(f"Columnas: {list(df.columns)}")
    lines.append("")
    lines.append("Tipos de datos:")
    for c in df.columns:
        lines.append(f"  - {c}: {df[c].dtype}, no nulos: {df[c].count()}")
    lines.append("")
    if numeric_cols:
        lines.append("Estadísticas descriptivas (numéricas):")
        lines.append(df[numeric_cols].describe().round(decimal_places).to_string())
        lines.append("")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        lines.append("Valores faltantes:")
        for c in missing.index:
            pct = (df[c].isna().sum() / len(df) * 100).round(2)
            lines.append(f"  - {c}: {missing[c]} ({pct}%)")
        lines.append("")
    if cat_cols:
        lines.append("Resumen categóricas (únicos / moda):")
        for c in cat_cols[:15]:
            uniq = df[c].nunique()
            mode = df[c].mode()
            mode_val = mode.iloc[0] if len(mode) else "N/A"
            lines.append(f"  - {c}: {uniq} valores únicos, moda: {mode_val}")
        lines.append("")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        lines.append("Correlaciones más fuertes (|r| > 0.3):")
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                r = corr.iloc[i, j]
                if abs(r) > 0.3:
                    pairs.append((numeric_cols[i], numeric_cols[j], round(r, decimal_places)))
        for a, b, r in sorted(pairs, key=lambda x: -abs(x[2]))[:15]:
            lines.append(f"  - {a} vs {b}: r = {r}")
        if not pairs:
            lines.append("  (ninguna con |r| > 0.3)")
    return "\n".join(lines)


with tab_asistente:
    st.header("Asistente de análisis con LLM")
    st.caption("El modelo Llama 3.3 70B (Groq) describe los hallazgos del EDA a partir del resumen de los datos.")

    if Groq is None:
        st.warning("Instala el paquete **groq** para usar el asistente: `pip install groq`")
        st.stop()

    if not groq_api_key or not groq_api_key.strip():
        st.info("Introduce tu **API Key de Groq** en la barra lateral para generar el análisis.")
        st.markdown("Obtén una API key gratuita en [console.groq.com](https://console.groq.com).")
    else:
        if st.button("Generar análisis con LLM", type="primary"):
            with st.spinner("Construyendo resumen del EDA y consultando al modelo..."):
                try:
                    summary = _build_eda_summary(df, numeric_cols, cat_cols, decimal_places)
                    client = Groq(api_key=groq_api_key.strip())
                    prompt = (
                        "Eres un analista de datos experto. A continuación se te proporciona un resumen "
                        "del análisis exploratorio (EDA) de un dataset. Describe de forma clara y concisa "
                        "los hallazgos principales: estructura de los datos, calidad (valores faltantes), "
                        "estadísticas relevantes de variables numéricas, patrones en categóricas y correlaciones "
                        "destacadas. Responde en español y en formato legible (párrafos o listas).\n\n"
                        "---\nResumen del EDA:\n---\n" + summary
                    )
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "Eres un analista de datos. Respondes siempre en español."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=2048,
                        temperature=0.3
                    )
                    result = response.choices[0].message.content
                    st.subheader("Hallazgos del EDA")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Error al llamar a la API de Groq: {e}")
                    st.caption("Comprueba que la API key sea correcta y que tengas acceso al modelo.")

st.sidebar.divider()
st.sidebar.caption("EDA Dashboard · pandas, numpy, matplotlib, seaborn, plotly, scipy")
