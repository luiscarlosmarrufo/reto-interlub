import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly import colors
import nltk
from nltk.corpus import stopwords

# ------------------- PALETA Y ESTILO GLOBAL -------------------
PRIMARY_RED = "#FF0000"   # rojo Interlub
DARK_GREY   = "#333333"   # texto principal
LIGHT_GREY  = "#F2F2F2"   # fondo sidebar / bloques
BACKGROUND  = "#FFFFFF"   # fondo principal

# Descargar stopwords si es necesario
try:
    STOP_WORDS_ES = stopwords.words("spanish")
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS_ES = stopwords.words("spanish")

# Configurar la página
st.set_page_config(
    page_title="Recomendador Inteligente de Grasas Interlub",
    page_icon="⚙",
    layout="wide"
)

# === CSS GLOBAL: VISIBILIDAD + ESTILO INTERLUB ===
st.markdown(
    f"""
    <style>
    :root {{
        --primary-color: {PRIMARY_RED};
        --text-color: {DARK_GREY};
        --bg-color: {BACKGROUND};
        --sidebar-bg: {LIGHT_GREY};
    }}

    /* Fondo general y tipografía */
    .stApp {{
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    /* Quitar barra superior oscura de Streamlit */
    header[data-testid="stHeader"] {{
        background-color: var(--bg-color);
        color: var(--text-color);
        box-shadow: none;
    }}
    header[data-testid="stHeader"] * {{
        color: var(--text-color) !important;
    }}

    /* SIDEBAR: más contraste, todo legible */
    section[data-testid="stSidebar"] {{
        background-color: var(--sidebar-bg);
        border-right: 1px solid #E0E0E0;
    }}
    section[data-testid="stSidebar"] * {{
        color: {DARK_GREY} !important;
        font-size: 0.93rem;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        font-weight: 700;
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
    }}

    /* Inputs del sidebar SIEMPRE blancos */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {{
        background-color: #FFFFFF !important;
        color: {DARK_GREY} !important;
        border-radius: 8px !important;
    }}
    section[data-testid="stSidebar"] select {{
        background-color: #FFFFFF !important;
        color: {DARK_GREY} !important;
        border-radius: 8px !important;
    }}

    /* Select / multiselect blancos (baseweb) */
    div[data-baseweb="select"] > div {{
        background-color: #FFFFFF !important;
        color: {DARK_GREY} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] [class*="placeholder"] {{
        color: #777777 !important;
    }}

    /* Menú desplegable del select */
    div[data-baseweb="popover"] div[role="listbox"] {{
        background-color: #FFFFFF !important;
        color: {DARK_GREY} !important;
    }}

    /* Botones tipo CTA rojo – texto blanco */
    .stButton>button, .stDownloadButton>button {{
        background-color: {PRIMARY_RED};
        border-radius: 999px;
        border: none;
        padding: 0.45rem 1.4rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .stButton>button, .stDownloadButton>button,
    .stButton>button *, .stDownloadButton>button * {{
        color: #FFFFFF !important;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        filter: brightness(0.92);
    }}

    /* Títulos principales tipo Interlub */
    h1, h2, h3, h4 {{
        color: var(--text-color);
        font-weight: 800;
    }}

    .hero-title {{
        font-size: 4rem;
        font-weight: 900;
        line-height: 1.05;
        color: {DARK_GREY};
        margin-bottom: 0.5rem;
    }}

    .hero-subtitle {{
        font-size: 1.25rem;
        color: #555555;
        margin-bottom: 1.5rem;
        max-width: 720px;
    }}

    .hero-dot {{
        color: {PRIMARY_RED};
    }}

    /* Expanders (banners) siempre claros */
    div[data-testid="stExpander"] > details > summary {{
        background-color: #F7F7F7 !important;
        color: {DARK_GREY} !important;
        border-radius: 12px !important;
    }}
    div[data-testid="stExpander"] > details > summary:hover {{
        background-color: #ECECEC !important;
    }}

    /* Métricas más limpias */
    div[data-testid="stMetricValue"] {{
        color: {DARK_GREY} !important;
        font-weight: 700;
    }}
    div[data-testid="stMetricLabel"] {{
        color: #666666 !important;
    }}
    div[data-testid="stMetricDelta"] {{
        color: #666666 !important;
    }}
    /* Asegurar que el texto dentro de metrics sea visible */
    div[data-testid="stMetric"] * {{
        color: {DARK_GREY} !important;
    }}
    div[data-testid="stMetric"] label {{
        color: #666666 !important;
    }}

    .stAlert > div {{
        border-radius: 10px;
    }}

    /* ========== TABLAS Y DATAFRAMES - FORZAR TEMA CLARO ========== */
    .stDataFrame, .stTable {{
        background-color: #FFFFFF !important;
    }}

    /* Headers de tabla */
    .stDataFrame thead tr th {{
        background-color: {LIGHT_GREY} !important;
        color: {DARK_GREY} !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #E0E0E0 !important;
    }}

    /* Celdas de tabla */
    .stDataFrame tbody tr td {{
        background-color: #FFFFFF !important;
        color: {DARK_GREY} !important;
        border-bottom: 1px solid #F0F0F0 !important;
    }}

    /* Hover en filas */
    .stDataFrame tbody tr:hover td {{
        background-color: #F9F9F9 !important;
    }}

    /* Forzar texto oscuro en todas las tablas */
    div[data-testid="stDataFrame"] {{
        background-color: #FFFFFF !important;
    }}
    div[data-testid="stDataFrame"] * {{
        color: {DARK_GREY} !important;
    }}

    /* ========== INPUTS DEL ÁREA PRINCIPAL ========== */
    /* Number inputs, text inputs, text areas */
    input[type="number"],
    input[type="text"],
    textarea {{
        background-color: #FFFFFF !important;
        color: {DARK_GREY} !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
    }}

    /* Labels de inputs */
    label {{
        color: {DARK_GREY} !important;
    }}

    /* Sliders */
    div[data-baseweb="slider"] {{
        background-color: transparent !important;
    }}
    div[data-baseweb="slider"] * {{
        color: {DARK_GREY} !important;
    }}

    /* Radio buttons */
    div[role="radiogroup"] label {{
        background-color: #FFFFFF !important;
        color: {DARK_GREY} !important;
        border-radius: 8px !important;
    }}
    div[role="radiogroup"] {{
        background-color: transparent !important;
    }}

    /* Checkboxes */
    div[data-testid="stCheckbox"] label {{
        color: {DARK_GREY} !important;
    }}

    /* Info, success, warning, error boxes */
    .stAlert {{
        background-color: #F0F8FF !important;
        color: {DARK_GREY} !important;
    }}
    div[data-testid="stNotification"] {{
        background-color: #FFFFFF !important;
        color: {DARK_GREY} !important;
    }}

    /* Texto general - forzar oscuro */
    p, span, div {{
        color: {DARK_GREY};
    }}

    /* Captions */
    .caption, small {{
        color: #666666 !important;
    }}

    /* Plotly charts - asegurar fondo blanco */
    .js-plotly-plot {{
        background-color: #FFFFFF !important;
    }}

    /* Markdown */
    .stMarkdown {{
        color: {DARK_GREY} !important;
    }}

    /* Code blocks */
    code {{
        background-color: #F7F7F7 !important;
        color: {DARK_GREY} !important;
        border-radius: 4px !important;
    }}

    pre {{
        background-color: #F7F7F7 !important;
        color: {DARK_GREY} !important;
        border-radius: 8px !important;
    }}

    /* ========== IMÁGENES ========== */
    /* Asegurar que las imágenes se muestren correctamente */
    div[data-testid="stImage"] {{
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}

    div[data-testid="stImage"] img {{
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        max-width: 100%;
        height: auto;
    }}

    /* Contenedor de columnas no debe ocultar contenido */
    div[data-testid="column"] {{
        visibility: visible !important;
        opacity: 1 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ==================== FUNCIONES DE CARGA Y PREPROCESAMIENTO ====================

@st.cache_data
def load_and_preprocess_data(csv_path, csv_competencia_path):
    """
    Cargar y preprocesar datos de grasas propias y de la competencia.
    Retorna todos los componentes necesarios para el sistema de recomendación.
    """
    try:
        # Cargar datos propios
        df_raw = pd.read_csv(csv_path, encoding="utf-8")

        # Cargar datos de la competencia
        df_competencia_raw = pd.read_csv(csv_competencia_path, encoding="utf-8", encoding_errors="ignore")
        df_competencia_raw["subtitulo"] = df_competencia_raw["descripcion"]
        df_competencia_raw["descripcion"] = df_competencia_raw["Descripcin"]
        df_competencia_raw.drop(["Descripcin"], axis=1, inplace=True)

        # Copiar y preparar datos propios
        df = df_raw.copy()
        df["codigoGrasa"] = [f"Grasa_{i+1}" for i in range(len(df))]
        df["esCompetencia"] = False
        df["empresaOrigen"] = "Interlub"

        # Preparar datos de competencia
        df_competencia = df_competencia_raw.copy()
        df_competencia["codigoGrasa"] = [f"Competencia_{i+1}" for i in range(len(df_competencia))]
        df_competencia["esCompetencia"] = True
        df_competencia["empresaOrigen"] = "Competencia"

        # Combinar ambos dataframes
        df = pd.concat([df, df_competencia], ignore_index=True)
        
        # Columnas a eliminar para features
        cols_drop_features = [
            "idDatosGrasas",
            "Indice de Carga-Desgaste",
            "categoria",
        ]
        
        df_v = df.drop(columns=cols_drop_features)
        df_v.set_index("codigoGrasa", inplace=True)
        
        # ===== PREPARAR DATOS DE TEXTO =====
        cols_texto = ["subtitulo", "descripcion", "beneficios", "aplicaciones"]
        descripcion_grasas = (
            df_v[cols_texto + ["Registro NSF"]]
            .copy()
            .reset_index()
        )
        
        # Registro NSF como binario
        descripcion_grasas["Registro NSF"] = (
            descripcion_grasas["Registro NSF"].notnull().astype(int)
        )
        
        # Rellenar NaN en columnas de texto
        for c in cols_texto:
            descripcion_grasas[c] = descripcion_grasas[c].fillna("")
        
        # Crear "soup" - combinación de todos los textos
        descripcion_grasas["soup"] = (
            descripcion_grasas["subtitulo"] + " " +
            descripcion_grasas["descripcion"] + " " +
            descripcion_grasas["beneficios"] + " " +
            descripcion_grasas["aplicaciones"]
        ).str.strip()
        
        # ===== PREPARAR FEATURES NUMÉRICAS =====
        df_features = df_v.copy()
        df_features["Registro NSF"] = df_features["Registro NSF"].notnull().astype(int)
        
        # Columnas que NO usaremos como features
        cols_drop_extra = ["Corrosión al Cobre", "Factor de Velocidad"]
        df_features = df_features.drop(columns=cols_drop_extra, errors='ignore')
        
        # One-hot encoding para categóricas
        df_features = pd.get_dummies(
            df_features,
            columns=["Aceite Base", "Espesante", "color", "textura"],
            drop_first=False
        )
        
        # Rellenar NaN con -99
        df_features = df_features.fillna(-99.0)
        
        # ===== CALCULAR SIMILITUD DE TEXTO (TF-IDF) =====
        tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            stop_words=STOP_WORDS_ES,
        )
        tfidf_matrix = tfidf.fit_transform(descripcion_grasas["soup"])
        cosine_sim_text = cosine_similarity(tfidf_matrix)
        
        # ===== CALCULAR SIMILITUD NUMÉRICA =====
        numeric_cols = df_features.select_dtypes(include="number").columns.tolist()
        df_numeric = df_features[numeric_cols].copy()
        df_numeric = df_numeric.fillna(df_numeric.median())
        
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(df_numeric)
        cosine_sim_numeric = cosine_similarity(numeric_scaled)
        
        # ===== CREAR ÍNDICES =====
        codigos = df_features.index.to_list()
        indices = pd.Series(range(len(codigos)), index=codigos)
        
        return {
            'df': df,
            'df_features': df_features,
            'descripcion_grasas': descripcion_grasas,
            'cosine_sim_text': cosine_sim_text,
            'cosine_sim_numeric': cosine_sim_numeric,
            'indices': indices,
            'numeric_cols': numeric_cols,
            'tfidf': tfidf,
            'scaler': scaler
        }
        
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None

# ==================== FUNCIONES DE RECOMENDACIÓN ====================

def recomendar_grasas_hibrido(
    codigo_grasa,
    data_dict,
    top_n=5,
    modo="hibrido",
    peso_texto=0.7,
    peso_numerico=0.3,
    grasa_personalizada=None,
):
    """
    Recomienda grasas similares usando el método híbrido.
    Si se proporciona grasa_personalizada, calcula similitud contra ella.
    """
    indices = data_dict['indices']
    df_features = data_dict['df_features']
    descripcion_grasas = data_dict['descripcion_grasas']
    cosine_sim_text = data_dict['cosine_sim_text']
    cosine_sim_numeric = data_dict['cosine_sim_numeric']
    scaler = data_dict['scaler']
    tfidf = data_dict['tfidf']

    # Si hay una grasa personalizada, calcular similitud contra ella
    if grasa_personalizada is not None:
        # Crear un DataFrame temporal con la grasa personalizada
        nueva_grasa_dict = {}

        # Primero, inicializar todas las columnas numéricas con -99.0
        for col in df_features.columns:
            if col in grasa_personalizada:
                nueva_grasa_dict[col] = grasa_personalizada[col]
            else:
                nueva_grasa_dict[col] = -99.0

        # Crear DataFrame temporal
        nueva_grasa_df = pd.DataFrame([nueva_grasa_dict])

        # Procesar columnas categóricas (one-hot encoding)
        # Las columnas categóricas originales son: Aceite Base, Espesante, color, textura
        categoricas = {
            'Aceite Base': grasa_personalizada.get('Aceite Base'),
            'Espesante': grasa_personalizada.get('Espesante'),
            'color': grasa_personalizada.get('color'),
            'textura': grasa_personalizada.get('textura')
        }

        # Para cada columna categórica, activar el one-hot correspondiente
        for cat_name, cat_value in categoricas.items():
            if cat_value is not None:
                # Buscar todas las columnas one-hot de esta categoría
                for col in df_features.columns:
                    if col.startswith(f"{cat_name}_"):
                        # Extraer el valor de la columna one-hot
                        # Ej: "Aceite Base_Mineral" -> "Mineral"
                        encoded_value = col.replace(f"{cat_name}_", "")
                        if encoded_value == str(cat_value):
                            nueva_grasa_df[col] = 1.0
                        else:
                            nueva_grasa_df[col] = 0.0

        # Obtener solo columnas numéricas
        df_numeric = df_features.select_dtypes(include="number")
        nueva_grasa_numeric = nueva_grasa_df[df_numeric.columns]

        # Escalar el vector
        nueva_grasa_scaled = scaler.transform(nueva_grasa_numeric)
        df_numeric_scaled = scaler.transform(df_numeric)

        # Calcular similitud numérica
        sim_vec_numeric = cosine_similarity(nueva_grasa_scaled, df_numeric_scaled)[0]

        # Para similitud de texto, usar descripción si existe
        if 'descripcion' in grasa_personalizada and grasa_personalizada['descripcion']:
            nueva_grasa_text = tfidf.transform([grasa_personalizada['descripcion']])
            tfidf_matrix = tfidf.transform(descripcion_grasas["soup"])
            sim_vec_text = cosine_similarity(nueva_grasa_text, tfidf_matrix)[0]
        else:
            sim_vec_text = np.zeros(len(df_features))

        # Combinar según el modo
        if modo == "texto":
            sim_vec = sim_vec_text
        elif modo == "numerico":
            sim_vec = sim_vec_numeric
        else:  # hibrido
            sim_vec = peso_texto * sim_vec_text + peso_numerico * sim_vec_numeric

        # Enumerar y ordenar
        sim_scores = list(enumerate(sim_vec))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]

    else:
        # Comportamiento original con grasa existente
        if codigo_grasa not in indices:
            raise ValueError(f"{codigo_grasa} no existe en la base.")

        idx = indices[codigo_grasa]

        # Seleccionar vector de similitud según el modo
        if modo == "texto":
            sim_vec = cosine_sim_text[idx]
        elif modo == "numerico":
            sim_vec = cosine_sim_numeric[idx]
        else:  # hibrido
            sim_vec = (
                peso_texto * cosine_sim_text[idx] +
                peso_numerico * cosine_sim_numeric[idx]
            )

        # Enumerar y ordenar por score
        sim_scores = list(enumerate(sim_vec))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Excluir la misma grasa y tomar top N
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

    indices_top = [s[0] for s in sim_scores]
    scores_top = [s[1] for s in sim_scores]

    # Crear DataFrame de resultados
    result = (
        df_features.iloc[indices_top]
        .reset_index()
        .rename(columns={"index": "codigoGrasa"})
    )
    result["Similitud"] = np.round(scores_top, 3)

    # Agregar subtítulo
    result = result.merge(
        descripcion_grasas[["codigoGrasa", "subtitulo"]],
        on="codigoGrasa",
        how="left"
    )

    return result


def recomendar_por_filtros(
    df,
    grado_nlgi=None,
    temp_min_trabajo=None,
    temp_max_trabajo=None,
    carga_timken_min=None,
    requiere_nsf=None,
    top_n=5
):
    """
    Recomienda grasas basándose en filtros de especificaciones técnicas.
    """
    base = df.copy()
    mask = pd.Series(True, index=base.index)
    
    # Aplicar filtros
    if grado_nlgi is not None:
        if isinstance(grado_nlgi, (list, tuple, set)):
            mask &= base["Grado NLGI Consistencia"].isin(grado_nlgi)
        else:
            mask &= base["Grado NLGI Consistencia"] == grado_nlgi
    
    if temp_min_trabajo is not None:
        mask &= base["Temperatura de Servicio °C, min"] <= temp_min_trabajo
    
    if temp_max_trabajo is not None:
        mask &= base["Temperatura de Servicio °C, max"] >= temp_max_trabajo
    
    if carga_timken_min is not None:
        mask &= base["Carga Timken Ok, lb"] >= carga_timken_min
    
    if requiere_nsf is not None:
        if int(requiere_nsf) == 1:
            mask &= base["Registro NSF"].notnull()
        else:
            mask &= base["Registro NSF"].isna()
    
    candidatos = base.loc[mask].copy()
    
    if candidatos.empty:
        return candidatos
    
    # Ordenar por cercanía a temp_max_trabajo si se especifica
    if temp_max_trabajo is not None:
        candidatos["score_temp"] = (
            candidatos["Temperatura de Servicio °C, max"] - temp_max_trabajo
        ).abs()
        candidatos = candidatos.sort_values("score_temp")
    
    return candidatos.head(top_n)


# ==================== INTERFAZ DE USUARIO ====================

def main():
    # ----- HERO PRINCIPAL ESTILO INTERLUB -----
    col_hero_text, col_hero_img = st.columns([2, 1])

    with col_hero_text:
        st.markdown(
            """
            <div class="hero-title">
            Tu proceso de selección de grasas no es común.<br>
            Nuestro recomendador tampoco<span class="hero-dot">.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hero-subtitle">'
            'Encuentra en segundos las grasas Interlub que mejor se ajustan a tus equipos, '
            'combinando experiencia técnica e inteligencia de datos.'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_hero_img:
        st.image("Tambor_Interlub_Flotado.png", use_container_width=True)
        
    # Cargar datos
    csv_path = "datos_grasas_Tec_limpio.csv"
    csv_competencia_path = "../../datos/FINAL/ID-grasas-Limpios.csv"

    data_dict = load_and_preprocess_data(csv_path, csv_competencia_path)
    
    if data_dict is None:
        st.error("No se pudieron cargar los datos. Verifica la ruta del archivo CSV.")
        st.info("Coloca tu archivo 'datos_grasas_Tec_limpio.csv' en el mismo directorio que este script.")
        return
    
    df = data_dict['df']
    df_features = data_dict['df_features']
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)

        st.image("Interlub_Logo_Negro.svg", width=160)

        st.markdown("</div>", unsafe_allow_html=True)

        st.header("Método de búsqueda")
        st.caption(
            "Elige si quieres partir de una grasa de referencia o de las condiciones de operación."
        )
        
        metodo_busqueda = st.radio(
            "Selecciona el método:",
            ["Búsqueda por Similitud", "Búsqueda por Filtros"],
            help="Similitud: encuentras alternativas a una grasa existente. Filtros: partes de requisitos técnicos."
        )
        
        # ===== BÚSQUEDA POR SIMILITUD =====
        if metodo_busqueda == "Búsqueda por Similitud":
            st.subheader("Configuración de similitud")

            # Opción para crear grasa personalizada
            tipo_referencia = st.radio(
                "Tipo de referencia:",
                ["Grasa existente", "Crear grasa personalizada"],
                help="Selecciona una grasa del catálogo o crea una con tus especificaciones."
            )

            grasa_referencia = None
            grasa_personalizada = None

            if tipo_referencia == "Grasa existente":
                # Crear opciones con etiqueta de origen
                opciones_grasas = []
                for codigo in df_features.index.tolist():
                    grasa_info = df[df["codigoGrasa"] == codigo].iloc[0]
                    origen = "🔴 Competencia" if grasa_info["esCompetencia"] else "⚙️ Interlub"
                    opciones_grasas.append(f"{codigo} ({origen})")

                grasa_referencia_display = st.selectbox(
                    "Grasa de referencia:",
                    options=opciones_grasas,
                    help="Usa la grasa que hoy empleas o la más cercana al caso que quieres resolver."
                )

                # Extraer el código de grasa sin la etiqueta
                grasa_referencia = grasa_referencia_display.split(" (")[0]

            else:  # Crear grasa personalizada
                st.markdown("**Define las especificaciones de tu grasa:**")
                st.info("💡 Usa modo 'numerico' para mejores resultados con grasa personalizada")

                grasa_personalizada = {}

                col1, col2 = st.columns(2)

                with col1:
                    grasa_personalizada["Grado NLGI Consistencia"] = st.number_input(
                        "Grado NLGI:",
                        min_value=0.0,
                        max_value=6.0,
                        value=2.0,
                        step=0.5
                    )

                    grasa_personalizada["Viscosidad del Aceite Base a 40°C. cSt"] = st.number_input(
                        "Viscosidad a 40°C (cSt):",
                        min_value=0.0,
                        value=100.0,
                        step=10.0
                    )

                    grasa_personalizada["Punto de Gota, °C"] = st.number_input(
                        "Punto de gota (°C):",
                        min_value=0.0,
                        value=180.0,
                        step=10.0
                    )

                with col2:
                    grasa_personalizada["Temperatura de Servicio °C, min"] = st.number_input(
                        "Temperatura mínima (°C):",
                        min_value=-100.0,
                        max_value=100.0,
                        value=-20.0,
                        step=5.0
                    )

                    grasa_personalizada["Temperatura de Servicio °C, max"] = st.number_input(
                        "Temperatura máxima (°C):",
                        min_value=-50.0,
                        max_value=300.0,
                        value=120.0,
                        step=10.0
                    )

                    grasa_personalizada["Carga Timken Ok, lb"] = st.number_input(
                        "Carga Timken (lb):",
                        min_value=0.0,
                        value=40.0,
                        step=5.0
                    )

                # Inicializar Registro NSF como 0 (no tiene)
                grasa_personalizada["Registro NSF"] = 0

                # Campos opcionales adicionales
                with st.expander("Especificaciones adicionales (opcional)"):
                    grasa_personalizada["descripcion"] = st.text_area(
                        "Descripción o aplicación:",
                        help="Describe el uso o características de la grasa para mejorar la búsqueda por texto."
                    )

                    aceite_base_options = sorted(df["Aceite Base"].dropna().unique())
                    aceite_base = st.selectbox(
                        "Aceite base:",
                        options=["No especificar"] + list(aceite_base_options)
                    )
                    if aceite_base != "No especificar":
                        grasa_personalizada["Aceite Base"] = aceite_base

                    espesante_options = sorted(df["Espesante"].dropna().unique())
                    espesante = st.selectbox(
                        "Espesante:",
                        options=["No especificar"] + list(espesante_options)
                    )
                    if espesante != "No especificar":
                        grasa_personalizada["Espesante"] = espesante

            modo_similitud = st.selectbox(
                "Modo de similitud:",
                ["hibrido", "texto", "numerico"],
                help="El modo híbrido combina descripción y datos técnicos (recomendado)."
            )

            if modo_similitud == "hibrido":
                st.markdown("**Ajustar pesos:**")
                peso_texto = st.slider(
                    "Peso texto (descripción):",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1
                )
                peso_numerico = 1.0 - peso_texto
                st.info(f"Peso numérico: {peso_numerico:.1f}")
            else:
                peso_texto = 0.7
                peso_numerico = 0.3

            top_n = st.slider(
                "Número de recomendaciones:",
                min_value=3,
                max_value=10,
                value=5
            )

            buscar_btn = st.button(
                "Ver recomendaciones",
                use_container_width=True,
                type="primary"
            )
            
        # ===== BÚSQUEDA POR FILTROS =====
        else:
            st.subheader("Especificaciones técnicas")
            
            grado_nlgi_options = sorted(df["Grado NLGI Consistencia"].dropna().unique())
            grado_nlgi = st.multiselect(
                "Grado NLGI:",
                options=grado_nlgi_options,
                help="Selecciona uno o más grados de consistencia."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                temp_min = st.number_input(
                    "Temperatura mínima del equipo (°C):",
                    value=None,
                    help="Temperatura mínima de operación."
                )
            with col2:
                temp_max = st.number_input(
                    "Temperatura máxima del equipo (°C):",
                    value=None,
                    help="Temperatura máxima de operación."
                )
            
            carga_timken = st.number_input(
                "Carga Timken mínima (lb):",
                min_value=0.0,
                value=None,
                help="Carga mínima requerida para el contacto."
            )
            
            requiere_nsf = st.selectbox(
                "Registro NSF:",
                options=[None, 1, 0],
                format_func=lambda x: "No importa" if x is None else ("Sí requiere" if x == 1 else "No requiere"),
                help="Selecciona 'Sí requiere' para aplicaciones en entorno alimenticio."
            )
            
            top_n = st.slider(
                "Número de resultados:",
                min_value=3,
                max_value=15,
                value=5
            )
            
            buscar_btn = st.button(
                "Buscar grasas",
                use_container_width=True,
                type="primary"
            )
        
        st.header("Estadísticas")
        st.metric("Grasas en catálogo", len(df))
        st.metric("Grasas Interlub", len(df[~df["esCompetencia"]]))
        st.metric("Grasas Competencia", len(df[df["esCompetencia"]]))
        st.metric("Características", len(df.columns))
    
    # ===== ÁREA PRINCIPAL =====
    
    if buscar_btn:
        try:
            if metodo_busqueda == "Búsqueda por Similitud":
                if grasa_personalizada is not None:
                    st.header("Opciones recomendadas para tu grasa personalizada")
                    st.caption(
                        "Estas grasas comparten características similares a las especificaciones que definiste. "
                        "Revisa siempre la ficha técnica antes de tomar una decisión."
                    )
                else:
                    st.header(f"Opciones recomendadas a partir de: {grasa_referencia}")
                    st.caption(
                        "Estas grasas comparten características clave con tu referencia. "
                        "Revisa siempre la ficha técnica antes de hacer un cambio en campo."
                    )

                recomendaciones = recomendar_grasas_hibrido(
                    codigo_grasa=grasa_referencia,
                    data_dict=data_dict,
                    top_n=top_n,
                    modo=modo_similitud,
                    peso_texto=peso_texto,
                    peso_numerico=peso_numerico,
                    grasa_personalizada=grasa_personalizada
                )

                with st.expander("Ver detalles de la grasa de referencia", expanded=True):
                    if grasa_personalizada is not None:
                        st.markdown(
                            "Especificaciones de la grasa personalizada que creaste:"
                        )
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Aceite base", grasa_personalizada.get("Aceite Base", "No especificado"))
                            st.metric("Espesante", grasa_personalizada.get("Espesante", "No especificado"))
                        with col2:
                            st.metric("Grado NLGI", grasa_personalizada.get("Grado NLGI Consistencia", "N/A"))
                            st.metric(
                                "Viscosidad 40°C",
                                f"{grasa_personalizada.get('Viscosidad del Aceite Base a 40°C. cSt', 0):.1f} cSt"
                            )
                        with col3:
                            st.metric("Temp. mín.", f"{grasa_personalizada.get('Temperatura de Servicio °C, min', 0)}°C")
                            st.metric("Temp. máx.", f"{grasa_personalizada.get('Temperatura de Servicio °C, max', 0)}°C")
                    else:
                        st.markdown(
                            "Resumen técnico de la grasa con la que estás comparando. "
                            "Úsalo para validar que el comparativo hace sentido con tu aplicación."
                        )
                        ref_data = df[df["codigoGrasa"] == grasa_referencia].iloc[0]
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Aceite base", ref_data.get("Aceite Base", "N/A"))
                            st.metric("Espesante", ref_data.get("Espesante", "N/A"))
                        with col2:
                            st.metric("Grado NLGI", ref_data.get("Grado NLGI Consistencia", "N/A"))
                            st.metric(
                                "Viscosidad 40°C",
                                f"{ref_data.get('Viscosidad del Aceite Base a 40°C. cSt', 0):.1f} cSt"
                            )
                        with col3:
                            st.metric("Temp. mín.", f"{ref_data.get('Temperatura de Servicio °C, min', 0)}°C")
                            st.metric("Temp. máx.", f"{ref_data.get('Temperatura de Servicio °C, max', 0)}°C")
                
                st.subheader("Nivel de similitud")
                fig = px.bar(
                    recomendaciones,
                    x='Similitud',
                    y='codigoGrasa',
                    orientation='h',
                    title=f'Top {top_n} grasas más similares (modo: {modo_similitud})',
                    labels={'codigoGrasa': 'Código', 'Similitud': 'Similitud'},
                    color='Similitud',
                    color_continuous_scale=[[0, "#FFE5E5"], [1, PRIMARY_RED]]
                )
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color=DARK_GREY,
                    title_font_color=DARK_GREY,
                )
                fig.update_xaxes(
                    color=DARK_GREY,
                    tickfont=dict(color=DARK_GREY),
                    title_font=dict(color=DARK_GREY)
                )
                fig.update_yaxes(
                    color=DARK_GREY,
                    tickfont=dict(color=DARK_GREY),
                    title_font=dict(color=DARK_GREY)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Detalles de las recomendaciones")

                # Agregar información de empresa origen
                recomendaciones = recomendaciones.merge(
                    df[["codigoGrasa", "esCompetencia", "empresaOrigen"]],
                    on="codigoGrasa",
                    how="left"
                )

                columnas_mostrar = [
                    'codigoGrasa', 'empresaOrigen', 'Similitud', 'subtitulo',
                    'Grado NLGI Consistencia',
                    'Viscosidad del Aceite Base a 40°C. cSt',
                    'Punto de Gota, °C',
                    'Temperatura de Servicio °C, min',
                    'Temperatura de Servicio °C, max'
                ]

                columnas_mostrar = [c for c in columnas_mostrar if c in recomendaciones.columns]
                display_df = recomendaciones[columnas_mostrar].copy()

                if 'Similitud' in display_df.columns:
                    display_df['Similitud'] = display_df['Similitud'].apply(lambda x: f"{x:.3f}")

                display_df = display_df.rename(columns={
                    "subtitulo": "Descripción corta",
                    "Grado NLGI Consistencia": "Grado NLGI",
                    "Punto de Gota, °C": "Punto de gota (°C)",
                    "empresaOrigen": "Origen",
                })

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
            else:
                st.header("Grasas que cumplen con los requisitos de tu aplicación")
                
                grado_nlgi_param = grado_nlgi if grado_nlgi else None
                
                resultados = recomendar_por_filtros(
                    df=df,
                    grado_nlgi=grado_nlgi_param,
                    temp_min_trabajo=temp_min,
                    temp_max_trabajo=temp_max,
                    carga_timken_min=carga_timken,
                    requiere_nsf=requiere_nsf,
                    top_n=top_n
                )
                
                if resultados.empty:
                    st.warning("No se encontraron grasas que cumplan con todos los criterios especificados.")
                    st.info(
                        "Prueba relajando uno de los filtros o contacta a tu asesor Interlub "
                        "para revisar condiciones especiales de operación."
                    )
                else:
                    st.success(f"Se encontraron {len(resultados)} grasa(s) que cumplen con los criterios definidos.")
                    
                    with st.expander("Criterios de búsqueda aplicados", expanded=True):
                        criterios = []
                        if grado_nlgi:
                            criterios.append(f"**Grado NLGI:** {', '.join(map(str, grado_nlgi))}")
                        if temp_min is not None:
                            criterios.append(f"**Temperatura mínima del equipo:** ≤ {temp_min}°C")
                        if temp_max is not None:
                            criterios.append(f"**Temperatura máxima del equipo:** ≥ {temp_max}°C")
                        if carga_timken is not None:
                            criterios.append(f"**Carga Timken mínima:** ≥ {carga_timken} lb")
                        if requiere_nsf is not None:
                            criterios.append(
                                f"**NSF:** {'Sí requiere' if requiere_nsf == 1 else 'No requiere'}"
                            )
                        
                        for criterio in criterios:
                            st.markdown(f"- {criterio}")
                    
                    st.subheader("Grasas encontradas")

                    columnas_mostrar = [
                        "codigoGrasa",
                        "empresaOrigen",
                        "subtitulo",
                        "Aceite Base",
                        "Espesante",
                        "Grado NLGI Consistencia",
                        "Temperatura de Servicio °C, min",
                        "Temperatura de Servicio °C, max",
                        "Punto de Gota, °C",
                        "Carga Timken Ok, lb",
                        "Resistencia al Lavado por Agua a 80°C, %",
                        "Registro NSF",
                    ]

                    columnas_mostrar = [c for c in columnas_mostrar if c in resultados.columns]
                    display_df = resultados[columnas_mostrar].copy()

                    display_df = display_df.rename(columns={
                        "subtitulo": "Descripción corta",
                        "Grado NLGI Consistencia": "Grado NLGI",
                        "Punto de Gota, °C": "Punto de gota (°C)",
                        "empresaOrigen": "Origen",
                    })

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # ---- RANGOS DE TEMPERATURA CON GRADIENTE FRÍO→CALIENTE ----
                    st.subheader("Rangos de temperatura")
                    fig = go.Figure()

                    # Para mapear color por temperatura media
                    global_min = resultados["Temperatura de Servicio °C, min"].min()
                    global_max = resultados["Temperatura de Servicio °C, max"].max()
                    temp_range = max(global_max - global_min, 1)

                    for idx, row in resultados.iterrows():
                        mid_temp = (row["Temperatura de Servicio °C, min"] +
                                    row["Temperatura de Servicio °C, max"]) / 2.0
                        t_norm = (mid_temp - global_min) / temp_range  # 0–1
                        # Escala: azul (frío) → amarillo → rojo (caliente)
                        color_val = colors.sample_colorscale(
                            [[0.0, "#005BFF"], [0.5, "#FFD54A"], [1.0, PRIMARY_RED]],
                            [t_norm]
                        )[0]

                        fig.add_trace(go.Scatter(
                            x=[row["Temperatura de Servicio °C, min"], row["Temperatura de Servicio °C, max"]],
                            y=[row["codigoGrasa"], row["codigoGrasa"]],
                            mode='lines+markers',
                            name=row["codigoGrasa"],
                            line=dict(width=10, color=color_val),
                            marker=dict(size=10, color=color_val)
                        ))
                    
                    if temp_min is not None or temp_max is not None:
                        shapes = []
                        if temp_min is not None:
                            shapes.append(dict(
                                type="line",
                                x0=temp_min,
                                x1=temp_min,
                                y0=0,
                                y1=1,
                                yref="paper",
                                line=dict(color="#005BFF", width=2, dash="dash")
                            ))
                        if temp_max is not None:
                            shapes.append(dict(
                                type="line",
                                x0=temp_max,
                                x1=temp_max,
                                y0=0,
                                y1=1,
                                yref="paper",
                                line=dict(color=PRIMARY_RED, width=2, dash="dash")
                            ))
                        fig.update_layout(shapes=shapes)
                    
                    fig.update_layout(
                        title="Rangos de temperatura de servicio",
                        xaxis_title="Temperatura (°C)",
                        yaxis_title="Código de grasa",
                        height=400,
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color=DARK_GREY,
                        title_font_color=DARK_GREY,
                    )
                    fig.update_xaxes(
                        color=DARK_GREY,
                        tickfont=dict(color=DARK_GREY),
                        title_font=dict(color=DARK_GREY)
                    )
                    fig.update_yaxes(
                        color=DARK_GREY,
                        tickfont=dict(color=DARK_GREY),
                        title_font=dict(color=DARK_GREY)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "Cada línea representa el rango de temperatura recomendado para cada grasa. "
                        "Los colores más cercanos al rojo indican productos para temperaturas más elevadas."
                    )
        
        except Exception as e:
            st.error(f"Error al procesar la búsqueda: {str(e)}")
            st.exception(e)
    
    else:
        st.header("¿Cómo quieres buscar tu grasa ideal?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Búsqueda con grasa de referencia
            
            Usa una grasa que ya conoces para encontrar alternativas Interlub con comportamiento similar.
            
            - Ideal para homologar productos  
            - Útil para sustituir grasas de otro proveedor
            """)
        
        with col2:
            st.markdown("""
            ### Búsqueda por requisitos técnicos
            
            Parte de las condiciones de operación del equipo y filtra por:
            
            - Rango de temperatura  
            - Grado NLGI  
            - Carga mínima y registro NSF
            """)
        
        st.subheader("Distribución del catálogo")
        
        col1, col2, col3 = st.columns(3)
        
        if "Aceite Base" in df.columns:
            with col1:
                fig1 = px.histogram(
                    df.dropna(subset=["Aceite Base"]),
                    x='Aceite Base',
                    title='Distribución por aceite base',
                    color_discrete_sequence=[PRIMARY_RED]
                )
                fig1.update_layout(
                    showlegend=False,
                    xaxis_tickangle=-45,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color=DARK_GREY,
                    title_font_color=DARK_GREY,
                )
                fig1.update_xaxes(
                    color=DARK_GREY,
                    tickfont=dict(color=DARK_GREY),
                    title_font=dict(color=DARK_GREY)
                )
                fig1.update_yaxes(
                    color=DARK_GREY,
                    tickfont=dict(color=DARK_GREY),
                    title_font=dict(color=DARK_GREY)
                )
                st.plotly_chart(fig1, use_container_width=True)
        
        if "Espesante" in df.columns:
            with col2:
                fig2 = px.histogram(
                    df.dropna(subset=["Espesante"]),
                    x='Espesante',
                    title='Distribución por espesante',
                    color_discrete_sequence=[PRIMARY_RED]
                )
                fig2.update_layout(
                    showlegend=False,
                    xaxis_tickangle=-45,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color=DARK_GREY,
                    title_font_color=DARK_GREY,
                )
                fig2.update_xaxes(
                    color=DARK_GREY,
                    tickfont=dict(color=DARK_GREY),
                    title_font=dict(color=DARK_GREY)
                )
                fig2.update_yaxes(
                    color=DARK_GREY,
                    tickfont=dict(color=DARK_GREY),
                    title_font=dict(color=DARK_GREY)
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        if "Grado NLGI Consistencia" in df.columns:
            with col3:
                fig3 = px.histogram(
                    df,
                    x='Grado NLGI Consistencia',
                    title='Distribución por grado NLGI',
                    color_discrete_sequence=[PRIMARY_RED]
                )
                fig3.update_layout(
                    showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color=DARK_GREY,
                    title_font_color=DARK_GREY,
                )
                fig3.update_xaxes(
                    color=DARK_GREY,
                    tickfont=dict(color=DARK_GREY),
                    title_font=dict(color=DARK_GREY)
                )
                fig3.update_yaxes(
                    color=DARK_GREY,
                    tickfont=dict(color=DARK_GREY),
                    title_font=dict(color=DARK_GREY)
                )
                st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()
