import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords

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

# Aplicar CSS personalizado
st.markdown("""
    <style>
    /* Forzar fondo blanco en toda la aplicación */
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    .main {
        background-color: #FFFFFF !important;
    }
    
    .block-container {
        background-color: #FFFFFF !important;
    }
    
    /* Colores principales */
    :root {
        --primary-color: #CC0000;
        --background-color: #FFFFFF;
        --secondary-bg: #F0F0F0;
        --text-color: #2B2B2B;
        --text-light: #666666;
    }
    
    /* Botones principales */
    .stButton>button {
        background-color: #CC0000 !important;
        color: #FFFFFF !important;
        border: 2px solid #CC0000 !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(204, 0, 0, 0.2) !important;
    }
    
    .stButton>button:hover {
        background-color: #990000 !important;
        border-color: #990000 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 8px rgba(204, 0, 0, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton>button:active {
        background-color: #660000 !important;
        transform: translateY(0) !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #2B2B2B !important;
        font-weight: 600 !important;
    }
    
    h1 {
        border-bottom: 4px solid #CC0000 !important;
        padding-bottom: 15px !important;
        margin-bottom: 20px !important;
    }
    
    h2 {
        color: #2B2B2B !important;
        margin-top: 1.5rem !important;
        font-size: 1.5rem !important;
    }
    
    h3 {
        color: #2B2B2B !important;
        font-size: 1.25rem !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8F8F8 !important;
        border-right: 1px solid #E0E0E0 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #2B2B2B !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #2B2B2B !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    [data-testid="stSidebar"] p {
        color: #2B2B2B !important;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        color: #CC0000 !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #2B2B2B !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #FFFFFF !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #F0F0F0 !important;
        color: #2B2B2B !important;
        border-radius: 8px 8px 0 0 !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border: 2px solid transparent !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #CC0000 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: 2px solid #CC0000 !important;
    }
    
    /* Selectbox y inputs */
    .stSelectbox label, 
    .stNumberInput label, 
    .stMultiSelect label,
    .stTextInput label,
    .stSlider label {
        color: #2B2B2B !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 2px solid #E0E0E0 !important;
        border-radius: 6px !important;
    }
    
    .stSelectbox div[data-baseweb="select"]:hover {
        border-color: #CC0000 !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        color: #2B2B2B !important;
        font-weight: 500 !important;
    }
    
    /* Number input */
    .stNumberInput input {
        color: #2B2B2B !important;
        background-color: #FFFFFF !important;
        border: 2px solid #E0E0E0 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    
    .stNumberInput input:focus {
        border-color: #CC0000 !important;
        box-shadow: 0 0 0 1px #CC0000 !important;
    }
    
    /* Dataframes */
    .dataframe {
        border: 2px solid #E0E0E0 !important;
        color: #2B2B2B !important;
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    .dataframe th {
        background-color: #F8F8F8 !important;
        color: #2B2B2B !important;
        font-weight: 700 !important;
        padding: 12px !important;
        border-bottom: 2px solid #E0E0E0 !important;
    }
    
    .dataframe td {
        color: #2B2B2B !important;
        padding: 10px !important;
        background-color: #FFFFFF !important;
    }
    
    .dataframe tr:hover {
        background-color: #FFF5F5 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #F8F8F8 !important;
        color: #2B2B2B !important;
        border-left: 4px solid #CC0000 !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 12px !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #F0F0F0 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #FFFFFF !important;
        color: #2B2B2B !important;
        border: 1px solid #E0E0E0 !important;
        border-top: none !important;
        padding: 15px !important;
    }
    
    /* Info boxes */
    .stAlert {
        color: #2B2B2B !important;
        border-radius: 8px !important;
        padding: 15px !important;
        font-weight: 500 !important;
    }
    
    div[data-baseweb="notification"] {
        background-color: #F8F8F8 !important;
        color: #2B2B2B !important;
        border-left: 4px solid #CC0000 !important;
        border-radius: 6px !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #2B2B2B !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    .stRadio label[data-baseweb="radio"] span {
        color: #2B2B2B !important;
        font-weight: 500 !important;
    }
    
    .stRadio [data-baseweb="radio"] > div:first-child {
        border-color: #CC0000 !important;
    }
    
    /* Slider */
    .stSlider label {
        color: #2B2B2B !important;
        font-weight: 600 !important;
    }
    
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #CC0000 !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        background: linear-gradient(to right, #CC0000 0%, #CC0000 var(--value), #E0E0E0 var(--value), #E0E0E0 100%) !important;
    }
    
    /* Links */
    a {
        color: #CC0000 !important;
        font-weight: 600 !important;
        text-decoration: none !important;
    }
    
    a:hover {
        color: #990000 !important;
        text-decoration: underline !important;
    }
    
    /* Mensajes */
    .stWarning {
        background-color: #FFF8E1 !important;
        color: #2B2B2B !important;
        border-left: 4px solid #FFA000 !important;
        border-radius: 6px !important;
        padding: 15px !important;
    }
    
    .stSuccess {
        background-color: #F1F8E9 !important;
        color: #2B2B2B !important;
        border-left: 4px solid #66BB6A !important;
        border-radius: 6px !important;
        padding: 15px !important;
    }
    
    .stError {
        background-color: #FFEBEE !important;
        color: #2B2B2B !important;
        border-left: 4px solid #EF5350 !important;
        border-radius: 6px !important;
        padding: 15px !important;
    }
    
    .stInfo {
        background-color: #E3F2FD !important;
        color: #2B2B2B !important;
        border-left: 4px solid #42A5F5 !important;
        border-radius: 6px !important;
        padding: 15px !important;
    }
    
    /* Texto general */
    p, span, div, li {
        color: #2B2B2B !important;
    }
    
    /* Markdown */
    .stMarkdown {
        color: #2B2B2B !important;
    }
    
    /* JSON display */
    .stJson {
        background-color: #F8F8F8 !important;
        color: #2B2B2B !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 6px !important;
        padding: 15px !important;
    }
    
    /* Code blocks */
    code {
        background-color: #F8F8F8 !important;
        color: #CC0000 !important;
        padding: 3px 6px !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
    }
    
    /* Multi-select */
    .stMultiSelect label {
        color: #2B2B2B !important;
        font-weight: 600 !important;
    }
    
    .stMultiSelect div[data-baseweb="tag"] {
        background-color: #CC0000 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 4px !important;
    }
    
    /* Divider */
    hr {
        border-color: #E0E0E0 !important;
        border-width: 2px !important;
        margin: 2rem 0 !important;
    }
    
    /* Tooltips */
    [data-baseweb="tooltip"] {
        background-color: #2B2B2B !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
    }
    
    /* Placeholder text */
    ::placeholder {
        color: #999999 !important;
    }
    
    /* Disabled elements */
    :disabled {
        color: #CCCCCC !important;
        background-color: #F5F5F5 !important;
    }
    
    /* Líneas de separación */
    [data-testid="stHorizontalBlock"] {
        background-color: #FFFFFF !important;
    }
    
    /* Columnas */
    [data-testid="column"] {
        background-color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== FUNCIONES DE CARGA Y PREPROCESAMIENTO ====================

@st.cache_data
def load_and_preprocess_data(csv_path):
    """
    Cargar y preprocesar datos de grasas.
    Retorna todos los componentes necesarios para el sistema de recomendación.
    """
    try:
        # Cargar datos
        df_raw = pd.read_csv(csv_path, encoding="utf-8")
        
        # Copiar y preparar datos
        df = df_raw.copy()
        df["codigoGrasa"] = [f"Grasa_{i+1}" for i in range(len(df))]
        
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
):
    """
    Recomienda grasas similares usando el método híbrido.
    
    Args:
        codigo_grasa: Código de la grasa de referencia
        data_dict: Diccionario con los datos preprocesados
        top_n: Número de recomendaciones
        modo: "texto", "numerico" o "hibrido"
        peso_texto: Peso para similitud de texto (default: 0.7)
        peso_numerico: Peso para similitud numérica (default: 0.3)
    
    Returns:
        DataFrame con las recomendaciones
    """
    indices = data_dict['indices']
    df_features = data_dict['df_features']
    descripcion_grasas = data_dict['descripcion_grasas']
    cosine_sim_text = data_dict['cosine_sim_text']
    cosine_sim_numeric = data_dict['cosine_sim_numeric']
    
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
    
    Args:
        df: DataFrame con los datos de grasas
        grado_nlgi: Grado NLGI deseado (puede ser lista)
        temp_min_trabajo: Temperatura mínima de operación requerida
        temp_max_trabajo: Temperatura máxima de operación requerida
        carga_timken_min: Carga Timken mínima requerida
        requiere_nsf: 1 para requerir NSF, 0 para no NSF, None para no filtrar
        top_n: Número de resultados
    
    Returns:
        DataFrame con las grasas filtradas
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
    # Título principal
    st.title("Recomendador Inteligente de Grasas Interlub")
    st.markdown("---")
    
    # Cargar datos
    # NOTA: Cambiar esta ruta por la ubicación de tu archivo CSV
    csv_path = "datos_grasas_Tec_limpio.csv"  # Modificar según tu estructura
    
    data_dict = load_and_preprocess_data(csv_path)
    
    if data_dict is None:
        st.error(" No se pudieron cargar los datos. Verifica la ruta del archivo CSV.")
        st.info("Coloca tu archivo 'datos_grasas_Tec_limpio.csv' en el mismo directorio que este script.")
        return
    
    df = data_dict['df']
    df_features = data_dict['df_features']
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.header("Método de Búsqueda")
        
        metodo_busqueda = st.radio(
            "Selecciona el método:",
            ["Búsqueda por Similitud", "Búsqueda por Filtros"],
            help="Similitud: encuentra grasas parecidas a una existente. Filtros: busca por especificaciones técnicas."
        )
        
        st.markdown("---")
        
        # ===== BÚSQUEDA POR SIMILITUD =====
        if metodo_busqueda == "Búsqueda por Similitud":
            st.subheader("Configuración de Similitud")
            
            # Seleccionar grasa de referencia
            grasa_referencia = st.selectbox(
                "Grasa de referencia:",
                options=df_features.index.tolist(),
                help="Selecciona una grasa del catálogo para encontrar similares"
            )
            
            # Modo de similitud
            modo_similitud = st.selectbox(
                "Modo de similitud:",
                ["hibrido", "texto", "numerico"],
                help="Híbrido combina texto y propiedades numéricas"
            )
            
            # Pesos (solo si es híbrido)
            if modo_similitud == "hibrido":
                st.markdown("**Ajustar pesos:**")
                peso_texto = st.slider(
                    "Peso Texto (descripción):",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1
                )
                peso_numerico = 1.0 - peso_texto
                st.info(f"Peso Numérico: {peso_numerico:.1f}")
            else:
                peso_texto = 0.7
                peso_numerico = 0.3
            
            # Número de recomendaciones
            top_n = st.slider(
                "Número de recomendaciones:",
                min_value=3,
                max_value=10,
                value=5
            )
            
            buscar_btn = st.button(
                " Buscar Similares",
                use_container_width=True,
                type="primary"
            )
            
        # ===== BÚSQUEDA POR FILTROS =====
        else:
            st.subheader(" Especificaciones Técnicas")
            
            # NLGI
            grado_nlgi_options = sorted(df["Grado NLGI Consistencia"].dropna().unique())
            grado_nlgi = st.multiselect(
                "Grado NLGI:",
                options=grado_nlgi_options,
                help="Selecciona uno o más grados NLGI"
            )
            
            # Temperaturas
            col1, col2 = st.columns(2)
            with col1:
                temp_min = st.number_input(
                    "Temp. Mín. (°C):",
                    value=None,
                    help="La grasa debe soportar esta temperatura mínima"
                )
            with col2:
                temp_max = st.number_input(
                    "Temp. Máx. (°C):",
                    value=None,
                    help="La grasa debe soportar esta temperatura máxima"
                )
            
            # Carga Timken
            carga_timken = st.number_input(
                "Carga Timken mín. (lb):",
                min_value=0.0,
                value=None,
                help="Carga mínima requerida"
            )
            
            # NSF
            requiere_nsf = st.selectbox(
                "Registro NSF:",
                options=[None, 1, 0],
                format_func=lambda x: "No importa" if x is None else ("Sí requiere" if x == 1 else "No requiere"),
                help="Para industria alimenticia"
            )
            
            top_n = st.slider(
                "Número de resultados:",
                min_value=3,
                max_value=15,
                value=5
            )
            
            buscar_btn = st.button(
                "🔎 Buscar por Filtros",
                use_container_width=True,
                type="primary"
            )
        
        # Estadísticas
        st.markdown("---")
        st.header(" Estadísticas")
        st.metric("Grasas en catálogo", len(df))
        st.metric("Características", len(df.columns))
    
    # ===== ÁREA PRINCIPAL =====
    
    if buscar_btn:
        try:
            if metodo_busqueda == "Búsqueda por Similitud":
                # Ejecutar búsqueda por similitud
                st.header(f" Grasas Similares a: {grasa_referencia}")
                
                recomendaciones = recomendar_grasas_hibrido(
                    codigo_grasa=grasa_referencia,
                    data_dict=data_dict,
                    top_n=top_n,
                    modo=modo_similitud,
                    peso_texto=peso_texto,
                    peso_numerico=peso_numerico
                )
                
                # Mostrar grasa de referencia
                with st.expander(" Ver detalles de la grasa de referencia", expanded=True):
                    ref_data = df[df["codigoGrasa"] == grasa_referencia].iloc[0]
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Aceite Base", ref_data.get("Aceite Base", "N/A"))
                        st.metric("Espesante", ref_data.get("Espesante", "N/A"))
                    with col2:
                        st.metric("NLGI", ref_data.get("Grado NLGI Consistencia", "N/A"))
                        st.metric("Viscosidad 40°C", f"{ref_data.get('Viscosidad del Aceite Base a 40°C. cSt', 0):.1f} cSt")
                    with col3:
                        st.metric("Temp. Mín.", f"{ref_data.get('Temperatura de Servicio °C, min', 0)}°C")
                        st.metric("Temp. Máx.", f"{ref_data.get('Temperatura de Servicio °C, max', 0)}°C")
                
                # Gráfico de similitudes
                st.subheader("Nivel de Similitud")
                fig = px.bar(
                    recomendaciones,
                    x='Similitud',
                    y='codigoGrasa',
                    orientation='h',
                    title=f'Top {top_n} Grasas Más Similares (Modo: {modo_similitud})',
                    labels={'codigoGrasa': 'Código', 'Similitud': 'Similitud'},
                    color='Similitud',
                    color_continuous_scale=['#FFE5E5', '#CC0000', '#660000']
                )
                fig.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#FFFFFF',
                    font=dict(color='#2B2B2B', size=12),
                    title_font=dict(color='#2B2B2B', size=16, family='sans-serif')
                )
                fig.update_xaxes(gridcolor='#E0E0E0', title_font=dict(color='#2B2B2B'))
                fig.update_yaxes(gridcolor='#E0E0E0', title_font=dict(color='#2B2B2B'))
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla de resultados
                st.subheader(" Detalles de las Recomendaciones")
                
                columnas_mostrar = [
                    'codigoGrasa', 'Similitud', 'subtitulo',
                    'Grado NLGI Consistencia',
                    'Viscosidad del Aceite Base a 40°C. cSt',
                    'Punto de Gota, °C',
                    'Temperatura de Servicio °C, min',
                    'Temperatura de Servicio °C, max'
                ]
                
                # Filtrar columnas existentes
                columnas_mostrar = [c for c in columnas_mostrar if c in recomendaciones.columns]
                display_df = recomendaciones[columnas_mostrar].copy()
                
                # Formatear similitud
                if 'Similitud' in display_df.columns:
                    display_df['Similitud'] = display_df['Similitud'].apply(lambda x: f"{x:.3f}")
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
            else:  # Búsqueda por filtros
                st.header(" Resultados de Búsqueda por Filtros")
                
                # Preparar parámetros
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
                    st.warning(" No se encontraron grasas que cumplan con todos los criterios especificados.")
                    st.info(" Intenta ajustar los filtros para obtener resultados.")
                else:
                    st.success(f"OK Se encontraron {len(resultados)} grasa(s) que cumplen los criterios")
                    
                    # Mostrar criterios de búsqueda
                    with st.expander(" Criterios de búsqueda aplicados", expanded=True):
                        criterios = []
                        if grado_nlgi:
                            criterios.append(f"**NLGI:** {', '.join(map(str, grado_nlgi))}")
                        if temp_min is not None:
                            criterios.append(f"**Temp. Mín.:** ≤ {temp_min}°C")
                        if temp_max is not None:
                            criterios.append(f"**Temp. Máx.:** ≥ {temp_max}°C")
                        if carga_timken is not None:
                            criterios.append(f"**Carga Timken:** ≥ {carga_timken} lb")
                        if requiere_nsf is not None:
                            criterios.append(f"**NSF:** {'Sí requiere' if requiere_nsf == 1 else 'No requiere'}")
                        
                        for criterio in criterios:
                            st.markdown(f"- {criterio}")
                    
                    # Tabla de resultados
                    st.subheader(" Grasas Encontradas")
                    
                    columnas_mostrar = [
                        "codigoGrasa",
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
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Visualización de rangos de temperatura
                    st.subheader(" Rangos de Temperatura")
                    fig = go.Figure()
                    
                    colors = ['#CC0000', '#990000', '#660000', '#AA0000', '#880000']
                    
                    for idx, row in enumerate(resultados.iterrows()):
                        i, row = row
                        color_idx = idx % len(colors)
                        fig.add_trace(go.Scatter(
                            x=[row["Temperatura de Servicio °C, min"], row["Temperatura de Servicio °C, max"]],
                            y=[row["codigoGrasa"], row["codigoGrasa"]],
                            mode='lines+markers',
                            name=row["codigoGrasa"],
                            line=dict(width=8, color=colors[color_idx]),
                            marker=dict(size=10, color=colors[color_idx])
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
                                line=dict(color="#2B2B2B", width=2, dash="dash")
                            ))
                        if temp_max is not None:
                            shapes.append(dict(
                                type="line",
                                x0=temp_max,
                                x1=temp_max,
                                y0=0,
                                y1=1,
                                yref="paper",
                                line=dict(color="#2B2B2B", width=2, dash="dash")
                            ))
                        fig.update_layout(shapes=shapes)
                    
                    fig.update_layout(
                        title="Rangos de Temperatura de Servicio",
                        xaxis_title="Temperatura (°C)",
                        yaxis_title="Código de Grasa",
                        height=400,
                        showlegend=False,
                        plot_bgcolor='#FFFFFF',
                        paper_bgcolor='#FFFFFF',
                        font=dict(color='#2B2B2B', size=12),
                        title_font=dict(color='#2B2B2B', size=16, family='sans-serif')
                    )
                    fig.update_xaxes(gridcolor='#E0E0E0', title_font=dict(color='#2B2B2B'))
                    fig.update_yaxes(gridcolor='#E0E0E0', title_font=dict(color='#2B2B2B'))
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"ERROR Error al procesar la búsqueda: {str(e)}")
            st.exception(e)
    
    else:
        # Pantalla de bienvenida
        st.header(" ¡Bienvenido!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ###  Búsqueda por Similitud
            
            Encuentra grasas **similares** a una grasa existente en el catálogo.
            
            **Ventajas:**
            - Usa inteligencia artificial para encontrar productos parecidos
            - Combina descripción textual y propiedades técnicas
            - Ideal cuando conoces un producto de referencia
            
            **Modos disponibles:**
            - **Híbrido:** Equilibra texto y números (recomendado)
            - **Texto:** Solo descripción y aplicaciones
            - **Numérico:** Solo propiedades técnicas
            """)
        
        with col2:
            st.markdown("""
            ###  Búsqueda por Filtros
            
            Busca grasas que cumplan **especificaciones técnicas** exactas.
            
            **Ventajas:**
            - Filtrado preciso por requisitos
            - Ideal para nuevos proyectos
            - Múltiples criterios combinables
            
            **Filtros disponibles:**
            - Grado NLGI de consistencia
            - Rangos de temperatura de servicio
            - Carga Timken mínima
            - Certificación NSF (industria alimenticia)
            """)
        
        st.markdown("---")
        st.subheader(" Distribución del Catálogo")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "Aceite Base" in df.columns:
                fig1 = px.histogram(
                    df.dropna(subset=["Aceite Base"]),
                    x='Aceite Base',
                    title='Distribución por Aceite Base',
                    color_discrete_sequence=['#CC0000']
                )
                fig1.update_layout(
                    showlegend=False, 
                    xaxis_tickangle=-45,
                    plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#FFFFFF',
                    font=dict(color='#2B2B2B', size=11),
                    title_font=dict(color='#2B2B2B', size=14, family='sans-serif')
                )
                fig1.update_xaxes(gridcolor='#E0E0E0', title_font=dict(color='#2B2B2B'))
                fig1.update_yaxes(gridcolor='#E0E0E0', title_font=dict(color='#2B2B2B'))
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            if "Espesante" in df.columns:
                fig2 = px.histogram(
                    df.dropna(subset=["Espesante"]),
                    x='Espesante',
                    title='Distribución por Espesante',
                    color_discrete_sequence=['#990000']
                )
                fig2.update_layout(
                    showlegend=False, 
                    xaxis_tickangle=-45,
                    plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#FFFFFF',
                    font=dict(color='#2B2B2B', size=11),
                    title_font=dict(color='#2B2B2B', size=14, family='sans-serif')
                )
                fig2.update_xaxes(gridcolor='#E0E0E0', title_font=dict(color='#2B2B2B'))
                fig2.update_yaxes(gridcolor='#E0E0E0', title_font=dict(color='#2B2B2B'))
                st.plotly_chart(fig2, use_container_width=True)
        
        with col3:
            if "Grado NLGI Consistencia" in df.columns:
                fig3 = px.histogram(
                    df,
                    x='Grado NLGI Consistencia',
                    title='Distribución por Grado NLGI',
                    color_discrete_sequence=['#660000']
                )
                fig3.update_layout(
                    showlegend=False,
                    plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#FFFFFF',
                    font=dict(color='#2B2B2B', size=11),
                    title_font=dict(color='#2B2B2B', size=14, family='sans-serif')
                )
                fig3.update_xaxes(gridcolor='#E0E0E0', title_font=dict(color='#2B2B2B'))
                fig3.update_yaxes(gridcolor='#E0E0E0', title_font=dict(color='#2B2B2B'))
                st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()