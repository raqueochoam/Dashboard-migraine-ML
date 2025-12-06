# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Migrañas - Proyecto ML",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2D3748;
        border-bottom: 3px solid #4C51BF;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .info-box {
        background-color: #F7FAFC;
        border-left: 4px solid #4C51BF;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">Análisis Exploratorio sobre síntomas de la Migraña</h1>', unsafe_allow_html=True)

# Barra lateral
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=100)
    st.title("Navegación")
    st.markdown("---")
    
    section = st.radio(
        "Selecciona una sección:",
        ["Introduccion","Descripcion de Datos", "Análisis Estadístico", "Visualizaciones", "Bibliografía"]
    )
    
    st.markdown("---")
    "Herramientas Aplicadas para la Ciencia De Datos"
    "Universidad Panamericana"

    st.markdown("---")
    st.info("Raquel Magdalena Ochoa Martinez - ID: 0235324")

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("migrain_df.csv")
    df_clean = pd.read_csv("migrain_df_clean.csv")
    return df, df_clean

df, df_clean = load_data()

# ========== SECCIÓN: INTRODUCCIÓN ==========
if section == "Introduccion":
    
    image_path = "cerebros.png"
    st.image(image_path, caption='Visualización de todas las redes cerebrales activadas', use_container_width=True)
    
    st.markdown('<h2 class="section-header">Introducción</h2>', unsafe_allow_html=True)

    intro_text = """
    Las **cefaleas de tensión**, comúnmente conocidas como **migrañas**, constituyen uno de los trastornos neurológicos más frecuentes a nivel global.  
    Se describen como un dolor de cabeza que puede causar un **dolor pulsátil intenso** o una **sensación pulsante**, generalmente de un solo lado.  
    Suelen acompañarse de **náuseas**, **vómitos** y **sensibilidad extrema a la luz y al sonido**.  
    _(Mayo Clinic, 2023)_
    """
    st.markdown(intro_text)

    #GRAFICA DE PIE PARA EL 40% DE LA POBLACION CON MIGRAÑA
    st.markdown("### Prevalencia global de las migrañas")

    fig = go.Figure()

    fig.add_trace(go.Pie(
        labels=["Población con migraña (40%)", "Población sin migraña (60%)"],
        values=[40, 60],
        hole=0.55,
        textinfo="label+percent",
        pull=[0.1, 0],
    ))

    fig.update_layout(
        showlegend=False,
        title_text="Más de 3,100 millones de personas sufren migraña en el mundo",
        title_x=0.5,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-box">
        <h3 style="color:#1E3A8A;">Situación de la migraña en México</h3>
        <p style="font-size:1.1rem; color:#2D3748;">
            En México, esta patología representa un importante problema de salud pública:  
            se calcula que alrededor del <b>15 % de la población presenta migraña</b>, afectando la calidad de vida de aproximadamente  
            <b>20 millones de personas</b>.  
            <br>(Pisa, 2025; Secretaría de Salud, 2024)
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Objetivo General")
    st.markdown("""
    Desarrollar y evaluar un **sistema de clasificación automática** capaz de distinguir entre distintos tipos de migraña utilizando técnicas de **Machine Learning** basadas exclusivamente en los síntomas reportados por los pacientes.
    """)

    st.markdown("### Objetivos Específicos")
    st.markdown("""
    - Analizar y describir el conjunto de datos utilizado, identificando las variables clínicas y sintomatológicas más relevantes para la clasificación.  
    - Preprocesar los datos mediante limpieza, codificación y normalización para garantizar la calidad del dataset.  
    - Entrenar y comparar distintos modelos de Machine Learning para determinar su desempeño en la clasificación de migraña.  
    - Evaluar e interpretar los resultados, identificando las características más importantes y discutiendo su utilidad en el apoyo al diagnóstico clínico.
    """)

# ========== SECCIÓN: descripcion de datos==========
elif section == "Descripcion de Datos":
    st.markdown('<h2 class="section-header">Exploración del Dataset</h2>', unsafe_allow_html=True)

    # Primeras 10 filas de dataset
    st.markdown("### Primeras 10 filas del dataset")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    # Tabla de variable
    st.markdown("### Diccionario de Variables")

    variables = {
        "AGE": "Edad del participante (años).",
        "DURATION": "Duración del episodio (horas).",
        "FREQUENCY": "Frecuencia de episodios en el periodo registrado.",
        "INTENSITY": "Intensidad del dolor (1–5).",
        "LOCATION": "Ubicación del dolor (1=Unilateral, 2=Bilateral, 3=Frontal, 4=Temporal).",
        "CHARACTER": "Característica del dolor (1=Punzante, 2=Opresivo, 3=Agudo, 4=Sordo).",
        "NAUSEA": "Presencia de náusea.",
        "VOMIT": "Vómito durante el episodio.",
        "PHONOPHOBIA": "Sensibilidad al sonido.",
        "PHOTOPHOBIA": "Sensibilidad a la luz.",
        "VISUAL": "Alteraciones visuales (contiene inconsistencias).",
        "SENSORY": "Alteraciones sensoriales.",
        "VERTIGO": "Presencia de mareo o vértigo.",
        "TINNITUS": "Zumbido o sonidos agudos.",
        "HYPOACUSIS": "Disminución de audición.",
        "DIPLOPIA": "Doble visión.",
        "DEFECT": "Defecto en campo visual.",
        "ATAXIA": "Problemas de coordinación (columna constante).",
        "CONSCIENCE": "Alteración de conciencia.",
        "PARESTHESIA": "Hormigueo o alteraciones sensitivas.",
        "DPF": "Factor físico disfuncional.",
        "TYPE": "Tipo de migraña (variable objetivo)."
    }

    dict_df = pd.DataFrame({
        "Variable": list(variables.keys()),
        "Descripción": list(variables.values())
    })

    st.dataframe(dict_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # resumen
    st.markdown("### Resumen General del Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total de registros", df.shape[0], help="Número total de observaciones en la base.")

    with col2:
        st.metric("Total de columnas", df.shape[1], help="Cantidad de variables disponibles.")

    with col3:
        st.metric("Valores nulos", df.isnull().sum().sum(), help="El dataset no contiene valores faltantes.")

    st.markdown("---")

    # Estadisticas descriptivas
    st.markdown("### Estadísticas Descriptivas Relevantes")

    age_mean = df["Age"].mean()
    age_min = df["Age"].min()
    age_max = df["Age"].max()
    age_std = df["Age"].std()

    duration_med = df["Duration"].median()
    duration_min = df["Duration"].min()
    duration_max = df["Duration"].max()

    freq_min = df["Frequency"].min()
    freq_max = df["Frequency"].max()

    # Prevalencias de binarios
    binary_cols = ["Nausea", "Photophobia", "Phonophobia", "Vertigo", "Tinnitus", "Hypoacusis",
                   "Diplopia", "Defect", "Conscience", "Paresthesia"]

    prevalences = df[binary_cols].mean().round(2)

    with st.container():
        st.markdown("""
        <div class="info-box">
            <h4 style="color:#1E3A8A;">Datos Demográficos</h4>
            <p style="color:#2D3748;">
                La edad promedio de los pacientes es de <b>{:.1f} años</b> (rango: {}–{}), con una desviación estándar de {:.1f}.  
                Esto indica una población diversa que incluye tanto jóvenes como adultos mayores.
            </p>
        </div>
        """.format(age_mean, age_min, age_max, age_std), unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h4 style="color:#1E3A8A;">Duración y Frecuencia</h4>
        <p style="color:#2D3748;">
            Los episodios suelen durar entre <b>{} y {} horas</b>, con una mediana de <b>{}</b>.  
            La frecuencia varía entre <b>{} y {} episodios</b> en el periodo analizado.
        </p>
    </div>
    """.format(duration_min, duration_max, duration_med, freq_min, freq_max), unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h4 style="color:#1E3A8A;">🧠 Síntomas Neurológicos y Autonómicos</h4>
        <p style="color:#2D3748;">
            La mayoría de los síntomas binarios presentan prevalencias bajas, acorde a la literatura clínica.  
            Sin embargo, <b>náusea</b>, <b>fotofobia</b> y <b>fonofobia</b> aparecen en casi todos los pacientes  
            (medias entre 0.97 y 0.98).  
            <br><br>
            Síntomas como vértigo, tinnitus, hipoacusia, diplopía, defecto visual, conciencia alterada y parestesias  
            muestran prevalencias <b>menores al 15%</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Problema
    st.markdown("### Problemas detectados en columnas")

    st.markdown("""
    <div class="info-box">
        <h4 style="color:#C53030;"> Columnas a eliminar</h4>
        <ul style="color:#2D3748;">
            <li><b>ATAXIA</b>: contiene únicamente valores de 0 → sin variabilidad.</li>
            <li><b>VISUAL</b>: contiene valores fuera de rango (no binarios).</li>
            <li><b>SENSORY</b>: contiene valores fuera de rango (no binarios).</li>
        </ul>
        <p style="color:#2D3748;">
            Por estas razones, ambas variables fueron removidas para crear <b>migrain_df_clean</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    #HISTOGRAMAS DE LOS DATOS USANDO PESTAÑAS
    st.markdown("---")

    st.markdown("### Distribución de Variables (Histogramas)")
    st.markdown("Explora la distribución individual de cada variable del dataset.")

    all_cols = list(df.columns)
    
    histogram_cols = [
        "Age", "Duration", "Frequency", "Intensity", "Location", 
        "Character", "Nausea", "Vomit", "Phonophobia", "Photophobia",
        "Visual", "Sensory", "Vertigo", "Tinnitus", "Hypoacusis", 
        "Diplopia", "Defect", "Ataxia", "Conscience", "Paresthesia", "DPF"
    ]

    cols_to_plot = [col for col in histogram_cols if col in df.columns]

    # Crear las pestañas
    tabs = st.tabs(cols_to_plot) 

    # Iteramos sobre columnas y pestañas
    for i, col in enumerate(cols_to_plot):
        with tabs[i]:
            st.subheader(f"Distribución de la variable **{col}**")
            
            unique_vals = df[col].nunique()
            
            if unique_vals <= 10:
                nbins = unique_vals
            elif df[col].dtype in ['int64', 'float64'] and unique_vals > 10:
                nbins = min(50, int(df[col].max() - df[col].min()) + 1)
            else:
                nbins = 30

            fig = px.histogram(
                df, 
                x=col, 
                title=f'Histograma de {col}',
                color_discrete_sequence=['#1E3A8A'],
                nbins=nbins
            )
            
            fig.update_layout(
                xaxis_title=col,
                yaxis_title="Conteo de Registros",
                bargap=0.05
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ========== SECCIÓN: ANÁLISIS ESTADÍSTICO ==========
elif section == "Análisis Estadístico":
    st.markdown('<h2 class="section-header">Análisis Estadístico Detallado</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Variables Numéricas", "Variable Type", "Matrices de Correlaciones", "Box plots"])
    
    with tab1:
        st.subheader("Distribución de Variables Numéricas")

        # Lista de variables permitidas
        var_numeric = ["Age", "Duration", "Frequency", "Intensity"]

        # Menú de selección
        selected_var = st.selectbox(
            "Selecciona una variable numérica:",
            var_numeric
        )

        # Gráfica dinámica
        fig = px.histogram(
            df_clean,
            x=selected_var,
            nbins=30,
            marginal="box",
            title=f"Distribución de {selected_var}",
            labels={selected_var: selected_var},
            color_discrete_sequence=["#4C51BF"]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Métricas dinámicas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mínimo", f"{df_clean[selected_var].min():.2f}")
        col2.metric("Máximo", f"{df_clean[selected_var].max():.2f}")
        col3.metric("Promedio", f"{df_clean[selected_var].mean():.2f}")
        col4.metric("Mediana", f"{df_clean[selected_var].median():.2f}")

    with tab2:
        st.subheader("Variable objetivo: Type")
        
        # Distribución completa
        type_dist = df_clean['Type'].value_counts()

        # histograma
        fig = px.bar(
            x=type_dist.index,
            y=type_dist.values,
            title='Distribución de Tipos de Migraña',
            labels={'x': 'Tipo de Migraña', 'y': 'Cantidad'},
            color=type_dist.values,
            color_continuous_scale='magma'
        )
        
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        #grafico de pie
        st.markdown("### Proporción de Tipos")
        fig = px.pie(
            values=type_dist.values,
            names=type_dist.index,
            title='',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

        type_text = """El análisis de la variable TYPE revela un desbalance severo donde Typical aura with
        migraine tiene 247 casos, Migraine without aura menos 50 casos y El resto de las categorías:
        <25 casos cada una Dado que esta categoría concentra más del 60 % de las observaciones,
        este desbalance afectará el desempeño de los modelos y requerirá técnicas de manejo del
        imbalance (oversampling, undersampling o ponderación)."""
        st.markdown(type_text)
        
        # Síntomas más comunes
        st.subheader("Síntomas Más Comunes")
        symptom_cols = ['Nausea', 'Vomit', 'Phonophobia', 'Photophobia', 'Vertigo']
        symptom_prevalence = {}
        
        for col in symptom_cols:
            if col in df_clean.columns:
                symptom_prevalence[col] = df_clean[col].mean() * 100
        
        symptom_df = pd.DataFrame({
            'Síntoma': list(symptom_prevalence.keys()),
            'Prevalencia (%)': list(symptom_prevalence.values())
        }).sort_values('Prevalencia (%)', ascending=False)
        
        fig = px.bar(
            symptom_df,
            x='Síntoma',
            y='Prevalencia (%)',
            title='Prevalencia de Síntomas (%)',
            color='Prevalencia (%)',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Matriz de variables categoricas binarias")
        corr_columnas = df_clean[["Nausea", "Vomit", "Phonophobia", "Photophobia", "Vertigo", "Tinnitus", "Hypoacusis", "Diplopia", "Defect", "Conscience", "Paresthesia", "DPF", "Intensity", "Age"]].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_columnas,
            annot=True,
            cmap="magma",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"label": "Correlación"},
            ax=ax
        )

        plt.tight_layout()
        st.pyplot(fig)

        binary_text = """La matriz de correlación muestra, en general, bajas asociaciones lineales entre los síntomas,
        lo cual es característico en datos clínicos de migraña debido a la alta variabilidad individual
        de los episodios. No obstante, se identifican algunas correlaciones clínicamente relevantes."""
        st.markdown(binary_text)
        
        st.markdown("#### Correlaciones más relevantes")

        st.markdown(""" *Photophobia - Phonophobia (0.70):* Correlación alta y clínicamente coherente, ya que ambos síntomas representan hipersensibilidades
            sensoriales típicas de la migraña.""")
        st.markdown(""" *Photophobia – Nausea (0.47):* Asociación moderada posiblemente relacionada con mareos y malestar inducido por la sensibilidad
            extrema a la luz.""")
        st.markdown(""" *Tinnitus – Vertigo (0.35):* Consistente con la relación entre síntomas vestibulares y auditivos, dado que el sistema auditivo
            está directamente involucrado en el equilibrio.""")
        st.markdown(""" *Hypoacusis – Tinnitus (0.32):* Ambos corresponden a alteraciones auditivas, por lo que su asociación resulta clínicamente esperable.""")
        st.markdown(""" *Tinnitus – Vertigo (0.35):* Consistente con la relación entre síntomas vestibulares y auditivos, dado que el sistema auditivo
            está directamente involucrado en el equilibrio.""")
        st.markdown(""" *Defect – Vertigo (0.30):* Asociación débil-moderada relacionada con alteraciones neurológicas transitorias durante los episodios.""")

        st.subheader("Relación entre Localización y Característica del Dolor")

        location_labels = {
            1: "Unilateral",
            2: "Bilateral",
            3: "Frontal",
            4: "Temporal"
        }
        
        character_labels = {
            1: "Punzante",
            2: "Opresivo",
            3: "Agudo",
            4: "Sordo"
        }

        heatmap_df = pd.crosstab(
        df_clean["Location"].map(location_labels),
        df_clean["Character"].map(character_labels),
        normalize="index"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            heatmap_df,
            annot=True,
            cmap="magma",
            fmt=".2%",
            linewidths=0.8,
            square=True,
            cbar_kws={"label": "Proporción"},
            ax=ax
        )

        ax.set_xlabel("Característica del Dolor")
        ax.set_ylabel("Localización del Dolor")

        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Matriz de correlacion de variables por el tipo de migraña")
        binary_vars = ["Nausea", "Vomit", "Phonophobia", "Photophobia", "Vertigo", "Tinnitus", "Hypoacusis", "Diplopia", "Defect", "Conscience", "Paresthesia", "DPF"]
        migraine_types = sorted(df_clean["Type"].unique())

        all_vars = ["Age", "Duration", "Frequency", "Intensity", "Location", "Character"] + binary_vars
        results = pd.DataFrame(index=all_vars, columns=migraine_types)

        for var in all_vars:
            for migraine_type in migraine_types:
                subset = df[df["Type"] == migraine_type]

                if var in ["Age", "Duration", "Frequency", "Intensity"]:
                    results.loc[var, migraine_type] = subset[var].mean()

                elif var in ["Location", "Character"]:
                    results.loc[var, migraine_type] = (
                        subset[var].mode()[0] if not subset[var].mode().empty else np.nan
                    )

                elif var in binary_vars:
                    results.loc[var, migraine_type] = subset[var].mean() * 100

        results = results.astype(float)

        # Heatmap original
        fig, ax = plt.subplots(figsize=(12, 14))
        sns.heatmap(
            results,
            annot=True,
            cmap="magma",
            linewidths=1,
            fmt=".1f",
            cbar_kws={"label": "Valor"},
            ax=ax
        )

        ax.set_title("Perfil de Variables por Tipo de Migraña")
        ax.set_xlabel("Tipo de Migraña")
        ax.set_ylabel("Variables")

        plt.tight_layout()
        st.pyplot(fig)
        
        txt_sintomas_tipo = """Nota: Es importante destacar que el conjunto de datos presenta un marcado desbalance en la distribución de los tipos de migraña, lo cual implica que las categorías con pocos casos pueden mostrar patrones menos estables o menos generalizables.
            En términos generales, variables como Nausea, Photophobia y Phonophobia muestran valores consistente-mente altos en casi todos los tipos de migraña, lo que coincide con la literatura clínica acerca de su ocurrencia
            habitual en pacientes migrañosos.
            Asimismo, algunas variables sensoriales, como Vertigo, Sensory y DPF, presentan diferencias notables entre los subtipos, lo que sugiere posibles asociaciones específicas entre estos síntomas y ciertos diagnósticos más particulares (por ejemplo, migrañas basilares o hemipléjicas)."""
        
        st.markdown(txt_sintomas_tipo)
        
    with tab4:
        st.markdown("### Box Plots de Variables Numéricas por Tipo de Migraña (Seaborn) 📦")

        # Definir las variables numéricas
        numeric_features = ["Age", "Duration", "Frequency", "Intensity"]
        numeric_features = [col for col in numeric_features if col in df.columns]

        # 2. Crear las pestañas
        tabs_boxplot = st.tabs(numeric_features) 

        # 3. Iterar sobre las columnas numéricas y las pestañas
        for i, col in enumerate(numeric_features):
            with tabs_boxplot[i]:
                st.subheader(f"Box Plot de **{col}** vs. **Type**")
        
                # 4. Generar el Box Plot usando Plotly Express
                fig = px.box(
                    df, 
                    x="Type",  # Variable categórica para el eje X (asumo que se llama 'TYPE' en mayúsculas)
                    y=col,     # Variable numérica para el eje Y
                    title=f"{col} por Tipo de Migraña",
                    # Puedes usar 'color' si quieres que las cajas tengan diferente color según el 'TYPE'
                    color="Type", 
                )
        
                # Personalizar el layout (opcional)
                fig.update_layout(
                    xaxis_title="Tipo de Migraña",
                    yaxis_title=col,
                    xaxis_tickangle=0 # No rotar etiquetas si son pocas
                )
        
                # 5. Mostrar el gráfico en Streamlit
                st.plotly_chart(fig, use_container_width=True)
        
# ========== SECCIÓN: VISUALIZACIONES ==========
elif section == "Visualizaciones":
    st.markdown("## Visualizaciones de Localización Cerebral")
    
    st.markdown("""
        Traté de visualizar las localizaciones de la migraña con una herramienta llamada **Brain Space**. 
        Esta herramienta utiliza el **atlas Schaefer 400**, el cual no ordena las regiones por su posición física 
        (como sería en un dato geográfico), sino por **Redes Funcionales** (Red Visual, Red de Atención, Red de Control, etc.).
        """)
    
    st.markdown("### Visualización de Zona Frontal del Cerebro")
    
    video_url = "frontal-video.mp4" 
    st.video(video_url)
    
    image_path = "frontal.png"
    st.image(image_path, caption='Visualización de las redes activadas al seleccionar la zona frontal.', use_container_width=True)
    
    st.markdown("""
        En la visualización de arriba, al tratar de seleccionar la parte de adelante del cerebro, 
        estas fueron las redes funcionales que se activaron: parecen ser la **Red de Atención Ventral** y una parte de la **Red Límbica**
        """)

elif section == "Bibliografía":
    st.markdown('<h2 class="section-header">Bibliografía</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <p style="color:#2D3748; font-size:1.05rem; line-height: 1.6;">
            Géron, A. (2019). <i>Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow:
            Concepts, Tools, and Techniques to Build Intelligent Systems</i> (2.ª ed.). O’Reilly Media.
        </p>
        <p style="color:#2D3748; font-size:1.05rem; line-height: 1.6;">
            Mayo Clinic. (2023). <i>Migraine – Symptoms &amp; Causes</i>.
            Recuperado de:
            <a href="https://www.mayoclinic.org/es/diseases-conditions/migraine-headache/symptoms-causes/syc-20360201"
               target="_blank">
               Mayo Clinic
            </a>
        </p>
        <p style="color:#2D3748; font-size:1.05rem; line-height: 1.6;">
            Pisa. (2025). <i>Migraña en México: más de 20 millones de personas viven con este padecimiento</i>.
            Recuperado de:
            <a href="https://www.pisa.com.mx/2025/08/07/migrana-en-mexico-mas-de-20-millones-de-personas-viven-con-este-padecimiento/"
               target="_blank">
               Pisa
            </a>
        </p>
        <p style="color:#2D3748; font-size:1.05rem; line-height: 1.6;">
            Secretaría de Salud. (2022). <i>Migraña, enfermedad incapacitante que afecta a 20 millones de personas en México</i>.
            Recuperado de:
            <a href="https://www.gob.mx/salud/prensa/462-migrana-enfermedad-incapacitante-que-afecta-a-20-millones-de-personas-en-mexico"
               target="_blank">
               Secretaría de Salud
            </a>
        </p>
        <p style="color:#2D3748; font-size:1.05rem; line-height: 1.6;">
            Secretaría de Salud. (2024). <i>Sufre migraña 15 % de la población en México: Instituto Nacional de Neurología</i>.
            Recuperado de:
            <a href="https://www.gob.mx/salud/prensa/373-sufre-migrana-15-de-la-poblacion-en-mexico-instituto-nacional-de-neurologia"
               target="_blank">
               Secretaría de Salud
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Dashboard creado para análisis exploratorio de datos de migrañas | Proyecto de Machine Learning")