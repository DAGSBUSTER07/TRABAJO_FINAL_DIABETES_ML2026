import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(page_title="Predicción de Diabetes | ML", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.encabezado {background: linear-gradient(90deg, #1f4e79 0%, #2e86c1 100%);
             padding: 1.4rem 1.8rem; border-radius: 12px; color: white; margin-bottom: 1rem;}
.encabezado h1 {color: white; margin: 0; font-size: 2rem;}
.encabezado p {color: #e8f1fa; margin: 0.3rem 0 0 0; font-size: 1rem;}
.tarjeta {padding: 1.1rem 1.3rem; border-radius: 12px; text-align: center; color: white;}
.tarjeta h2 {color: white; margin: 0;}
.tarjeta p {margin: 0; font-size: 0.95rem;}
.positivo {background-color: #c0392b;}
.negativo {background-color: #1e8449;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTES DEL PROYECTO
# ============================================================
VARIABLES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Rangos clínicos válidos (reglas de calidad del ETL)
REGLAS = {'Pregnancies': (0, 20), 'Glucose': (40, 300), 'BloodPressure': (30, 130),
          'SkinThickness': (2, 60), 'Insulin': (1, 600), 'BMI': (12, 70),
          'DiabetesPedigreeFunction': (0.05, 2.50), 'Age': (18, 110)}

# Nombre en pantalla, unidad, valor inicial (mediana del dataset) y paso del control
DESCRIPCION = {
    'Pregnancies':              ('Número de embarazos', 'conteo', 7.0, 1.0),
    'Glucose':                  ('Glucosa plasmática', 'mg/dL', 121.3, 1.0),
    'BloodPressure':            ('Presión arterial diastólica', 'mmHg', 70.1, 1.0),
    'SkinThickness':            ('Pliegue cutáneo del tríceps', 'mm', 21.8, 0.5),
    'Insulin':                  ('Insulina sérica (2 horas)', 'µU/mL', 57.3, 1.0),
    'BMI':                      ('Índice de masa corporal', 'kg/m²', 31.81, 0.1),
    'DiabetesPedigreeFunction': ('Antecedente familiar de diabetes', 'índice', 0.316, 0.01),
    'Age':                      ('Edad', 'años', 34.0, 1.0),
}

# Columnas del dataset original (Diabetes_Patients.csv) sin Outcome, en su orden original
COLUMNAS_ORIGINALES = ['RecordID', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                       'BMI', 'DiabetesPedigreeFunction', 'Age', 'BirthYear', 'Country', 'LabOrderCode',
                       'FullName', 'Phone', 'ZodiacSign', 'FavoriteColor', 'Hobby']

# Variables que se eliminan en el ETL y no entran al modelo
COLUMNAS_ELIMINADAS = ['RecordID', 'BirthYear', 'Country', 'LabOrderCode', 'FullName',
                       'Phone', 'ZodiacSign', 'FavoriteColor', 'Hobby']

PAISES = ['Colombia', 'Chile', 'Peru', 'Mexico']
SIGNOS = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio',
          'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
COLORES = ['Black', 'Blue', 'Gray', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
HOBBIES = ['Cinema', 'Cooking', 'Gaming', 'Gardening', 'Hiking', 'Music', 'Photography',
           'Reading', 'Sports', 'Travel']

# Resultados de la optimización de hiperparámetros (validación cruzada, 5 pliegues)
RESULTADOS_MODELOS = pd.DataFrame({
    'Modelo': ['KNN', 'Árbol de Decisión', 'SVM', 'Red Neuronal', 'Random Forest', 'Stacking', 'Boosting', 'Bagging'],
    'F1 Score': [0.5623, 0.5432, 0.5317, 0.5317, 0.5268, 0.5106, 0.4972, 0.4965],
    'Sensibilidad': [0.5777, 0.5658, 0.5608, 0.5488, 0.5358, 0.5241, 0.5000, 0.4829],
    'Precisión': [0.5497, 0.5304, 0.5091, 0.5231, 0.5219, 0.5005, 0.4978, 0.5214],
    'Exactitud': [0.5533, 0.5354, 0.5120, 0.5300, 0.5208, 0.5031, 0.4969, 0.5208],
})

# ============================================================
# CARGA DE ARTEFACTOS DEL MODELO
# ============================================================
@st.cache_resource
def load_artifacts():
    try:
        columnas = joblib.load('columnas_modelo.joblib')
        scaler = joblib.load('min_max_scaler.joblib')
        model = joblib.load('knn_optimizado.joblib')
        return columnas, scaler, model
    except Exception as e:
        st.error(f"Error al cargar los archivos .joblib: {e}")
        return None, None, None

columnas_modelo, scaler, model = load_artifacts()


def preprocesar(df_entrada):
    """Aplica el mismo preprocesamiento del ETL: reglas clínicas, escalado e imputación."""
    datos = df_entrada[columnas_modelo].apply(pd.to_numeric, errors='coerce').astype(float)
    # Valores fuera de los rangos clínicos -> faltantes
    for col, (minimo, maximo) in REGLAS.items():
        datos.loc[(datos[col] < minimo) | (datos[col] > maximo), col] = np.nan
    # Escalado con el MinMaxScaler entrenado en el ETL
    escalados = pd.DataFrame(scaler.transform(datos), columns=columnas_modelo, index=datos.index)
    # Faltantes -> valor inicial (mediana del dataset) ya escalado
    medianas = pd.DataFrame([{v: DESCRIPCION[v][2] for v in columnas_modelo}])
    medianas_esc = pd.Series(scaler.transform(medianas[columnas_modelo])[0], index=columnas_modelo)
    return escalados.fillna(medianas_esc), datos.isna().sum().sum()


def grafica_gauge(probabilidad):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilidad * 100,
        number={'suffix': ' %', 'font': {'size': 40}},
        title={'text': 'Probabilidad de diabetes'},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': '#1f4e79'},
               'steps': [{'range': [0, 50], 'color': '#d5f5e3'},
                         {'range': [50, 100], 'color': '#fadbd8'}],
               'threshold': {'line': {'color': 'black', 'width': 3}, 'value': 50}}))
    fig.update_layout(height=280, margin=dict(l=40, r=40, t=50, b=10))
    return fig


def plantilla_excel():
    # Plantilla con TODAS las columnas del dataset original (sin Outcome)
    ejemplo = pd.DataFrame([
        {'RecordID': 'REC00769', 'Pregnancies': 2, 'Glucose': 165, 'BloodPressure': 80, 'SkinThickness': 30,
         'Insulin': 120, 'BMI': 34.5, 'DiabetesPedigreeFunction': 0.6, 'Age': 45, 'BirthYear': 1981,
         'Country': 'Colombia', 'LabOrderCode': 5210, 'FullName': 'Person_769', 'Phone': 3001234567,
         'ZodiacSign': 'Leo', 'FavoriteColor': 'Blue', 'Hobby': 'Reading'},
        {'RecordID': 'REC00770', 'Pregnancies': 1, 'Glucose': 95, 'BloodPressure': 68, 'SkinThickness': 20,
         'Insulin': 60, 'BMI': 24.0, 'DiabetesPedigreeFunction': 0.25, 'Age': 28, 'BirthYear': 1998,
         'Country': 'Colombia', 'LabOrderCode': 7731, 'FullName': 'Person_770', 'Phone': 3109876543,
         'ZodiacSign': 'Virgo', 'FavoriteColor': 'Green', 'Hobby': 'Sports'},
    ])[COLUMNAS_ORIGINALES]
    buffer = io.BytesIO()
    ejemplo.to_excel(buffer, index=False)
    return buffer.getvalue()


# ============================================================
# ENCABEZADO
# ============================================================
st.markdown("""
<div class="encabezado">
  <h1>🩺 Predicción del diagnóstico de diabetes tipo 2</h1>
  <p>Modelo de clasificación con Machine Learning a partir de variables clínicas y de laboratorio</p>
  <p><b>Integrantes:</b> Diego Alejandro Gómez Samudio · Juan Esteban Hernández Cardona &nbsp;|&nbsp;
     <b>Módulo:</b> Machine Learning &nbsp;|&nbsp; <b>Fase:</b> Despliegue</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# BARRA LATERAL
# ============================================================
with st.sidebar:
    st.header("📌 Sobre el proyecto")
    st.write("**Institución Universitaria EAM**")
    st.write("Módulo de Machine Learning — Trabajo final")
    st.divider()
    st.subheader("🤖 Modelo desplegado")
    st.write("**KNN optimizado** (5 vecinos, ponderados por distancia)")
    c1, c2 = st.columns(2)
    c1.metric("F1 Score", "0,562")
    c2.metric("Sensibilidad", "0,578")
    c1.metric("Precisión", "0,550")
    c2.metric("Exactitud", "0,553")
    st.caption("Métricas promedio de validación cruzada estratificada de 5 pliegues.")
    st.divider()
    st.subheader("🧭 Cómo usarla")
    st.write("1. **Paciente individual:** ingresa los datos y presiona *Predecir*.")
    st.write("2. **Archivo Excel:** sube varios pacientes y descarga los resultados.")
    st.write("3. **Acerca del modelo:** revisa el proceso y la comparación de modelos.")
    st.divider()
    st.caption("⚠️ Herramienta académica. No reemplaza el criterio médico ni la prueba confirmatoria.")

if not (columnas_modelo and scaler is not None and model is not None):
    st.warning("Asegúrate de que los archivos 'columnas_modelo.joblib', 'min_max_scaler.joblib' y "
               "'knn_optimizado.joblib' estén en la misma carpeta que app.py.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["👤 Paciente individual", "📂 Carga de archivo Excel (lote)", "📊 Acerca del modelo"])

# ============================================================
# PESTAÑA 1: PREDICCIÓN INDIVIDUAL
# ============================================================
with tab1:
    st.subheader("Registro completo del paciente")
    st.write("Se ingresan todas las variables del dataset original. Las variables de identificación y "
             "adicionales se registran, pero el modelo solo usa las 8 variables clínicas, igual que en el ETL.")

    with st.form("formulario_individual"):
        valores = {}

        st.markdown("##### 🪪 Identificación y datos adicionales *(no usados por el modelo)*")
        i1, i2, i3 = st.columns(3)
        valores['RecordID'] = i1.text_input("ID del registro (RecordID)", value="REC00769")
        valores['FullName'] = i2.text_input("Nombre completo (FullName)", value="Person_769")
        valores['Phone'] = i3.text_input("Teléfono (Phone)", value="3001234567")
        i4, i5, i6 = st.columns(3)
        valores['Country'] = i4.selectbox("País (Country)", PAISES)
        valores['BirthYear'] = i5.number_input("Año de nacimiento (BirthYear)", min_value=1915,
                                               max_value=2008, value=1992, step=1)
        valores['LabOrderCode'] = i6.number_input("Código de orden de laboratorio (LabOrderCode)",
                                                  min_value=1000, max_value=9999, value=5429, step=1)
        i7, i8, i9 = st.columns(3)
        valores['ZodiacSign'] = i7.selectbox("Signo zodiacal (ZodiacSign)", SIGNOS, index=3)
        valores['FavoriteColor'] = i8.selectbox("Color favorito (FavoriteColor)", COLORES, index=6)
        valores['Hobby'] = i9.selectbox("Pasatiempo (Hobby)", HOBBIES, index=7)

        st.markdown("##### 🩺 Variables clínicas *(usadas por el modelo)*")
        st.caption("Los valores iniciales corresponden a la mediana del dataset.")
        col_a, col_b = st.columns(2)
        for i, var in enumerate(columnas_modelo):
            nombre, unidad, inicial, paso = DESCRIPCION[var]
            minimo, maximo = REGLAS[var]
            with (col_a if i % 2 == 0 else col_b):
                valores[var] = st.number_input(f"{nombre} ({unidad}) — {var}", min_value=float(minimo),
                                               max_value=float(maximo), value=float(inicial),
                                               step=paso, key=f"ind_{var}")
        enviar = st.form_submit_button("🔍 Predecir", type="primary", width="stretch")

    if enviar:
        # Registro completo con las mismas columnas del dataset original
        df_input = pd.DataFrame([valores])[COLUMNAS_ORIGINALES]
        df_procesado, _ = preprocesar(df_input)

        prediccion = int(model.predict(df_procesado)[0])
        probabilidad = float(model.predict_proba(df_procesado)[0][1])

        col1, col2 = st.columns([1, 1])
        with col1:
            if prediccion == 1:
                st.markdown(f"""<div class="tarjeta positivo"><p>Resultado del modelo</p>
                <h2>CON DIABETES</h2><p>Se recomienda ordenar la prueba confirmatoria</p></div>""",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="tarjeta negativo"><p>Resultado del modelo</p>
                <h2>SIN DIABETES</h2><p>Sin indicación de prueba confirmatoria según el modelo</p></div>""",
                            unsafe_allow_html=True)
            st.write("")
            m1, m2 = st.columns(2)
            m1.metric("Probabilidad sin diabetes", f"{1 - probabilidad:.1%}")
            m2.metric("Probabilidad con diabetes", f"{probabilidad:.1%}")
        with col2:
            st.plotly_chart(grafica_gauge(probabilidad), width="stretch")

        st.subheader("Perfil del paciente (valores escalados entre 0 y 1)")
        perfil = pd.DataFrame({'Variable': [DESCRIPCION[v][0] for v in columnas_modelo],
                               'Valor escalado': df_procesado.iloc[0].values})
        fig_perfil = px.bar(perfil, x='Valor escalado', y='Variable', orientation='h',
                            range_x=[0, 1], color='Valor escalado', color_continuous_scale='Blues')
        fig_perfil.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_perfil, width="stretch")

        with st.expander("Ver el registro completo y el preprocesamiento aplicado"):
            st.write("**1. Registro completo ingresado (17 variables del dataset original):**")
            st.dataframe(df_input, width="stretch")
            st.write("**2. Variables eliminadas en el preprocesamiento (no aportan al modelo):** " +
                     ", ".join(f"`{c}`" for c in COLUMNAS_ELIMINADAS))
            st.write("**3. Variables clínicas escaladas que recibe el modelo:**")
            st.dataframe(df_procesado.round(4), width="stretch")

# ============================================================
# PESTAÑA 2: CARGA DE ARCHIVO EXCEL
# ============================================================
with tab2:
    st.subheader("Predicción por lote desde un archivo Excel")
    st.write("Sube un archivo con las mismas columnas del dataset original (descarga la plantilla). "
             "Las columnas obligatorias para predecir son: " + ", ".join(f"`{v}`" for v in columnas_modelo) +
             ". Las demás (" + ", ".join(f"`{c}`" for c in COLUMNAS_ELIMINADAS) +
             ") se conservan en los resultados, pero no las usa el modelo.")
    st.download_button("📥 Descargar plantilla de Excel", data=plantilla_excel(),
                       file_name="plantilla_pacientes.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    uploaded_file = st.file_uploader("Selecciona un archivo Excel (.xlsx) o CSV", type=["xlsx", "csv"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                df_uploaded = pd.read_csv(uploaded_file)
            else:
                df_uploaded = pd.read_excel(uploaded_file)

            st.write(f"**Vista previa** ({len(df_uploaded)} registros):")
            st.dataframe(df_uploaded.head(), width="stretch")

            missing_cols = [c for c in columnas_modelo if c not in df_uploaded.columns]
            faltan_opcionales = [c for c in COLUMNAS_ELIMINADAS if c not in df_uploaded.columns]
            if faltan_opcionales and not missing_cols:
                st.caption("Columnas del dataset original que no vienen en el archivo (no afectan la predicción): "
                           + ", ".join(faltan_opcionales))
            if missing_cols:
                st.error(f"El archivo no contiene las siguientes columnas requeridas: {missing_cols}")
            elif st.button("⚙️ Procesar y predecir en lote", type="primary"):
                df_procesado_lote, n_corregidos = preprocesar(df_uploaded)
                if n_corregidos > 0:
                    st.info(f"Se encontraron {n_corregidos} valores vacíos o fuera de rango; "
                            "se reemplazaron por la mediana del dataset.")

                predicciones = model.predict(df_procesado_lote)
                probabilidades = model.predict_proba(df_procesado_lote)[:, 1]

                resultados = df_uploaded.copy()
                resultados['Probabilidad_Diabetes'] = probabilidades.round(3)
                resultados['Prediccion_Outcome'] = predicciones
                resultados['Diagnostico_Predicho'] = np.where(predicciones == 1, 'Con diabetes', 'Sin diabetes')

                st.success("¡Predicciones calculadas exitosamente!")

                k1, k2, k3 = st.columns(3)
                k1.metric("Pacientes evaluados", len(resultados))
                k2.metric("Con diabetes (predicho)", int(predicciones.sum()))
                k3.metric("Sin diabetes (predicho)", int(len(predicciones) - predicciones.sum()))

                g1, g2 = st.columns(2)
                with g1:
                    conteo = resultados['Diagnostico_Predicho'].value_counts().reset_index()
                    conteo.columns = ['Diagnóstico', 'Pacientes']
                    fig_dona = px.pie(conteo, names='Diagnóstico', values='Pacientes', hole=0.5,
                                      color='Diagnóstico',
                                      color_discrete_map={'Con diabetes': '#c0392b', 'Sin diabetes': '#1e8449'},
                                      title='Distribución de diagnósticos predichos')
                    st.plotly_chart(fig_dona, width="stretch")
                with g2:
                    fig_hist = px.histogram(resultados, x='Probabilidad_Diabetes', nbins=10, range_x=[0, 1],
                                            title='Distribución de probabilidades',
                                            color_discrete_sequence=['#2e86c1'])
                    fig_hist.update_layout(xaxis_title='Probabilidad de diabetes', yaxis_title='Pacientes')
                    st.plotly_chart(fig_hist, width="stretch")

                st.subheader("Resultados de las predicciones")
                st.dataframe(resultados.sort_values('Probabilidad_Diabetes', ascending=False), width="stretch")

                # Si el archivo trae el diagnóstico real, se compara con lo predicho
                if 'Outcome' in df_uploaded.columns:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                    reales = pd.to_numeric(df_uploaded['Outcome'], errors='coerce')
                    validos = reales.notna()
                    if validos.sum() > 0:
                        st.subheader("Comparación con el diagnóstico real (columna Outcome)")
                        y_real, y_pred = reales[validos].astype(int), predicciones[validos.values]
                        v1, v2, v3, v4 = st.columns(4)
                        v1.metric("Exactitud", f"{accuracy_score(y_real, y_pred):.3f}")
                        v2.metric("Precisión", f"{precision_score(y_real, y_pred, zero_division=0):.3f}")
                        v3.metric("Sensibilidad", f"{recall_score(y_real, y_pred, zero_division=0):.3f}")
                        v4.metric("F1 Score", f"{f1_score(y_real, y_pred, zero_division=0):.3f}")
                        matriz = confusion_matrix(y_real, y_pred, labels=[0, 1])
                        fig_mc = px.imshow(matriz, text_auto=True, color_continuous_scale='Blues',
                                           x=['Sin diabetes', 'Con diabetes'], y=['Sin diabetes', 'Con diabetes'],
                                           labels=dict(x='Predicción', y='Valor real', color='Pacientes'),
                                           title='Matriz de confusión')
                        fig_mc.update_layout(height=380, coloraxis_showscale=False)
                        st.plotly_chart(fig_mc, width="stretch")

                # Descargas de resultados
                buffer = io.BytesIO()
                resultados.to_excel(buffer, index=False)
                d1, d2 = st.columns(2)
                d1.download_button("📥 Descargar resultados en Excel", data=buffer.getvalue(),
                                   file_name="predicciones_lote.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                d2.download_button("📥 Descargar resultados en CSV",
                                   data=resultados.to_csv(index=False).encode('utf-8'),
                                   file_name="predicciones_lote.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")

# ============================================================
# PESTAÑA 3: ACERCA DEL MODELO
# ============================================================
with tab3:
    st.subheader("Proceso de construcción del modelo")
    p1, p2, p3, p4 = st.columns(4)
    p1.info("**1. ETL**\n\nLimpieza de 768 registros, eliminación de 9 variables sin valor predictivo y corrección de valores imposibles.")
    p2.info("**2. Preparación**\n\nTratamiento de atípicos (IQR), imputación con KNN, escalado MinMax y balanceo con undersampling.")
    p3.info("**3. Modelado**\n\n8 modelos evaluados con validación cruzada y optimizados con GridSearchCV.")
    p4.info("**4. Despliegue**\n\nEl mejor modelo (KNN) se publica en esta aplicación de Streamlit.")

    st.subheader("Comparación de modelos optimizados")
    metrica = st.radio("Métrica a visualizar", ['F1 Score', 'Sensibilidad', 'Precisión', 'Exactitud'], horizontal=True)
    orden = RESULTADOS_MODELOS.sort_values(metrica, ascending=True)
    fig_modelos = px.bar(orden, x=metrica, y='Modelo', orientation='h', text=metrica, range_x=[0, 1],
                         color=orden['Modelo'].eq('KNN').map({True: 'Modelo seleccionado', False: 'Otros modelos'}),
                         color_discrete_map={'Modelo seleccionado': '#1f4e79', 'Otros modelos': '#aed6f1'})
    fig_modelos.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_modelos.update_layout(height=380, legend_title_text='', margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_modelos, width="stretch")
    st.dataframe(RESULTADOS_MODELOS.set_index('Modelo'), width="stretch")

    st.subheader("¿Por qué F1 y sensibilidad?")
    st.write("La **sensibilidad** es la métrica principal porque se trata de un problema de tamizaje: "
             "un falso negativo (un paciente con diabetes que el modelo clasifica como sano) tiene un costo "
             "clínico mayor que un falso positivo. Para seleccionar los hiperparámetros se usó el **F1 Score**, "
             "que equilibra sensibilidad y precisión, con el fin de evitar modelos que obtengan alta sensibilidad "
             "simplemente clasificando a todos los pacientes como positivos. El KNN optimizado obtuvo el mayor F1 "
             "y también la mayor sensibilidad entre los 8 modelos.")

    with st.expander("Variables utilizadas por el modelo"):
        st.table(pd.DataFrame({
            'Variable': columnas_modelo,
            'Descripción': [DESCRIPCION[v][0] for v in columnas_modelo],
            'Unidad': [DESCRIPCION[v][1] for v in columnas_modelo],
            'Rango válido': [f"{REGLAS[v][0]} – {REGLAS[v][1]}" for v in columnas_modelo],
        }))

st.divider()
st.caption("Trabajo final — Módulo de Machine Learning · Diego Gómez y Juan Hernández · "
           "Dataset: Diabetes_Patients.csv · Aplicación desarrollada con Streamlit")
