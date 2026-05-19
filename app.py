import streamlit as st
import pandas as pd
import joblib

# =========================
# CARGAR MODELOS
# =========================

modelo_logistico = joblib.load("modelos/modelo_logistico.pkl")
modelo_random_forest = joblib.load("modelos/modelo_random_forest.pkl")

# =========================
# CONFIGURACIÓN DE PÁGINA
# =========================

st.set_page_config(
    page_title="Predicción de Enfermedad Cardíaca",
    page_icon="❤️",
    layout="centered"
)

# =========================
# TÍTULO
# =========================

st.title("❤️ Predicción de Enfermedad Cardíaca")

st.write("""
Este aplicativo utiliza modelos de Machine Learning para predecir
si un paciente presenta riesgo de enfermedad cardíaca.
""")

# =========================
# DATOS DEL ESTUDIANTE
# =========================

st.write("### Información del Proyecto")
st.write("**Nombre:** Mercedes Fiorela Paz Alejandro")
st.write("**Código ISIL:** 75723526")

st.write("[Ver cuaderno Google Colab](https://colab.research.google.com/drive/1dBmYDTp4iFJ6Onoa7XzO0D4Q1lB5YwXY?usp=sharing)")

# =========================
# SELECCIÓN DE MODELO
# =========================

modelo_seleccionado = st.selectbox(
    "Seleccione el modelo:",
    ["Regresión Logística", "Random Forest"]
)

# =========================
# INGRESO DE DATOS
# =========================

st.write("## Ingrese los datos del paciente")

age = st.number_input("Edad", 20, 100, 50)

sex = st.selectbox(
    "Sexo",
    [0,1],
    format_func=lambda x: "Femenino" if x == 0 else "Masculino"
)

cp = st.slider("Tipo de dolor de pecho (cp)", 0, 3, 1)

trestbps = st.number_input(
    "Presión arterial en reposo",
    80,
    220,
    120
)

chol = st.number_input(
    "Colesterol",
    100,
    600,
    200
)

fbs = st.selectbox(
    "Azúcar en sangre > 120 mg/dl",
    [0,1]
)

restecg = st.slider(
    "Resultado electrocardiográfico",
    0,
    2,
    1
)

thalach = st.number_input(
    "Frecuencia cardíaca máxima",
    60,
    220,
    150
)

exang = st.selectbox(
    "Angina inducida por ejercicio",
    [0,1]
)

oldpeak = st.slider(
    "Depresión ST",
    0.0,
    10.0,
    1.0
)

slope = st.slider(
    "Pendiente del segmento ST",
    0,
    2,
    1
)

ca = st.slider(
    "Número de vasos principales",
    0,
    4,
    0
)

thal = st.slider(
    "Thal",
    0,
    3,
    1
)

# =========================
# DATAFRAME
# =========================

datos = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})

# =========================
# PREDICCIÓN
# =========================

if st.button("Realizar Predicción"):

    if modelo_seleccionado == "Regresión Logística":
        prediccion = modelo_logistico.predict(datos)

    else:
        prediccion = modelo_random_forest.predict(datos)

if prediccion[0] == 0:
    st.error("⚠️ El modelo predice mayor riesgo de enfermedad cardíaca.")
else:
    st.success("✅ El modelo predice menor riesgo de enfermedad cardíaca.")
