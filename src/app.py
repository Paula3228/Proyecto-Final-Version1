from utils import db_connect
engine = db_connect()

# your code here
import streamlit as st
import pandas as pd

st.title("📊 Análisis de Sentimientos en Español")
st.subheader("Explora, analiza y visualiza datos de comentarios.")

# Cargar el dataset desde la carpeta data/raw
file_path = "./data/raw/sentiment_analysis_dataset.csv"

try:
    df = pd.read_csv(file_path, encoding="latin1")
    st.write("Dataset cargado exitosamente:")
    st.write(df.head())  # Mostrar las primeras filas
except FileNotFoundError:
    st.error("Error al cargar el archivo. Verifica si está en la ruta ./data/raw/")


st.subheader("Predicción de Sentimientos")
comentario_usuario = st.text_input("Escribe tu comentario aquí:")
if st.button("Predecir Sentimiento"):
    if comentario_usuario:
        # Preprocesar el comentario (limpieza, vectorización, etc.)
        # Suponemos que tienes una función llamada `preprocesar_comentario`
        comentario_preprocesado = preprocesar_comentario(comentario_usuario)
        
        # Hacer la predicción
        prediccion = modelo.predict([comentario_preprocesado])
        probabilidad = modelo.predict_proba([comentario_preprocesado])[0]  # Si el modelo soporta probabilidades
        
        # Mostrar resultados
        st.write(f"Sentimiento Predicho: {'Positivo' if prediccion[0] == 1 else 'Negativo'}")
        st.write(f"Probabilidad de Sentimiento Positivo: {probabilidad[1] * 100:.2f}%")
        st.write(f"Probabilidad de Sentimiento Negativo: {probabilidad[0] * 100:.2f}%")
    else:
        st.warning("Por favor ingresa un comentario.")

