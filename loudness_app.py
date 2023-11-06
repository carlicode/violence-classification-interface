import streamlit as st
from pydub import AudioSegment
import numpy as np

# Define la función para calcular el loudness por segundo
def calculate_loudness_per_second(audio_file):
    try:
        audio = AudioSegment.from_file(audio_file)
        duration_in_seconds = len(audio) / 1000
        loudness_values = []

        for second in range(int(duration_in_seconds)):
            start_time = second * 1000  # Milisegundos
            end_time = (second + 1) * 1000  # Milisegundos
            segment = audio[start_time:end_time]
            loudness = segment.dBFS
            loudness_values.append(loudness)

        return loudness_values
    except Exception as e:
        return None

# Resto de tu código Streamlit

st.title('Detector de Audio')

uploaded_file = st.file_uploader("Sube un archivo de audio largo", type=["wav"])

if uploaded_file:
    st.write("Procesando el archivo de audio largo...")
    st.audio(uploaded_file, format="audio/wav")
    
    loudness_values = calculate_loudness_per_second(uploaded_file)

    if loudness_values is not None:
        st.write("Loudness por segundo:")
        for second, loudness in enumerate(loudness_values, start=1):
            st.write(f"Segundo {second}: {loudness:.2f} dB")
        
        predictions_per_second = hacer_prediccion(uploaded_file)
        
        st.write("Predicciones por segundo:")
        for i, segment_predictions in enumerate(predictions_per_second):
            st.write(f"Segundo {i + 1}:")
            for label, confidence in segment_predictions:
                st.write(f"  Etiqueta: {label}, Confianza: {confidence:.2f}%")
    else:
        st.write("No se pudo calcular el loudness por segundos.")
