import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import pandas as pd
import io

from calculate_loudness import calculate_loudness_per_second


model = tf.keras.models.load_model('experimento.h5')
labels = ["crying", "glass_breaking", "screams", "gun_shot", "people_talking"]

label_encoder = LabelEncoder()
label_encoder.fit(labels)  

st.title('Detector de Audio')

uploaded_file = st.file_uploader("Sube un archivo de audio largo", type=["wav"])

def hacer_prediccion(audio_data):
    audio, _ = librosa.load(audio_data, sr=16000)

    segment_duration = 1  # Duración
    samples_per_segment = int(16000 * segment_duration)
    num_segments = len(audio) // samples_per_segment

    predictions = []

    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = (i + 1) * samples_per_segment
        segment = audio[start_sample:end_sample]

        # Espectrograma
        spectrogram = librosa.feature.melspectrogram(y=segment, sr=16000)
        input_data = np.expand_dims(spectrogram, axis=0)

        prediction = model.predict(input_data)[0]

        decoded_labels = label_encoder.inverse_transform(range(len(labels)))
        
        segment_predictions = []
        for label, conf in zip(decoded_labels, prediction):
            if conf >= 0.75:
                segment_predictions.append((label, conf * 100))
        
        if not segment_predictions:
            segment_predictions.append(("No se identificaron etiquetas", 0))
        
        predictions.append(segment_predictions)

    return predictions

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