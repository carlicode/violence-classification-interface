import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import io

model = tf.keras.models.load_model('experimento.h5')
labels = ["crying", "glass_breaking", "screams", "gun_shot", "people_talking"]

label_encoder = LabelEncoder()
label_encoder.fit(labels)  

st.title('Detector de Audio')

uploaded_file = st.file_uploader("Sube un archivo de audio largo", type=["wav"])

# Función para realizar predicciones en segmentos de un segundo
def hacer_prediccion(audio_data):
    audio, _ = librosa.load(audio_data, sr=16000)

    # Dividir el audio en segmentos de 1 segundo
    segment_duration = 1  # Duración de cada segmento en segundos
    samples_per_segment = int(16000 * segment_duration)
    num_segments = len(audio) // samples_per_segment

    predictions = []

    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = (i + 1) * samples_per_segment
        segment = audio[start_sample:end_sample]

        # Calcular el espectrograma para el segmento
        spectrogram = librosa.feature.melspectrogram(y=segment, sr=16000)
        input_data = np.expand_dims(spectrogram, axis=0)

        # Realizar la predicción utilizando el modelo cargado
        prediction = model.predict(input_data)[0]

        # Decodificar la salida one-hot a etiquetas de texto
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
    predictions_per_second = hacer_prediccion(uploaded_file)

    st.write("Predicciones por segundo:")
    for i, segment_predictions in enumerate(predictions_per_second):
        st.write(f"Segundo {i + 1}:")
        for label, confidence in segment_predictions:
            st.write(f"  Etiqueta: {label}, Confianza: {confidence:.2f}%")
