import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import io

model = tf.keras.models.load_model('experimento4.h5')
labels = ["crying", "glass_breaking", "screams", "gun_shot", "people_talking"]

label_encoder = LabelEncoder()
label_encoder.fit(labels)  

st.title('Detector de Audio')

uploaded_file = st.file_uploader("Sube un archivo de audio", type=["wav"])

# Función para realizar predicciones
def hacer_prediccion(audio_data):
    audio, _ = librosa.load(audio_data, sr=16000, duration=1)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=16000)
    input_data = np.expand_dims(spectrogram, axis=0)

    prediction = model.predict(input_data)[0]

    # Decodificar la salida one-hot a etiquetas de texto
    decoded_labels = label_encoder.inverse_transform(range(len(labels)))
    
    predictions = []
    for label, conf in zip(decoded_labels, prediction):
        predictions.append((label, conf * 100))

    return predictions

if uploaded_file:
    st.write("Reproduciendo archivo de audio...")
    st.audio(uploaded_file, format="audio/wav")
    predictions = hacer_prediccion(uploaded_file)

    st.write("Predicciones:")
    for label, confidence in predictions:
        st.write(f"Etiqueta: {label}, Confianza: {confidence:.2f}%")
#http://localhost:8501/