import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import pandas as pd
from pydub import AudioSegment 


model = tf.keras.models.load_model('experimento.h5')
labels = ["crying", "glass_breaking", "screams", "gun_shot", "people_talking"]

label_encoder = LabelEncoder()
label_encoder.fit(labels)

st.title('Detector de Audio')

uploaded_file = st.file_uploader("Sube un archivo de audio largo", type=["wav"])

# Función para calcular el loudness
def calcular_loudness(segment):
    audio = AudioSegment.from_numpy_array(segment, frame_rate=16000, sample_width=2, channels=1)
    loudness = audio.rms
    return loudness

def hacer_prediccion(audio_data):
    audio, _ = librosa.load(audio_data, sr=16000)

    segment_duration = 1
    samples_per_segment = int(16000 * segment_duration)
    num_segments = len(audio) // samples_per_segment

    data = []

    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = (i + 1) * samples_per_segment
        segment = audio[start_sample:end_sample]

        # Calcular el loudness del segmento
        loudness = calcular_loudness(segment)

        spectrogram = librosa.feature.melspectrogram(y=segment, sr=16000)
        input_data = np.expand_dims(spectrogram, axis=0)

        prediction = model.predict(input_data)[0]

        decoded_labels = label_encoder.inverse_transform(range(len(labels)))

        segment_predictions = []
        for label, conf in zip(decoded_labels, prediction):
            if conf >= 0.75:
                segment_predictions.append((i + 1, label, conf * 100, loudness))

        if not segment_predictions:
            segment_predictions.append((i + 1, "No se identificaron etiquetas", 0, loudness))

        data.extend(segment_predictions)

    return data

if uploaded_file:
    st.write("Procesando el archivo de audio largo...")
    st.audio(uploaded_file, format="audio/wav")
    predictions_per_second = hacer_prediccion(uploaded_file)

    st.write("Predicciones por segundo:")
    predictions_df = pd.DataFrame(predictions_per_second, columns=["Segundo", "Etiqueta", "Confianza", "Loudness"])
    st.table(predictions_df)
