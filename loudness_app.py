import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import pandas as pd
from pydub import AudioSegment
from io import BytesIO
import matplotlib.pyplot as plt
import speech_recognition as sr

model = tf.keras.models.load_model('experimento_7.h5')
labels = ["crying", "glass_breaking", "screams", "gun_shot", "people_talking"]

loudness_values = []  

label_encoder = LabelEncoder()
label_encoder.fit(labels)

st.title('Detector de Audio')

uploaded_file = st.file_uploader("Subir un archivo de audio largo", type=["wav"])

def calcular_loudness(segment):
    buffer = BytesIO()
    sf.write(buffer, segment, 16000, subtype='PCM_16', format='wav')
    buffer.seek(0)

    audio = AudioSegment.from_wav(buffer)

    loudness = audio.rms
    return loudness

def transcribir_audio(segment):
    buffer = BytesIO()
    sf.write(buffer, segment, 16000, subtype='PCM_16', format='wav')
    buffer.seek(0)

    audio = AudioSegment.from_wav(buffer)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio.export(format="wav")) as source:
        audio_data = recognizer.record(source, duration=3)  # Transcribir 5 segundos del audio
        try:
            transcription = recognizer.recognize_google(audio_data, language="es-ES")
            return transcription
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return "Error"

def hacer_prediccion(audio_data):
    audio, _ = librosa.load(audio_data, sr=16000)

    segment_duration = 2
    samples_per_segment = int(16000 * segment_duration)
    num_segments = len(audio) // samples_per_segment

    data = []

    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = (i + 1) * samples_per_segment
        segment = audio[start_sample:end_sample]

        loudness = calcular_loudness(segment)
        loudness_values.append(loudness)

        transcription = transcribir_audio(segment)

        spectrogram = librosa.feature.melspectrogram(y=segment, sr=16000)
        input_data = np.expand_dims(spectrogram, axis=0)

        prediction = model.predict(input_data)[0]

        decoded_labels = label_encoder.inverse_transform(range(len(labels)))

        segment_predictions = []
        for label, conf in zip(decoded_labels, prediction):
            if conf >= 0.4:
                segment_predictions.append(("people_talking" if label == "people_talking" else label, conf * 100, loudness, transcription, segment))

        if not segment_predictions:
            segment_predictions.append(("No se identificaron etiquetas", 0, loudness, transcription, segment))

        data.extend(segment_predictions)

    return data

def crear_grafico_loudness(loudness_values):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(loudness_values)), loudness_values)
    plt.xlabel("Tiempo (segmentos)")
    plt.ylabel("Loudness")
    st.pyplot(plt)

if uploaded_file:
    st.write("Procesando el archivo de audio largo...")

    st.audio(uploaded_file, format="audio/wav")
    predictions_per_second = hacer_prediccion(uploaded_file)
    
    st.write("Predicciones por segundo:")
    predictions_df = pd.DataFrame(predictions_per_second, columns=["Etiqueta", "Confianza", "Loudness", "Texto", "Fragmento"])
    
    predictions_df["Análisis"] = ["people_talking" if texto else "" for texto in predictions_df["Texto"]]
    
    # Mostrar el fragmento de audio en la tabla
    for index, row in predictions_df.iterrows():
        st.audio(row["Fragmento"], sample_rate=16000, format="audio/wav")
        st.table(row[["Etiqueta", "Confianza", "Loudness", "Texto", "Análisis"]])

    crear_grafico_loudness(loudness_values)
    embeddings_db = predictions_df[["Loudness", "Texto"]]
    embeddings_db["Embeddings"] = None
    embeddings_db["Distancia"] = None

    st.table(embeddings_db)
