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

model = tf.keras.models.load_model('experimento.h5')
labels = ["crying", "glass_breaking", "screams", "gun_shot", "people_talking"]

loudness_values = []  # Almacenar los valores de loudness a lo largo del tiempo

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
        audio_data = recognizer.record(source, duration=5)  # Transcribir 5 segundos del audio
        try:
            transcription = recognizer.recognize_google(audio_data, language="es-ES")
            return transcription
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return "Error"

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

        loudness = calcular_loudness(segment)
        loudness_values.append(loudness)

        transcription = transcribir_audio(segment)

        spectrogram = librosa.feature.melspectrogram(y=segment, sr=16000)
        input_data = np.expand_dims(spectrogram, axis=0)

        prediction = model.predict(input_data)[0]

        decoded_labels = label_encoder.inverse_transform(range(len(labels)))

        segment_predictions = []
        for label, conf in zip(decoded_labels, prediction):
            if conf >= 0.5:
                segment_predictions.append((i + 1, label, conf * 100, loudness, transcription))

        if not segment_predictions:
            segment_predictions.append((i + 1, "No se identificaron etiquetas", 0, loudness, transcription))

        data.extend(segment_predictions)

    return data

def crear_grafico_loudness(loudness_values):
    # Crear un gráfico de loudness a lo largo del tiempo
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(loudness_values)), loudness_values)
    plt.xlabel("Tiempo (segmentos)")
    plt.ylabel("Loudness")
    st.pyplot(plt)

def modificar_etiqueta(dataframe):
    analisis = []
    etiquetas_modificadas = 0  # Contador para etiquetas modificadas
    for index, row in dataframe.iterrows():
        texto = row["Texto"]
        etiqueta = row["Etiqueta"]
        if texto and etiqueta != "people_talking":
            analisis.append("people_talking")
            etiquetas_modificadas += 1
        else:
            analisis.append(etiqueta)
    
    dataframe["Análisis"] = analisis
    
    # Imprime el número de etiquetas modificadas
    st.write(f"Se modificaron {etiquetas_modificadas} etiquetas a 'people_talking'.")
    
    return dataframe

if uploaded_file:
    st.write("Procesando el archivo de audio largo...")
    st.audio(uploaded_file, format="audio/wav")
    predictions_per_second = hacer_prediccion(uploaded_file)

    st.write("Predicciones por segundo:")
    predictions_df = pd.DataFrame(predictions_per_second, columns=["Segundo", "Etiqueta", "Confianza", "Loudness", "Texto"])
    
    # Llama a la función para modificar las etiquetas y crea la columna "analisis"
    predictions_df = modificar_etiqueta(predictions_df)
    
    st.table(predictions_df[["Segundo", "Etiqueta", "Confianza", "Loudness", "Texto", "Análisis"]])
    
    # Obtiene el número de etiquetas modificadas
    etiquetas_modificadas = predictions_df[predictions_df["Etiqueta"] != predictions_df["Análisis"]].shape[0]
    
    # Imprime el número de etiquetas modificadas en Streamlit
    filas = predictions_df.shape[0]
    porcentaje = int(etiquetas_modificadas/filas*100)
    st.write(f"Número de etiquetas modificadas a 'people_talking': {etiquetas_modificadas} / {filas} que representa el {porcentaje} %")
    
    crear_grafico_loudness(loudness_values)
    embeddings_db = predictions_df[["Segundo","Loudness", "Texto"]]
    embeddings_db["Embeddings"] = None
    embeddings_db["Distancia"] = None

    st.table(embeddings_db)