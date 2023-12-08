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
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
import os

# Cargar el modelo de embeddings
embeddings_model = SentenceTransformer('distiluse-base-multilingual-cased')

model = tf.keras.models.load_model('17.h5')

labels = ["llanto", "vidrio_rompiéndose", "grito", "bala", "persona_hablando"]

loudness_values = []
predictions_list = []
labels_list = []

label_encoder = LabelEncoder()
label_encoder.fit(labels)

st.title('Detector de Audio')

uploaded_file = st.file_uploader("Subir un archivo de audio largo", type=["wav"])

def convertir_audio(segment):
    buffer = BytesIO()
    sf.write(buffer, segment, 16000, subtype='PCM_16', format='wav')
    buffer.seek(0)
    return AudioSegment.from_wav(buffer)

def calcular_loudness(segment):
    audio = convertir_audio(segment)
    loudness = audio.rms
    return loudness

def transcribir_audio(segment):
    # Obtener la duración total del audio en milisegundos
    duracion_total = len(audio)

    # Inicializar el reconocedor de voz
    reconocedor = sr.Recognizer()

    # Lista para almacenar las transcripciones
    transcripciones = []

    # Dividir y transcribir en segmentos de 2 segundos
    for inicio in range(0, duracion_total, 2000):
        # Establecer el rango de tiempo para el segmento actual
        fin = min(inicio + 2000, duracion_total)
        segmento = audio[inicio:fin]

        # Guardar el segmento como un nuevo archivo temporal
        segmento.export("temp.wav", format="wav")

        # Cargar el archivo temporal y realizar reconocimiento de voz
        with sr.AudioFile("temp.wav") as fuente:
            audio_temp = reconocedor.record(fuente)
            try:
                # Obtener la transcripción del segmento y agregar a la lista
                transcripcion = reconocedor.recognize_google(audio_temp, language="es")
                transcripciones.append(transcripcion)
            except sr.UnknownValueError:
                # Agregar un valor nulo si no se pudo reconocer el audio
                transcripciones.append(None)
            except sr.RequestError as e:
                print(f"Error en la solicitud del servicio de reconocimiento de voz; {e}")

    # Eliminar el archivo temporal
    os.remove("temp.wav")

    return transcripciones

def query_collection(input_text):
    if not input_text:
        return 0, ""

    client_persistent = chromadb.PersistentClient(path='data_embeddings')
    db = client_persistent.get_collection(name='violence_embeddings_DB')
    results = db.query(query_texts=[input_text], n_results=1)

    if results['distances'] and results['documents']:
        documents = results['documents'][0][0]
        distance = results['distances'][0][0]
    else:
        documents = ""
        distance = 0

    return documents, distance

def hacer_prediccion(audio_data,t):
    audio, _ = librosa.load(audio_data, sr=16000)
    segment_duration = 2
    samples_per_segment = int(16000 * segment_duration)
    num_segments = len(audio) // samples_per_segment

    data = []

    transcription = transcribir_audio(t)

    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = (i + 1) * samples_per_segment
        segment = audio[start_sample:end_sample]

        loudness = calcular_loudness(segment)
        loudness_values.append(loudness)
        
        embeddings, distance = query_collection(transcription[i])

        spectrogram = librosa.feature.melspectrogram(y=segment, sr=16000)
        input_data = np.expand_dims(spectrogram, axis=0)
        prediction = model.predict(input_data)[0]

        decoded_labels = label_encoder.inverse_transform(range(len(labels)))

        segment_predictions = []

        for label, conf in zip(decoded_labels, prediction):
            predictions_list.append(prediction)
            labels_list.append(decoded_labels)
            #print(label, conf)
            if conf >= 0.4:
                segment_predictions.append((label, conf, loudness, transcription[i], segment, embeddings, distance))

        if not segment_predictions:
            segment_predictions.append(("No concluyente", 0, loudness, transcription[i], segment, embeddings, distance))

        data.extend(segment_predictions)
    return data

def crear_grafico_loudness(loudness_values):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(loudness_values)), loudness_values)
    plt.xlabel("Tiempo (segmentos)")
    plt.ylabel("Loudness")
    st.pyplot(plt)

if uploaded_file:
    audio_for_transcription = audio = AudioSegment.from_wav(uploaded_file)
    st.write("Procesando el archivo de audio largo...")

    st.audio(uploaded_file, format="audio/wav")
    predictions_per_second = hacer_prediccion(uploaded_file, audio_for_transcription)

    st.write("Predicciones por segundo:")
    predictions_df = pd.DataFrame(predictions_per_second, columns=["Etiqueta", "Confianza", "Loudness", "Texto", "Fragmento", "Embeddings", "Distancia"])

    for index, row in predictions_df.iterrows():
        st.audio(row["Fragmento"], sample_rate=16000, format="audio/wav")

        table_data = {
            "Etiqueta": row["Etiqueta"],
            "Confianza": row["Confianza"],
            "Loudness": row["Loudness"],
            "Texto": row["Texto"],
            "Embeddings": row["Embeddings"] if row["Distancia"] and float(row["Distancia"]) <= 0.4 else "",
            "Distancia": float(row["Distancia"]) if row["Distancia"] and float(row["Distancia"]) <= 0.4 else ""


        }
        st.table(table_data)

    crear_grafico_loudness(loudness_values)

    filtered_embeddings_df = predictions_df[["Texto", "Embeddings", "Distancia"]].copy()
    st.table(filtered_embeddings_df)


