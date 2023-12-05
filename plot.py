import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import pandas as pd

model = tf.keras.models.load_model('experimento_11_Plateau.h5')

labels = ["crying", "glass_breaking", "screams", "gun_shot", "people_talking"]

label_encoder = LabelEncoder()
label_encoder.fit(labels)

st.title('Detector de Audio')

uploaded_file = st.file_uploader("Subir un archivo de audio largo", type=["wav"])

def hacer_prediccion(audio_data):
    audio, _ = librosa.load(audio_data, sr=16000)

    segment_duration = 2
    samples_per_segment = int(16000 * segment_duration)
    num_segments = len(audio) // samples_per_segment

    results = {
        'Label': [],
        'Confidence': [],
        'Segment': []
    }

    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = (i + 1) * samples_per_segment
        segment = audio[start_sample:end_sample]

        spectrogram = librosa.feature.melspectrogram(y=segment, sr=16000)
        input_data = np.expand_dims(spectrogram, axis=0)

        prediction = model.predict(input_data)[0]
        decoded_labels = label_encoder.inverse_transform(range(len(labels)))

        for label, conf in zip(decoded_labels, prediction):
            results['Label'].append(label)
            results['Confidence'].append(conf)
        results['Segment'].append(segment)

    return results

if uploaded_file:
    st.write("Procesando el archivo de audio largo...")

    st.audio(uploaded_file, format="audio/wav")
    results = hacer_prediccion(uploaded_file)

    # Mostrar tabla con resultados de predicción
    st.write("Resultados de predicción cada 2 segundos:")

    # Agrupar de 5 en 5
    labels5 = [results['Label'][i:i + 5] for i in range(0, len(results['Label']), 5)]
    confianzas5 = [results['Confidence'][i:i + 5] for i in range(0, len(results['Confidence']), 5)]
    segmentos5 = results['Segment']

    for i in range(len(labels5)):
        st.audio(segmentos5[i], sample_rate=16000, format="audio/wav")
        df = pd.DataFrame({'Labels': labels5[i], 'Confidences': confianzas5[i]})
        st.dataframe(df)
