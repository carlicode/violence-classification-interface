import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import pandas as pd
import io

model = tf.keras.models.load_model('experiment.h5')
labels = ["crying", "glass_breaking", "screams", "gun_shot", "people_talking"]

label_encoder = LabelEncoder()
label_encoder.fit(labels)  

st.title('Audio Detector')

uploaded_file = st.file_uploader("Upload a long audio file", type=["wav"])

def make_prediction(audio_data):
    audio, _ = librosa.load(audio_data, sr=16000)

    segment_duration = 1  # Duration
    samples_per_segment = int(16000 * segment_duration)
    num_segments = len(audio) // samples_per_segment

    predictions = []

    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = (i + 1) * samples_per_segment
        segment = audio[start_sample:end_sample]

        spectrogram = librosa.feature.melspectrogram(y=segment, sr=16000)
        input_data = np.expand_dims(spectrogram, axis=0)

        prediction = model.predict(input_data)[0]

        decoded_labels = label_encoder.inverse_transform(range(len(labels))
        
        segment_predictions = []
        for label, conf in zip(decoded_labels, prediction):
            if conf >= 0.75:
                segment_predictions.append((label, conf * 100))
        
        if not segment_predictions:
            segment_predictions.append(("No labels identified", 0))
        
        predictions.append(segment_predictions)

    return predictions

if uploaded_file:
    st.write("Processing the long audio file...")
    st.audio(uploaded_file, format="audio/wav")
    predictions_per_second = make_prediction(uploaded_file)

    st.write("Predictions per second:")
    data = []
    for i, segment_predictions in enumerate(predictions_per_second):
        for label, confidence in segment_predictions:
            data.append((i + 1, label, f'{confidence:.2f}%'))

    df = pd.DataFrame(data, columns=["Second", "Label", "Confidence"])
    st.table(df)
