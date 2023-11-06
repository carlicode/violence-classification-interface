import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import io
from pydub import AudioSegment

model = tf.keras.models.load_model('experiment.h5')
labels = ["crying", "glass_breaking", "screams", "gun_shot", "people_talking"]

label_encoder = LabelEncoder()
label_encoder.fit(labels)

st.title('Audio Detector')

uploaded_file = st.file_uploader("Upload a long audio file", type=["wav"])

def calculate_loudness_per_second(audio_file):
    try:
        audio = AudioSegment.from_file(audio_file)
        duration_in_seconds = len(audio) / 1000
        loudness_values = []

        for second in range(int(duration_in_seconds)):
            start_time = second * 1000
            end_time = (second + 1) * 1000
            segment = audio[start_time:end_time]
            loudness = segment.dBFS
            loudness_values.append(loudness)

        return loudness_values
    except Exception as e:
        return None

def make_prediction(audio_data):
    audio, _ = librosa.load(audio_data, sr=16000)
    loudness_values = calculate_loudness_per_second(audio_data)

    segment_duration = 1
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

    return predictions, loudness_values

if uploaded_file:
    st.write("Processing the long audio file...")
    st.audio(uploaded_file, format="audio/wav")
    predictions_per_second, loudness_values = make_prediction(uploaded_file)

    st.write("Results per second:")
    st.write("Second | Predicted Label | Loudness (dB) | Confidence")
    for i, (segment_predictions, loudness) in enumerate(zip(predictions_per_second, loudness_values)):
        st.write(f"{i + 1} | {', '.join([label for label, _ in segment_predictions])} | {loudness:.2f} | {', '.join([f'{conf:.2f}%' for _, conf in segment_predictions])}")
