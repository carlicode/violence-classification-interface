import streamlit as st
import librosa
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import tempfile
import os

# Cargar el modelo y el label encoder
model = load_model('17.h5')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["crying", "glass_breaking", "screams", "gun_shot", "people_talking"])

# Función para preprocesar y clasificar fragmentos de audio
def classify_audio(audio_fragment):
    # Normalizar el espectrograma
    spectrogram = librosa.feature.melspectrogram(y=audio_fragment, sr=16000)
    normalized_spectrogram = librosa.util.normalize(spectrogram)

    # Reshape para que coincida con el formato de entrada del modelo
    input_data = normalized_spectrogram.reshape(1, 128, 63, 1)

    # Realizar la predicción
    predictions = model.predict(input_data)[0]

    return predictions

# Función para reproducir un fragmento de audio
def play_audio(audio_fragment):
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio_path = temp_audio.name + ".wav"
        sf.write(temp_audio_path, audio_fragment, 16000)
        audio_file = open(temp_audio_path, "rb")
        st.audio(audio_file, format="audio/wav")

# Configuración de la aplicación Streamlit
st.title("Clasificación de Audio")
uploaded_file = st.file_uploader("Cargar archivo de audio", type=["wav"])

if uploaded_file is not None:
    audio, _ = librosa.load(uploaded_file, sr=16000)
    st.audio(uploaded_file, format='audio/wav')

    st.header("Fragmentos de Audio y Predicciones")
    fragment_duration = 2  # segundos
    num_fragments = len(audio) // (16000 * fragment_duration)

    for i in range(num_fragments):
        start_sample = i * 16000 * fragment_duration
        end_sample = start_sample + 16000 * fragment_duration
        audio_fragment = audio[start_sample:end_sample]

        st.subheader(f"Fragmento {i+1}")
        play_audio(audio_fragment)

        # Clasificar el fragmento de audio
        predictions = classify_audio(audio_fragment)

        # Mostrar las predicciones en una tabla
        st.write("Predicciones:")
        st.write("| Etiqueta | % de Predicción |")
        st.write("| --- | --- |")

        for label, percentage in zip(label_encoder.classes_, predictions):
            st.write(f"| {label} | {percentage*100:.2f}% |")

# Limpiar archivos temporales
temp_files = [f for f in os.listdir() if f.endswith(".wav")]
for temp_file in temp_files:
    os.remove(temp_file)
