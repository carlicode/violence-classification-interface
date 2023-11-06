import librosa

def calculate_loudness_per_second(audio_file):
    try:
        audio, _ = librosa.load(audio_file, sr=16000)
        duration_in_seconds = len(audio) / 16000  # Duración en segundos
        loudness_values = []

        for second in range(int(duration_in_seconds)):
            start_sample = second * 16000  # Muestras por segundo (16000 muestras por segundo)
            end_sample = (second + 1) * 16000  # Muestras por segundo
            segment = audio[start_sample:end_sample]

            # Calcular el loudness utilizando el valor RMS (Root Mean Square) de la señal
            loudness = librosa.feature.rms(segment)
            loudness_values.append(loudness[0][0])  # Tomar el valor RMS como loudness

        return loudness_values
    except Exception as e:
        return None
