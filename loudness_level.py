from pydub import AudioSegment
import numpy as np

def calculate_loudness_per_second(audio_file):
    try:
        audio = AudioSegment.from_file(audio_file)
        duration_in_seconds = len(audio) / 1000
        loudness_values = []

        for second in range(int(duration_in_seconds)):
            start_time = second * 1000  # Milisegundos
            end_time = (second + 1) * 1000  # Milisegundos
            segment = audio[start_time:end_time]
            loudness = segment.dBFS
            loudness_values.append(loudness)

        return loudness_values
    except Exception as e:
        return None

audio_file = 'audio.wav'
loudness_values = calculate_loudness_per_second(audio_file)

if loudness_values is not None:
    for second, loudness in enumerate(loudness_values, start=1):
        print(f"Segundo {second}: {loudness} dB")
else:
    print("No se pudo calcular el loudness por segundos.")
