import speech_recognition as sr

def transcribir_audio(audio_path):
    r = sr.Recognizer()
    transcripciones = []  # Lista para almacenar las transcripciones

    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)

    audio_duration = len(audio.frame_data) / audio.sample_rate
    segment_duration = 5  # Duración de cada segmento en segundos

    print("Duración total del audio:", int(audio_duration), "segundos")
    lista_transcripciones = [0] * int(audio_duration)
    posiciones = []

    for i in range(4, int(audio_duration) + 1, 5):
        posiciones.append(i)
    posiciones.append(int(audio_duration)-1)

    conjunto_sin_repetir = set(posiciones)
    posiciones_res = list(conjunto_sin_repetir)
    posiciones_res.sort()

    for segment_start_time in range(0, int(audio_duration), segment_duration):
        start_sample = int(segment_start_time * audio.sample_rate)
        end_sample = int((segment_start_time + segment_duration) * audio.sample_rate)
        segment_audio_data = audio.frame_data[start_sample:end_sample]

        segment_audio = sr.AudioData(segment_audio_data, audio.sample_rate, audio.sample_width)

        try:
            segment_transcript = r.recognize_google(segment_audio, language="es-ES", show_all=True)
            if 'alternative' in segment_transcript:
                best_alternative = max(segment_transcript['alternative'], key=lambda x: x.get('confidence', 0))
                segment_transcript = best_alternative['transcript']
            else:
                segment_transcript = ""
        except sr.RequestError:
            segment_transcript = "(Error en la transcripción)"

        transcripciones.append(segment_transcript)
    
    for i in range(len(posiciones)):
            lista_transcripciones[posiciones[i]] = transcripciones[i]
    
    print(transcripciones)
    return lista_transcripciones

