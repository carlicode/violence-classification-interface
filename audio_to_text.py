import speech_recognition as sr

def reconocer_audio(ruta_audio):
    recognizer = sr.Recognizer()

    with sr.AudioFile(ruta_audio) as source:
        # Ajusta el umbral de energía según sea necesario
        recognizer.adjust_for_ambient_noise(source)
        
        print("Reconociendo audio...")

        try:
            audio_data = recognizer.record(source)
            texto_reconocido = recognizer.recognize_google(audio_data, language="es-ES")
            print("Texto reconocido:", texto_reconocido)
        except sr.UnknownValueError:
            print("No se pudo reconocer el audio")
        except sr.RequestError as e:
            print(f"Error en la solicitud al servicio de reconocimiento de voz: {e}")

# Ruta del archivo de audio que quieres reconocer
ruta_del_audio = "C:/Users/Carlita/Desktop/tesis/Dataset collection/castigador_test.wav"

# Llama a la función para reconocer el audio
reconocer_audio(ruta_del_audio)
