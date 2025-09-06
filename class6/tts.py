import pyttsx3

def synthesize_speech(text, filename="response.wav"):

    tts_engine = pyttsx3.init()
    tts_engine.save_to_file(text, filename)
    tts_engine.runAndWait()
    return filename
