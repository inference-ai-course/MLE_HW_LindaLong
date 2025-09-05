import pyttsx3

'''
from cozyvoice import CozyVoice

tts_engine = CozyVoice()

def synthesize_speech(text, filename="response.wav"):
    tts_engine.generate(text, output_file=filename)
    return filename

'''
def synthesize_speech(text, filename="response.wav"):

    tts_engine = pyttsx3.init()
    #ensure it's always saved to fresh file
    #if os.path.exists(filename):
    #    os.remove(filename)
    tts_engine.save_to_file(text, filename)
    tts_engine.runAndWait()
    return filename