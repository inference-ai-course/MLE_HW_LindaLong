
import whisper

# ========== ASR ==========


asr_model = whisper.load_model("small")

def transcribe_audio(audio_bytes):
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe("temp.wav", language="en")  # Force English transcription
    #os.remove("temp.wav")  # Remove temp.wav after use
    return result["text"]

