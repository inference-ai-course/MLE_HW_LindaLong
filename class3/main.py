from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from fastapi import FastAPI
import whisper

# ========== TTS ==========

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


# ========== LLM ==========
from transformers import pipeline

#llm = pipeline("text-generation", model="meta-llama/Llama-3-8B")
llm = pipeline("text-generation", model="gpt2")
conversation_history = []

def generate_response(user_text):
    conversation_history.append({"role": "user", "text": user_text})
    # Construct prompt from history
    #prompt = f"user: {user_text}\nassistant:"
    prompt = ""
    
    for turn in conversation_history[-5:]:
        prompt += f"{turn['role']}: {turn['text']}\n"

    outputs = llm(prompt, max_new_tokens=100)
    bot_response = outputs[0]["generated_text"]
       # Extract only the last assistant reply
    if "assistant:" in bot_response:
        bot_response = bot_response.split("assistant:")[-1].strip()
 
    conversation_history.append({"role": "assistant", "text": bot_response})
    return bot_response


# ========== ASR ==========

import os
asr_model = whisper.load_model("small")

def transcribe_audio(audio_bytes):
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr_model.transcribe("temp.wav", language="en")  # Force English transcription
    #os.remove("temp.wav")  # Remove temp.wav after use
    return result["text"]


# ========== FASTAPI ==========

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Welcome to ASR World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):

    audio_bytes = await file.read()
    print(f"Received file: {file.filename}")
    # ASR
    results = transcribe_audio(audio_bytes)
    print("request:", results)

    #LLM
    bot_text = generate_response(results)
    print("response:", bot_text)
    #TTS
    audio_path = synthesize_speech(bot_text)

    return FileResponse(audio_path, media_type="audio/wav")

# ========== ublock cross-origin request for test purpose only ==========
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== COMMAND TO RUN SERVER/CLIENT ==========

#run server: uvicorn main:app --reload
#run client: curl -X POST "http://localhost:8000/chat/" -F "file=@testAudio.wav"
