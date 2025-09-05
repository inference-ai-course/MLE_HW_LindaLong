
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from asr import transcribe_audio
from llm import generate_response
from tts import synthesize_speech




# ========== FASTAPI ==========

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Welcome to ASR/LLM/TTS World"}


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