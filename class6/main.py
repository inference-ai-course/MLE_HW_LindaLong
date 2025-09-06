
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import agent
from tts import synthesize_speech
from asr import transcribe_audio



# ========== FASTAPI ==========

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Welcome to Audio World"}


# ========== voice query on arxiv search or math calculation ==========

@app.post("/api/voice-query/")

async def voice_query_endpoint (file: UploadFile = File(...)): #(request: QueryRequest):
    
    audio_bytes = await file.read()
    print(f"Received file: {file.filename}")

    # ASR
    user_text = transcribe_audio(audio_bytes)
    print("Transcribed text:", user_text)

    # Call Llama 3 model (instructed to output function calls when needed)
    llm_response = agent.llama3_chat_model(user_text)
    print("LLM response:", llm_response)

    # Process LLM output and possibly call tools
    reply_text = agent.route_llm_output(llm_response)
    print("Final response:", reply_text)

    # TTS:Convert reply_text to speech (TTS) and return it
    audio_path = synthesize_speech(reply_text)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)