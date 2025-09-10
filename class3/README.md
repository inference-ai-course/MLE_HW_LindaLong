
# Class 3: Voice Assistant with FastAPI, ASR, TTS, and LLM Integration

This folder contains scripts and resources for building a simple voice assistant using FastAPI, automatic speech recognition (ASR), text-to-speech (TTS), and large language models (LLMs) such as OpenAI or Hugging Face.

## Contents

- **main.py**  
  FastAPI app for handling audio file uploads, running ASR, generating responses with an LLM, and synthesizing voice replies.
- **asr.py**  
  Utilities for transcribing audio to text (ASR).
- **tts.py**  
  Utilities for converting text responses to speech (TTS).
- **llm.py**  
  Logic for generating responses using OpenAI or Hugging Face LLMs, with conversation history support.
- **test_audio.html**  
  Simple HTML client for recording and uploading audio to the FastAPI server.


## Features

- **Audio Upload:** Accepts audio files via HTTP POST.
- **ASR:** Converts uploaded audio to text.
- **LLM Response:** Generates context-aware responses using OpenAI or Hugging Face models.
- **TTS:** Converts text responses back to audio for playback.
- **Web Client:** Includes a simple HTML/JS client for recording and sending audio.
- **Conversation History:** Maintains recent conversation turns for context.
