# Class 6: Advanced Function Calling, LLM Integration, and Tools

This folder contains Python scripts for advanced language model (LLM) integration, function calling, and tool augmentation, as part of the AI Learning project.

## Contents

- **main.py**  
  FastAPI app for serving endpoints that combine ASR (automatic speech recognition), TTS (text-to-speech), and LLM-based reasoning or function calling.

- **llm.py**  
  Utilities for interacting with large language models (OpenAI, Hugging Face, etc.), including conversation management, prompt construction, and response generation.

- **tools.py**  
  Implements various callable tools and functions (e.g., math calculation, arXiv search) that can be triggered by the LLM or API endpoints.

- **asr.py**  
  Functions for transcribing audio to text using ASR models or APIs.

- **tts.py**  
  Functions for synthesizing speech from text using TTS models or APIs.

- **agent.py**  
  Logic for orchestrating multi-step reasoning, tool selection, and function calling based on user input and LLM output.

- **sqlite_crud.py**  
  Utilities for performing CRUD operations on a SQLite database, useful for storing metadata, logs, or user interactions.

- **Other utility scripts**  
  Additional scripts for experimentation, testing, or extending the core functionality.

## Features

- **Function Calling:**  
  - LLM can decide when to call external tools (e.g., calculator, search) and integrate results into responses.
- **ASR & TTS Integration:**  
  - Seamless pipeline for audio input (speech-to-text) and output (text-to-speech).
- **Math & Search Tools:**  
  - Built-in support for mathematical expression evaluation and academic search (e.g., arXiv).
- **Conversation Management:**  
  - Maintains context and conversation history for multi-turn interactions.
- **Database Support:**  
  - Store and retrieve structured data using SQLite.

## Usage

1. **Install requirements:**
   ```sh
   pip install fastapi uvicorn openai transformers soundfile pydub sympy faiss-cpu sentence-transformers sqlite3
