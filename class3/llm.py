from transformers import pipeline
import os
from openai import OpenAI

INIT_SYS_PROMPT = {"role": "system", "content": "You are a helpful and warm voice assistant. "}
CONVERSATION_TURN_LIMIT = 5  # Limit to last 10 exchanges (user + assistant)
global conversation_history
conversation_history = []

#===================set env file =====================

from dotenv import get_key, load_dotenv, find_dotenv
_env_path = find_dotenv(usecwd=True)
load_dotenv(_env_path, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("YOUR_") or OPENAI_API_KEY.strip() == "":
    raise RuntimeError(f"OPENAI_API_KEY missing or not set.")
# Set your OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY)


def reset_conversation_in_need():

    if len(conversation_history)>=CONVERSATION_TURN_LIMIT *2+ 1:
        conversation_history.clear()
        conversation_history.append(INIT_SYS_PROMPT)
        print("[INFO] Conversation history exceeded limit. Resetting...")
    
    elif len(conversation_history)==0:
        conversation_history.append(INIT_SYS_PROMPT)
        print("[INFO] Starting Conversation...")


def generate_response(user_text):

        reset_conversation_in_need()

        conversation_history.append({"role": "user", "content": user_text})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=conversation_history,
            max_tokens=100,
            temperature=0.7,
        )
        bot_response = response.choices[0].message.content.strip()
        conversation_history.append({"role": "assistant", "content": bot_response})

        return bot_response

