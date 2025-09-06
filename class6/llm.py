from transformers import pipeline
import os
from openai import OpenAI

INIT_SYS_PROMPT = {"role": "system", "content": "You are a helpful and friendly voice assistant. "}
CONVERSATION_TURN_LIMIT = 5  # Limit to last 10 exchanges (user + assistant)
global conversation_history
conversation_history = []

#===================llm settings =====================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("YOUR_") or OPENAI_API_KEY.strip() == "":
    raise RuntimeError(f"OPENAI_API_KEY missing or not set.")
client = OpenAI(api_key=OPENAI_API_KEY)

# #llm = pipeline("text-generation", model="meta-llama/Llama-3-8B")
llm = pipeline("text-generation", model="gpt2")


def reset_conversation_in_need():

    if len(conversation_history)>=CONVERSATION_TURN_LIMIT *2+ 1:
        conversation_history.clear()
        conversation_history.append(INIT_SYS_PROMPT)
        print("[INFO] Conversation history exceeded limit. Resetting...")
    
    elif len(conversation_history)==0:
        conversation_history.append(INIT_SYS_PROMPT)
        print("[INFO] Starting Conversation...")


def generate_response_without_history(user_text):

    outputs = llm(user_text, max_new_tokens=100)
    bot_response = outputs[0]["generated_text"]
       # Extract only the last assistant reply
    if "assistant:" in bot_response:
       bot_response = bot_response.split("assistant:")[-1].strip()

    return bot_response


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


def extract_math_expression(user_text):
    """
    Extract math expression from user text using LLM, return 'none' if not found.
    """

    prompt = (
        "Extract the math expression from the following sentence using mathematical symbols only, "
        "so that the result can be directly parsed by sympy.sympify in Python. "
        "If there is no math expression, reply with 'none'.\n"
        f"Sentence: {user_text}\n"
        "Math expression:"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=30,
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()
    print(f"[DEBUG] Extracted math expression: {answer}")
    return answer


# if __name__ == "__main__":

#     # # Example usage:
#     print(generate_response("what is 3 times5?")) # Output: 15
#     print(generate_response("What is 2 plus 3 times 4?")) # Output: 2 + 3 * 4
#     print(generate_response("Thanks!")) # Output: none