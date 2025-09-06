from tools import simulate_search_arxiv, calculate
from llm import extract_math_expression,generate_response
import json

def is_arxiv_query(user_text):
    """Check if the user text is likely an arXiv search query.
    """
    keywords = ['search', 'arxiv', 'research']
    return any(word in user_text.lower() for word in keywords)


def llama3_chat_model(user_text):
    """
    Call Llama 3 chat model to get response with function call when needed.
    """

    prompt = ""

    if (is_arxiv_query(user_text)):

        prompt = f"""
            {{ "function": "search_arxiv", "arguments": {{ "query": "{user_text}" }} }}
            """
    elif (math_expression := extract_math_expression(user_text).lower()) != "none":
        prompt = f"""
            {{ "function": "calculate", "arguments": {{ "expression": "{math_expression}" }} }}
            """
    else:# make a multi_turn llm call.
        prompt = f"""
            {{ "function": "others", "arguments": {{ "query": "{user_text}" }} }}
            """
    return prompt



def route_llm_output(llm_output, filename="response.wav"):

    """
    Route LLM response to the correct tool if it's a function call, else return the text.
    Expects LLM output in JSON format like {'function': ..., 'arguments': {...}}.
    """
    try:
        output = json.loads(llm_output)
        func_name = output.get("function")
        args = output.get("arguments", {})
    except (json.JSONDecodeError, TypeError):
        return "I'm sorry, I might have difficulties in answering that. Could you please rephrase?"

    if func_name == "search_arxiv":
        query = args.get("query", "")
        return simulate_search_arxiv(query)
    elif func_name == "calculate":
        expr = args.get("expression", "")
        return calculate(expr)
    elif func_name == "others":
        query = args.get("query", "")
        return generate_response (query)
    else:
        # Fallback reply on unknown function
        return "I'm sorry, I might not be able to help with that. Could you please rephrase?"