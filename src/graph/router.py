"""Routing logic to classify queries into architecture vs specific Q&A."""

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL


def classify_question(question: str) -> str:
    """Classify a question about a codebase as either 'architecture' or 'specific'.
    
    Args:
        question: User query string.
        
    Returns:
        String 'architecture' or 'specific'.
    """
    if not GROQ_API_KEY:
        print("[WARNING] GROQ_API_KEY not set. Defaulting routing classification to 'specific'.")
        return "specific"

    prompt = f"""Classify the following user question about a software codebase into exactly one of two categories:

'architecture': Questions about overall project structure, high-level directory layout, general module overview, repository purpose, tech stack, or high-level architecture principles.
'specific': Questions about a specific feature component (e.g. authentication, sessions, request dispatching, proxies, error handling), file, function, class, implementation detail, method, or code snippet.

Question: "{question}"

Respond with ONLY ONE word: either 'architecture' or 'specific'."""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a classifier. Respond ONLY with one word: 'architecture' or 'specific'."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw_output = (response.choices[0].message.content or "").strip().lower()
        print(f"[ROUTER] Classification LLM response: '{raw_output}'")

        if "architecture" in raw_output and "specific" not in raw_output:
            return "architecture"
        elif "specific" in raw_output:
            return "specific"
        else:
            print(f"[ROUTER] Ambiguous response '{raw_output}'. Defaulting to 'specific'.")
            return "specific"
    except Exception as e:
        print(f"[ROUTER ERROR] Failed during question classification: {e}. Defaulting to 'specific'.")
        return "specific"
