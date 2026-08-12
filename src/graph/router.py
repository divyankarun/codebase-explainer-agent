"""Routing logic to classify queries into architecture vs specific Q&A."""

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL


def classify_question(question: str, conversation_history: list = None) -> str:
    """Classify a question about a codebase as either 'architecture' or 'specific'.
    
    Args:
        question: User query string.
        conversation_history: Optional list of prior conversation messages.
        
    Returns:
        String 'architecture' or 'specific'.
    """
    if not GROQ_API_KEY:
        print("[WARNING] GROQ_API_KEY not set. Defaulting routing classification to 'specific'.")
        return "specific"

    # Rule-Based Vague Follow-up Detection
    vague_patterns = [
        "explain more", "tell me more", "what you just showed", "go deeper",
        "elaborate", "show me more", "can you explain", "what about that", "how does that work"
    ]
    q_lower = question.lower().strip()
    is_vague = any(pat in q_lower for pat in vague_patterns) or (
        len(q_lower.split()) <= 6 and any(p in q_lower for p in ["that", "it", "this"])
    )

    if is_vague and conversation_history and len(conversation_history) > 1:
        print(f"[ROUTER] Vague follow-up query detected with active conversation history. Routing to 'specific'.")
        return "specific"

    # Build conversation context string if available
    context_str = ""
    if conversation_history and len(conversation_history) > 1:
        prior_turns = []
        for msg in conversation_history[-3:-1]:
            content = getattr(msg, "content", "") if hasattr(msg, "content") else str(msg)
            role = "User" if (getattr(msg, "type", "") == "human" or msg.__class__.__name__ == "HumanMessage") else "Assistant"
            if content:
                # Truncate prior turn content to keep prompt compact
                prior_turns.append(f"{role}: {content[:150]}")
        if prior_turns:
            context_str = "=== PRIOR CONVERSATION CONTEXT ===\n" + "\n".join(prior_turns) + "\n\n"

    prompt = f"""{context_str}Classify the following user question about a software codebase into exactly one of two categories:

'architecture': Questions about overall project purpose, top-level directory layout, repository broad structure, high-level overview, or general tech stack list.
'specific': Questions about a specific feature, component, mechanism, data/retrieval pipeline (e.g. document vector retrieval pipeline, vector store indexing, session pooling, authentication), file, function, class, implementation detail, method, or follow-up on a previous specific code discussion.

Question: "{question}"

Respond with ONLY ONE word: either 'architecture' or 'specific'."""

    try:
        client = Groq(api_key=GROQ_API_KEY)
        messages = [
            {
                "role": "system",
                "content": "You are a classifier. Respond ONLY with one word: 'architecture' or 'specific'."
            },
            {"role": "user", "content": prompt}
        ]
        
        response = None
        for m in [GROQ_MODEL, "llama-3.1-8b-instant"]:
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=m,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=10,
                    )
                    break
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        import time
                        print(f"[ROUTER RATE LIMIT] Hit 429 on model '{m}' (attempt {attempt+1}/3). Waiting 4s...")
                        time.sleep(4)
                    else:
                        raise e
            if response:
                break

        raw_output = (response.choices[0].message.content or "").strip().lower() if response else "specific"
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


