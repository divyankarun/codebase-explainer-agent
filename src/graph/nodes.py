"""Execution nodes for retrieval, summary response, and citation generation."""

from typing import Any
import json
from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL, TOP_K_RETRIEVAL
from src.embedding.embedder import embed_query
from src.embedding.faiss_store import search


def _format_summary_text(summary: Any) -> str:
    """Format architecture summary object or dict into clean Markdown context."""
    if hasattr(summary, "model_dump"):
        data = summary.model_dump()
    elif isinstance(summary, dict):
        data = summary
    else:
        return str(summary)

    main_modules = "\n".join(f"- {m}" for m in data.get("main_modules", [])) or "None listed."
    entry_points = "\n".join(f"- {e}" for e in data.get("entry_points", [])) or "None listed."
    tech_stack = "\n".join(f"- {t}" for t in data.get("tech_stack", [])) or "None listed."
    how_it_connects = data.get("how_it_connects", "No connection details available.")
    notable_patterns = "\n".join(f"- {p}" for p in data.get("notable_patterns", [])) or "None listed."

    return f"""### Main Modules:
{main_modules}

### Entry Points:
{entry_points}

### Tech Stack:
{tech_stack}

### How Component Data & Control Flows:
{how_it_connects}

### Architectural Patterns:
{notable_patterns}"""


def _build_groq_messages(system_prompt: str, question: str, messages: list = None, citation_reminder: str = None) -> list[dict]:
    """Construct Groq messages payload including system context, conversation history, and latest query."""
    groq_msgs = [{"role": "system", "content": system_prompt}]

    if messages and len(messages) > 1:
        # Include prior turns (excluding current last message which is the current question)
        for msg in messages[:-1]:
            role = "assistant"
            content = ""
            if hasattr(msg, "content"):
                content = msg.content
                msg_type = getattr(msg, "type", "")
                if msg_type == "human" or msg.__class__.__name__ == "HumanMessage":
                    role = "user"
            elif isinstance(msg, dict):
                content = msg.get("content", "")
                if msg.get("role") in ["user", "human"]:
                    role = "user"

            if content:
                groq_msgs.append({"role": role, "content": content})

    user_content = question
    if citation_reminder:
        user_content = f"{question}\n\n{citation_reminder}"

    groq_msgs.append({"role": "user", "content": user_content})
    return groq_msgs


def answer_from_summary(question: str, summary: Any, messages: list = None) -> str:
    """Builds a prompt with architecture summary as context, calls Groq, and returns the answer.
    
    Args:
        question: User query string.
        summary: ArchitectureSummary object or dict.
        messages: Optional prior conversation history list.
        
    Returns:
        Generated answer text string.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is missing or empty.")

    summary_ctx = _format_summary_text(summary)
    system_prompt = f"""You are an expert codebase architecture assistant.
Use the following high-level Architecture Summary of the codebase to answer the user's question accurately, clearly, and comprehensively.

=== ARCHITECTURE SUMMARY ===
{summary_ctx}
"""

    groq_msgs = _build_groq_messages(system_prompt, question, messages)

    client = Groq(api_key=GROQ_API_KEY)
    response = None
    for m in [GROQ_MODEL, "llama-3.1-8b-instant"]:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=m,
                    messages=groq_msgs,
                    temperature=0.2,
                )
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    import time
                    print(f"[SUMMARY RATE LIMIT] Hit 429 on model '{m}' (attempt {attempt+1}/3). Waiting 4s...")
                    time.sleep(4)
                else:
                    raise e
        if response:
            break

    return response.choices[0].message.content or "" if response else ""


def answer_from_retrieval(
    question: str,
    index: Any,
    chunk_metadata: list[dict],
    summary: Any,
    messages: list = None
) -> str:
    """Embeds query, retrieves top-k code chunks from FAISS index, and generates answer with citations.
    
    Args:
        question: User query string.
        index: FAISS index instance.
        chunk_metadata: List of chunk metadata dicts corresponding to index vectors.
        summary: ArchitectureSummary object or dict as fallback context.
        messages: Optional prior conversation history list.
        
    Returns:
        Generated answer text string with (file_path:start_line-end_line) citations.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is missing or empty.")

    # 1. Embed query & search (contextualize query string with prior turn if follow-up)
    search_query = question
    if messages and len(messages) > 1:
        prior_user_msgs = []
        for m in messages[:-1]:
            if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage":
                content = getattr(m, "content", "")
                if content:
                    prior_user_msgs.append(content)
            elif isinstance(m, dict) and m.get("role") in ["user", "human"]:
                content = m.get("content", "")
                if content:
                    prior_user_msgs.append(content)
        if prior_user_msgs:
            search_query = f"{prior_user_msgs[-1]} {question}"

    # Detect if query is code/implementation seeking
    code_seeking_keywords = ["function", "method", "implementation", "code", "class", "def ", "how is it implemented", "show me the code"]
    combined_query_lower = f"{question} {search_query}".lower()
    is_code_seeking = any(kw in combined_query_lower for kw in code_seeking_keywords)

    query_vec = embed_query(search_query)

    if is_code_seeking:
        # Retrieve larger candidate pool to ensure code chunks are captured
        pool_k = max(30, TOP_K_RETRIEVAL * 5)
        candidate_indices = search(index, query_vec, k=pool_k)
        
        # Filter for code chunks
        code_indices = [idx for idx in candidate_indices if 0 <= idx < len(chunk_metadata) and chunk_metadata[idx].get("file_type") == "code"]
        
        if code_indices:
            top_indices = code_indices[:TOP_K_RETRIEVAL]
            print(f"[RETRIEVAL] Code-seeking query detected. Filtered {len(candidate_indices)} candidates to {len(code_indices)} code chunks. Using top {len(top_indices)} code chunks.")
        else:
            top_indices = candidate_indices[:TOP_K_RETRIEVAL]
            print(f"[RETRIEVAL] Code-seeking query detected, but no code chunks found in top {pool_k}. Falling back to top {len(top_indices)} general candidates.")
    else:
        top_indices = search(index, query_vec, k=TOP_K_RETRIEVAL)
        print(f"[RETRIEVAL] General query. Using top {len(top_indices)} candidates.")

    # 2. Format retrieved chunks & print metadata summary
    formatted_chunks = []
    retrieved_meta_logs = []
    for idx in top_indices:
        if 0 <= idx < len(chunk_metadata):
            chunk = chunk_metadata[idx]
            file_path = chunk.get("file_path", "unknown")
            file_type = chunk.get("file_type", "unknown")
            start_line = chunk.get("start_line", 1)
            end_line = chunk.get("end_line", 1)
            text = chunk.get("text", "")
            formatted_chunks.append(f"[{file_path}:{start_line}-{end_line}]\n{text}")
            retrieved_meta_logs.append(f"  * {file_path}:{start_line}-{end_line} (type: {file_type})")

    print("[RETRIEVED CHUNKS]:\n" + "\n".join(retrieved_meta_logs))

    retrieved_code_ctx = "\n\n---\n\n".join(formatted_chunks) if formatted_chunks else "No relevant code chunks found."
    summary_ctx = _format_summary_text(summary)

    citation_instruction = """CRITICAL CITATION REQUIREMENT:
Every factual claim about the code, function, class, or implementation detail MUST end with an inline citation in this exact format: (file_path:start_line-end_line)
Example: "The AuthBase class defines the interface for authentication (src/requests/auth.py:34-52)."
Do this for every claim, even in follow-up answers later in the conversation. Do not skip citations just because earlier turns already established context."""

    system_prompt = f"""You are an expert code Q&A assistant specializing in line-accurate code explanation.
Answer the user's question using the provided code snippets and architecture context.

=== ARCHITECTURE SUMMARY (FALLBACK CONTEXT) ===
{summary_ctx}

=== RETRIEVED CODE SNIPPETS ===
{retrieved_code_ctx}

=== INSTRUCTIONS ===
1. Base your answer on the retrieved code snippets.
2. If the provided chunks do not contain the actual implementation code, say so directly and do not invent or guess code that isn't present in the context. Never fabricate example implementations.
3. Be precise, concise, and helpful.

{citation_instruction}
"""

    groq_msgs = _build_groq_messages(system_prompt, question, messages, citation_reminder=citation_instruction)

    client = Groq(api_key=GROQ_API_KEY)
    response = None
    for m in [GROQ_MODEL, "llama-3.1-8b-instant"]:
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=m,
                    messages=groq_msgs,
                    temperature=0.2,
                )
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    import time
                    print(f"[RETRIEVAL RATE LIMIT] Hit 429 on model '{m}' (attempt {attempt+1}/3). Waiting 4s...")
                    time.sleep(4)
                else:
                    raise e
        if response:
            break

    return response.choices[0].message.content or "" if response else ""



