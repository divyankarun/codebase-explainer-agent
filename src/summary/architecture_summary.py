"""LLM pipeline to generate structured architecture overview from README and file tree."""

import os
import json
import re
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL, CACHE_DIR
from src.summary.schema import ArchitectureSummary


def gather_context(files: list[dict]) -> str:
    """
    Extracts README, file tree, and entry-point snippets from raw files list.
    """
    # 1. Extract README content (match on 'readme' in path, case-insensitive)
    readme_text = "No README file found."
    for f in files:
        path = f.get("path", "")
        filename = os.path.basename(path).lower()
        if "readme" in filename or "readme" in path.lower():
            readme_text = f.get("content", "")
            break

    # 2. Build plain list of file paths (directory tree)
    file_paths = [f.get("path", "") for f in files if f.get("path")]
    file_tree_str = "\n".join(file_paths)

    # 3. Extract entry-point snippets (~500 chars)
    target_entry_names = {"main.py", "app.py", "index.js", "index.ts", "__init__.py"}
    snippets = []

    for f in files:
        path = f.get("path", "")
        filename = os.path.basename(path).lower()
        if filename in target_entry_names:
            content = f.get("content", "")
            snippet = content[:500]
            snippets.append(f"--- File: {path} ---\n{snippet}")

    snippets_str = "\n\n".join(snippets) if snippets else "No common entry point snippets found."

    context = f"""=== README ===
{readme_text}

=== FILE TREE ===
{file_tree_str}

=== ENTRY POINT SNIPPETS ===
{snippets_str}
"""
    return context


def generate_summary(files: list[dict]) -> ArchitectureSummary:
    """
    Calls Groq API to generate a structured ArchitectureSummary from gather_context output.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is missing or empty.")

    context = gather_context(files)

    prompt = f"""You are a software architecture analysis assistant.
Analyze the following codebase context (README, file directory structure, and key entry point snippets):

{context}

Produce a structured high-level architectural overview.
Return ONLY a valid JSON object matching the following fields exactly:
- "main_modules": list of strings naming the core modules/components and their responsibilities
- "entry_points": list of strings identifying key entry points or execution flow triggers
- "tech_stack": list of strings specifying languages, major dependencies, frameworks, tools used
- "how_it_connects": a single detailed string explaining how data/control flows between modules
- "notable_patterns": list of strings highlighting architectural design patterns, software principles, or distinctive mechanisms

Do NOT include markdown formatting, preambles, or explanations outside the JSON object."""

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a software architecture expert. Output ONLY valid JSON matching the requested schema."
            },
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )

    raw_content = response.choices[0].message.content or ""

    # Safety fallback: strip accidental markdown code fences
    cleaned_content = raw_content.strip()
    if cleaned_content.startswith("```"):
        cleaned_content = re.sub(r"^```(?:json)?\s*", "", cleaned_content, flags=re.IGNORECASE)
        cleaned_content = re.sub(r"\s*```$", "", cleaned_content)

    try:
        parsed_json = json.loads(cleaned_content)
        summary = ArchitectureSummary.model_validate(parsed_json)
        return summary
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse or validate Groq response into ArchitectureSummary: {e}\nRaw Response:\n{raw_content}"
        ) from e


def get_or_create_summary(repo_name: str, files: list[dict]) -> ArchitectureSummary:
    """
    Checks if cached summary exists for repo_name, loads if present, otherwise generates via Groq and caches.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{repo_name}_summary.json")

    if os.path.exists(cache_path):
        print(f"[CACHE HIT] Loading existing summary for '{repo_name}' from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ArchitectureSummary.model_validate(data)
    else:
        print(f"[CACHE MISS] Generating fresh architecture summary for '{repo_name}' via Groq API...")
        summary = generate_summary(files)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(summary.model_dump_json(indent=2))
        print(f"Saved generated summary to {cache_path}")
        return summary
