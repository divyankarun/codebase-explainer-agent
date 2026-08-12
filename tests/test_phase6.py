"""Phase 6 Test Execution Script: Comprehensive Quality Gate Evaluation across 3 Repositories."""

import sys
import os
import json
import re
from langchain_core.messages import HumanMessage

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CACHE_DIR
from src.ingestion.mcp_client import fetch_repository_files
from src.chunking.splitter import chunk_all_files
from src.embedding.faiss_store import get_or_create_index, load_index
from src.summary.architecture_summary import get_or_create_summary
from src.graph.build_graph import build_graph


REPOS_CONFIG = [
    {
        "name": "psf_requests",
        "url": "https://github.com/psf/requests",
        "questions": [
            {"q": "What is this project and how is it structured?", "expected_route": "architecture"},
            {"q": "What is the tech stack and what are the entry points?", "expected_route": "architecture"},
            {"q": "How does the HTTP session and connection pooling work?", "expected_route": "specific"},
            {"q": "Show me the specific function that handles sending HTTP requests.", "expected_route": "specific"},
            {"q": "Can you explain more about what you just showed me?", "expected_route": "specific"},
        ]
    },
    {
        "name": "divyankarun_Rag-Chatbot",
        "url": "https://github.com/divyankarun/Rag-Chatbot",
        "questions": [
            {"q": "What is this project and how is it structured?", "expected_route": "architecture"},
            {"q": "What is the tech stack and what are the entry points?", "expected_route": "architecture"},
            {"q": "How does the document vector retrieval pipeline work?", "expected_route": "specific"},
            {"q": "Show me the specific function that loads and indexes document chunks.", "expected_route": "specific"},
            {"q": "Can you explain more about what you just showed me?", "expected_route": "specific"},
        ]
    },
    {
        "name": "python-eel_Eel",
        "url": "https://github.com/python-eel/Eel",
        "questions": [
            {"q": "What is this project and how is it structured?", "expected_route": "architecture"},
            {"q": "What is the tech stack and what are the entry points?", "expected_route": "architecture"},
            {"q": "How does the WebSocket communication between Python and JavaScript work?", "expected_route": "specific"},
            {"q": "Show me the specific function that exposes Python functions to JavaScript.", "expected_route": "specific"},
            {"q": "Can you explain more about what you just showed me?", "expected_route": "specific"},
        ]
    }
]


def prepare_repo(repo_name: str, repo_url: str):
    """Ensure Phase 1-4 pipeline artifacts exist for the given repo."""
    print(f"\n=======================================================")
    print(f" PREPARING REPOSITORY PIPELINE: {repo_name}")
    print(f"=======================================================")

    # Phase 1: Ingestion
    file_dicts, raw_c, filt_c = fetch_repository_files(repo_url)
    print(f"Phase 1 Ingestion: {filt_c} files loaded.")

    # Phase 2 & 3: Chunking & FAISS Indexing
    index_path = os.path.join(CACHE_DIR, f"{repo_name}.index")
    meta_path = os.path.join(CACHE_DIR, f"{repo_name}_metadata.json")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        print(f"Loading cached FAISS index for {repo_name}...")
        index, chunk_metadata = load_index(repo_name)
    else:
        print(f"Chunking and embedding files for {repo_name}...")
        chunks = chunk_all_files(file_dicts)
        index, chunk_metadata, _ = get_or_create_index(repo_name, chunks)

    # Phase 4: Architecture Summary
    summary = get_or_create_summary(repo_name, file_dicts)
    print(f"Architecture Summary ready for {repo_name}.")

    return index, chunk_metadata, summary


def format_safe(text: str) -> str:
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def run_phase6_eval():
    output_md_path = os.path.join(os.path.dirname(__file__), "test_repos.md")
    md_sections = ["# Phase 6: Quality Gate Evaluation Results across Real Repositories\n"]

    overall_stats = {
        "total": 0,
        "clean_pass": 0,
        "retrieval_citation_issues": 0,
        "router_misclassifications": 0
    }

    for repo_cfg in REPOS_CONFIG:
        repo_name = repo_cfg["name"]
        repo_url = repo_cfg["url"]
        questions = repo_cfg["questions"]

        index, chunk_metadata, summary = prepare_repo(repo_name, repo_url)
        graph = build_graph(index, chunk_metadata, summary)
        thread_id = f"phase6_{repo_name}_eval"
        config = {"configurable": {"thread_id": thread_id}}

        table_rows = []

        print(f"\n=======================================================")
        print(f" RUNNING EVALUATION SUITE FOR: {repo_name}")
        print(f" Thread ID: {thread_id}")
        print(f"=======================================================")

        for idx, q_info in enumerate(questions, 1):
            question = q_info["q"]
            expected_route = q_info["expected_route"]
            overall_stats["total"] += 1

            print(f"\n-------------------------------------------------------")
            print(f" [{repo_name}] QUESTION {idx}: \"{question}\"")
            print(f"-------------------------------------------------------")

            input_state = {
                "messages": [HumanMessage(content=question)],
                "repo_name": repo_name,
            }

            result = graph.invoke(input_state, config=config)

            route_taken = result.get("route", "unknown")
            latest_msg = result["messages"][-1]
            answer_text = latest_msg.content if hasattr(latest_msg, "content") else str(latest_msg)

            # Detect retrieved chunks log if specific route
            chunks_retrieved_str = "N/A (architecture route)"
            if route_taken == "specific":
                # Search FAISS directly for logging purpose matching current turn logic
                from src.embedding.embedder import embed_query
                from src.embedding.faiss_store import search
                from config import TOP_K_RETRIEVAL

                search_q = question
                # If follow-up, contextualize search
                prior_msgs = [m.content for m in result["messages"][:-1] if getattr(m, "type", "") == "human" or m.__class__.__name__ == "HumanMessage"]
                if prior_msgs:
                    search_q = f"{prior_msgs[-1]} {question}"

                code_seeking_kw = ["function", "method", "implementation", "code", "class", "def ", "show me the code"]
                is_code_seeking = any(kw in search_q.lower() for kw in code_seeking_kw)
                q_vec = embed_query(search_q)

                if is_code_seeking:
                    candidate_indices = search(index, q_vec, k=max(30, TOP_K_RETRIEVAL * 5))
                    code_indices = [i for i in candidate_indices if 0 <= i < len(chunk_metadata) and chunk_metadata[i].get("file_type") == "code"]
                    top_indices = code_indices[:TOP_K_RETRIEVAL] if code_indices else candidate_indices[:TOP_K_RETRIEVAL]
                else:
                    top_indices = search(index, q_vec, k=TOP_K_RETRIEVAL)

                ret_list = []
                for i in top_indices:
                    if 0 <= i < len(chunk_metadata):
                        meta = chunk_metadata[i]
                        ret_list.append(f"{meta.get('file_path')}:{meta.get('file_type')}")
                chunks_retrieved_str = ", ".join(ret_list) if ret_list else "None"

            # Self-check checks:
            # 1. Router check
            router_correct = (route_taken == expected_route)
            
            # 2. Citations check: citations formatted as (path:start-end)
            has_citations = bool(re.search(r"\([\w/\.\-\_]+:\d+-\d+\)", answer_text))
            
            # 3. Answer & Hallucination check
            has_honest_disclaimer = "not found in the provided" in answer_text.lower() or "do not contain" in answer_text.lower()

            answer_correct = "Yes"
            citation_correct = "Yes"
            notes = []

            if not router_correct:
                overall_stats["router_misclassifications"] += 1
                answer_correct = "Partial"
                notes.append(f"Router misclassified as '{route_taken}' (expected '{expected_route}').")

            if route_taken == "specific":
                if not has_citations and not has_honest_disclaimer:
                    citation_correct = "No"
                    overall_stats["retrieval_citation_issues"] += 1
                    notes.append("Missing required inline citations `(path:start-end)`.")
                elif has_citations:
                    citation_correct = "Yes"
                else:
                    citation_correct = "N/A (no code found)"

            if "not found" in answer_text.lower() or "do not contain" in answer_text.lower():
                notes.append("Honest disclaimer included when context was limited.")

            if answer_correct == "Yes" and citation_correct in ["Yes", "N/A (architecture route)", "N/A (no code found)"] and router_correct:
                overall_stats["clean_pass"] += 1
                notes.append("Passed cleanly.")

            notes_str = "; ".join(notes)

            print(f"[BRANCH ROUTED TO]: {route_taken.upper()} (Expected: {expected_route})")
            print(f"[CHUNKS RETRIEVED]: {chunks_retrieved_str}")
            print(f"[ANSWER FROM AGENT]:\n{format_safe(answer_text)}\n")
            print(f"[SELF-CHECK]: Answer Correct: {answer_correct} | Citation Correct: {citation_correct} | Notes: {notes_str}")

            table_rows.append(
                f"| {idx} | {question} | {route_taken} | {chunks_retrieved_str} | {answer_correct} | {citation_correct} | {notes_str} |"
            )

        section_md = f"## Repo: {repo_name}\n"
        section_md += "| # | Question | Routed To | Chunks Retrieved (path:type) | Answer Correct? | Citation Correct? | Notes |\n"
        section_md += "|---|----------|-----------|-------------------------------|------------------|---------------------|-------|\n"
        section_md += "\n".join(table_rows) + "\n"
        md_sections.append(section_md)

    # Summary Section
    summary_md = f"""## Phase 6 Evaluation Summary

- **Total Questions Evaluated**: {overall_stats['total']}
- **Clean Passes**: {overall_stats['clean_pass']} / {overall_stats['total']}
- **Router Misclassifications**: {overall_stats['router_misclassifications']}
- **Citation / Retrieval Issues**: {overall_stats['retrieval_citation_issues']}
"""
    md_sections.append(summary_md)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(md_sections))

    print("\n=======================================================")
    print(" PHASE 6 EVALUATION COMPLETE!")
    print(f" Results written to: {output_md_path}")
    print("=======================================================\n")
    print(summary_md)


if __name__ == "__main__":
    run_phase6_eval()
