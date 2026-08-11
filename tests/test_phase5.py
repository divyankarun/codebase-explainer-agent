"""Integration test script for Phase 5 (LangGraph Orchestration)."""

import sys
import os
import json
from langchain_core.messages import HumanMessage

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CACHE_DIR
from src.embedding.faiss_store import load_index
from src.graph.build_graph import build_graph


def run_phase5_test(repo_name: str = "psf_requests", turns: list[str] = None):
    print(f"\n=======================================================")
    print(f" RUNNING PHASE 5 INTEGRATION TEST FOR: {repo_name}")
    print(f"=======================================================")

    # 1. Load cached index, chunk metadata, and summary
    summary_path = os.path.join(CACHE_DIR, f"{repo_name}_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Cached summary not found at {summary_path}")

    index, chunk_metadata = load_index(repo_name)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print(f"Loaded FAISS index ({index.ntotal} vectors), {len(chunk_metadata)} chunk dicts, and architecture summary.")

    # 2. Build graph
    graph = build_graph(index, chunk_metadata, summary)
    thread_id = f"phase5_{repo_name}_session"
    config = {"configurable": {"thread_id": thread_id}}

    # 3. 3-Turn Conversation Test
    if not turns:
        turns = [
            "What is this project and how is it structured?",
            "How does the authentication system work?",
            "Can you show me the specific function that handles that?"
        ]

    for turn_num, question in enumerate(turns, 1):
        print(f"\n-------------------------------------------------------")
        print(f" TURN {turn_num} QUESTION: \"{question}\"")
        print(f"-------------------------------------------------------")

        input_state = {
            "messages": [HumanMessage(content=question)],
            "repo_name": repo_name,
        }

        # Invoke graph (persists state across turns via thread_id)
        result = graph.invoke(input_state, config=config)

        route_taken = result.get("route", "unknown")
        latest_msg = result["messages"][-1]
        answer_text = latest_msg.content if hasattr(latest_msg, "content") else str(latest_msg)

        print(f"[BRANCH ROUTED TO]: {route_taken.upper()}")
        enc = sys.stdout.encoding or "utf-8"
        safe_answer = answer_text.encode(enc, errors="replace").decode(enc, errors="replace")
        print(f"[ANSWER FROM AGENT]:\n{safe_answer}\n")

    # 4. Verify conversation memory for Turn 3
    final_messages = graph.get_state(config).values.get("messages", [])
    print(f"Total messages accumulated in state for thread '{thread_id}': {len(final_messages)}")
    assert len(final_messages) >= len(turns) * 2, f"Expected at least {len(turns) * 2} messages in thread state memory!"

    print(f"\n[SUCCESS] PHASE 5 INTEGRATION TEST PASSED FOR '{repo_name}'!")


if __name__ == "__main__":
    # Test 1: psf_requests
    run_phase5_test("psf_requests")

    # Test 2: divyankarun_Rag-Chatbot
    rag_chatbot_turns = [
        "What is this project and how is it structured?",
        "How does the basic RAG retrieval pipeline work?",
        "Can you show me the specific function that loads and indexes the document chunks?"
    ]
    run_phase5_test("divyankarun_Rag-Chatbot", turns=rag_chatbot_turns)
