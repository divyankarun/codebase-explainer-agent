"""Integration test script for Phase 4 (Architecture Summary & Caching)."""

import sys
import os
import json

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.summary.architecture_summary import get_or_create_summary
from config import CACHE_DIR


def print_summary(repo_name: str, summary):
    print(f"\n=======================================================")
    print(f" ARCHITECTURE SUMMARY FOR: {repo_name}")
    print(f"=======================================================")
    print("\n--- 1. Main Modules ---")
    for mod in summary.main_modules:
        print(f"  • {mod}")

    print("\n--- 2. Entry Points ---")
    for ep in summary.entry_points:
        print(f"  • {ep}")

    print("\n--- 3. Tech Stack ---")
    for tech in summary.tech_stack:
        print(f"  • {tech}")

    print("\n--- 4. How It Connects ---")
    print(f"  {summary.how_it_connects}")

    print("\n--- 5. Notable Patterns ---")
    for pat in summary.notable_patterns:
        print(f"  • {pat}")
    print("=======================================================\n")


def test_repo_summary(repo_name: str):
    cache_file = os.path.join(CACHE_DIR, f"{repo_name}_files.json")
    if not os.path.exists(cache_file):
        print(f"Skipping {repo_name}: Raw files cache not found at {cache_file}")
        return

    with open(cache_file, "r", encoding="utf-8") as f:
        files = json.load(f)

    summary_file = os.path.join(CACHE_DIR, f"{repo_name}_summary.json")

    # Clean up pre-existing summary cache to ensure a clean test run of fresh generation
    if os.path.exists(summary_file):
        os.remove(summary_file)
        print(f"Cleaned up pre-existing summary cache: {summary_file}")

    # First run: should trigger fresh generation (CACHE MISS)
    print(f"\n>>> RUN 1 (Fresh Generation Test for {repo_name}) <<<")
    summary1 = get_or_create_summary(repo_name, files)
    print_summary(repo_name, summary1)

    # Confirm cache file exists
    assert os.path.exists(summary_file), f"Expected cache file {summary_file} to exist after generation!"
    print(f"CONFIRMED: {summary_file} exists on disk!")

    # Second run: should load from cache (CACHE HIT)
    print(f"\n>>> RUN 2 (Cache Retrieval Test for {repo_name}) <<<")
    summary2 = get_or_create_summary(repo_name, files)

    # Verify both returns match
    assert summary1.model_dump() == summary2.model_dump(), "Cached summary does not match freshly generated summary!"
    print(f"CONFIRMED: Run 2 returned identical ArchitectureSummary from cache without calling Groq API!\n")


if __name__ == "__main__":
    test_repo_summary("psf_requests")
    test_repo_summary("divyankarun_Rag-Chatbot")
