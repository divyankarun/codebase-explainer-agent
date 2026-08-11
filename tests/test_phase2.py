"""Integration test script for Phase 2 (Language-Aware Chunking)."""

import sys
import os
import json

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chunking.splitter import chunk_all_files

CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".tsx"}


def run_phase2_test(cache_file_name: str):
    cache_path = os.path.join(os.path.dirname(__file__), "..", "data", "cache", cache_file_name)
    if not os.path.exists(cache_path):
        print(f"Cache file {cache_file_name} not found at {cache_path}")
        return

    print(f"\n=======================================================")
    print(f" TESTING PHASE 2 CHUNKING: {cache_file_name}")
    print(f"=======================================================")

    with open(cache_path, "r", encoding="utf-8") as f:
        file_dicts = json.load(f)

    chunks = chunk_all_files(file_dicts)

    py_count = 0
    doc_count = 0
    config_count = 0
    dotfile_empty_count = 0

    py_example = None
    md_example = None
    dotfile_example = None

    for c in chunks:
        path = c["file_path"]
        _, ext = os.path.splitext(path)
        ext_lower = ext.lower()

        # Categorization
        if ext_lower == ".py":
            py_count += 1
            if not py_example:
                py_example = c
        elif ext_lower in {".md", ".txt"}:
            doc_count += 1
            if not md_example:
                md_example = c
        elif ext_lower in CONFIG_EXTENSIONS or c["file_type"] == "config":
            config_count += 1
        elif ext_lower == "" or path.startswith("."):
            dotfile_empty_count += 1
            if not dotfile_example:
                dotfile_example = c
        else:
            doc_count += 1

    print(f"Total files in cache:        {len(file_dicts)}")
    print(f"Total chunks produced:       {len(chunks)}")
    print(f"\nChunk breakdown:")
    print(f" - .py chunks:               {py_count}")
    print(f" - .md/doc chunks:           {doc_count}")
    print(f" - config chunks:            {config_count}")
    print(f" - empty ext / dotfile chunks: {dotfile_empty_count}")

    print("\n-------------------------------------------------------")
    print(" 3 EXAMPLE CHUNKS WITH FULL METADATA")
    print("-------------------------------------------------------")

    examples = [
        ("Python (.py) Chunk", py_example),
        ("Documentation (.md) Chunk", md_example),
        ("Empty Extension / Dotfile Chunk", dotfile_example),
    ]

    for title, ex in examples:
        print(f"\n>>> [{title}]")
        if ex:
            print(f"  file_path  : {ex['file_path']}")
            print(f"  start_line : {ex['start_line']}")
            print(f"  end_line   : {ex['end_line']}")
            print(f"  language   : {ex['language']}")
            print(f"  file_type  : {ex['file_type']}")
            preview = ex['text'][:120].replace('\n', ' ')
            print(f"  text preview: \"{preview}...\"")

            # Verification check
            matching_file = next((f for f in file_dicts if f["path"] == ex["file_path"]), None)
            if matching_file:
                lines = matching_file["content"].splitlines()
                extracted = "\n".join(lines[ex["start_line"] - 1 : ex["end_line"]])
                is_valid = ex["text"].strip() in extracted.strip() or extracted.strip() in ex["text"].strip()
                print(f"  Line Verification: {'SUCCESS' if is_valid else 'FAILED'} (Extracted lines {ex['start_line']}-{ex['end_line']})")
        else:
            print("  (No chunk available for this category in test data)")

if __name__ == "__main__":
    run_phase2_test("psf_requests_files.json")
    run_phase2_test("divyankarun_Rag-Chatbot_files.json")
