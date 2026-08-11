"""Integration test script for Phase 3 (Embedding + FAISS Indexing)."""

import sys
import os
import json

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chunking.splitter import chunk_all_files
from src.embedding.embedder import embed_query, embed_chunks
from src.embedding.faiss_store import get_or_create_index, search
from config import CACHE_DIR, TOP_K_RETRIEVAL


def test_phase3_pipeline(repo_name: str, test_query: str):
    cache_file = os.path.join(CACHE_DIR, f"{repo_name}_files.json")
    if not os.path.exists(cache_file):
        print(f"Skipping {repo_name}: Cache file not found at {cache_file}")
        return

    print(f"\n=======================================================")
    print(f" TESTING PHASE 3 PIPELINE: {repo_name}")
    print(f"=======================================================")

    # 1. Load Phase 1 raw files & Chunk in Phase 2
    with open(cache_file, "r", encoding="utf-8") as f:
        raw_files = json.load(f)

    chunks = chunk_all_files(raw_files)
    print(f"Phase 2 produced {len(chunks)} total chunks from {len(raw_files)} files.")

    # 2. Embed & Index in Phase 3
    index, cached_chunks, new_embeddings = get_or_create_index(repo_name, chunks)

    # 3. Check embedding shape & file creation
    if new_embeddings is None:
        emb_shape = (index.ntotal, index.d)
    else:
        emb_shape = new_embeddings.shape

    print(f"Embedding Array Shape: {emb_shape} ([num_chunks, dim])")

    index_path = os.path.join(CACHE_DIR, f"{repo_name}.index")
    meta_path = os.path.join(CACHE_DIR, f"{repo_name}_metadata.json")

    print(f"Index File Exists ({index_path}) : {os.path.exists(index_path)}")
    print(f"Metadata File Exists ({meta_path}): {os.path.exists(meta_path)}")

    # 4. Search Query Test
    print(f"\n-------------------------------------------------------")
    print(f" TOP {TOP_K_RETRIEVAL} SEARCH RETRIEVAL TEST")
    print(f" Query: \"{test_query}\"")
    print(f"-------------------------------------------------------")

    query_vec = embed_query(test_query)
    top_indices = search(index, query_vec, k=TOP_K_RETRIEVAL)

    for i, idx in enumerate(top_indices, 1):
        if idx < len(cached_chunks):
            c = cached_chunks[idx]
            preview = c["text"][:100].replace("\n", " ")
            print(f"[{i}] Path       : {c['file_path']} (Lines {c['start_line']}-{c['end_line']})")
            print(f"    Language   : {c['language']} | Type: {c['file_type']}")
            print(f"    Text Snippet: \"{preview}...\"\n")
        else:
            print(f"[{i}] Invalid index: {idx}")


if __name__ == "__main__":
    test_phase3_pipeline("psf_requests", "how does authentication work")
    test_phase3_pipeline("divyankarun_Rag-Chatbot", "how does basic rag pipeline work")
