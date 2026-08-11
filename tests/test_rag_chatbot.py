import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.mcp_client import fetch_repository_files

def test_rag_chatbot_ingestion():
    repo_url = "https://github.com/divyankarun/Rag-Chatbot"
    print(f"Testing ingestion for {repo_url}...")
    
    file_dicts, raw_count, filtered_count = fetch_repository_files(repo_url)
    
    print("\n--- PHASE 1 TEST RESULTS ---")
    print(f"File count before filtering: {raw_count}")
    print(f"File count after filtering:  {filtered_count}")
    print("\nFirst 3 entries of the result:")
    for i, item in enumerate(file_dicts[:3]):
        preview = item['content'][:100].replace('\n', ' ')
        print(f"[{i+1}] Path: {item['path']} (ext: {item['extension']})")
        print(f"    Content preview: {preview}...")

if __name__ == "__main__":
    test_rag_chatbot_ingestion()
