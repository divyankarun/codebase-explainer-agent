"""Integration test script for Phase 1 (MCP client setup and repository file ingestion)."""

import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.mcp_client import fetch_repository_files
from src.ingestion.filters import filter_files


def test_filters_unit():
    """Unit test for filter_files logic."""
    raw_paths = [
        "src/requests/__init__.py",
        ".git/config",
        "node_modules/express/index.js",
        "dist/bundle.js",
        "build/main.o",
        "src/__pycache__/app.cpython-311.pyc",
        ".venv/lib/site-packages/pip/__init__.py",
        "images/logo.png",
        "fonts/inter.woff2",
        "package-lock.json",
        "poetry.lock",
        "README.md",
        "src/requests/api.py"
    ]
    filtered = filter_files(raw_paths)
    print("Unit Test filter_files output:")
    for p in filtered:
        print(" -", p)
    assert "src/requests/__init__.py" in filtered
    assert "README.md" in filtered
    assert "src/requests/api.py" in filtered
    assert ".git/config" not in filtered
    assert "node_modules/express/index.js" not in filtered
    assert "images/logo.png" not in filtered
    assert "package-lock.json" not in filtered
    print("Unit test passed successfully!\n")


def test_psf_requests_ingestion():
    """Integration test against https://github.com/psf/requests."""
    repo_url = "https://github.com/psf/requests"
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
    test_filters_unit()
    test_psf_requests_ingestion()
