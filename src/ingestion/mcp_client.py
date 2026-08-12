"""MCP client implementation to connect with GitHub MCP server over stdio transport."""

import asyncio
import json
import os
import re
from typing import Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import CACHE_DIR, GITHUB_TOKEN
from src.ingestion.filters import filter_files


def parse_github_url(url: str) -> tuple[str, str]:
    """Parse GitHub URL or 'owner/repo' string into (owner, repo)."""
    clean_url = url.strip().rstrip("/")
    if clean_url.endswith(".git"):
        clean_url = clean_url[:-4]
    
    match = re.search(r"github\.com/([^/]+)/([^/]+)", clean_url)
    if match:
        return match.group(1), match.group(2)
        
    parts = clean_url.split("/")
    if len(parts) == 2:
        return parts[0], parts[1]
        
    raise ValueError(f"Invalid GitHub repository URL or format: {url}")


def get_cache_path(owner: str, repo: str) -> str:
    """Get the local cache file path for a repository."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{owner}_{repo}_files.json")


async def _fetch_tree_recursive(session: ClientSession, owner: str, repo: str, path: str = "") -> list[str]:
    """Recursively fetch all file paths in the repository using MCP get_file_contents."""
    file_paths = []
    
    try:
        res = await session.call_tool("get_file_contents", arguments={"owner": owner, "repo": repo, "path": path})
        if not res.content:
            return file_paths
            
        # Directory listings are returned in res.content[0].text or EmbeddedResource text
        text_data = ""
        for block in res.content:
            resource = getattr(block, "resource", None)
            if resource and getattr(resource, "text", None):
                text_data = resource.text
                break
            elif getattr(block, "text", None) and not block.text.startswith("successfully downloaded"):
                text_data = block.text
                break

        if not text_data and res.content:
            text_data = getattr(res.content[0], "text", "")
        
        try:
            items = json.loads(text_data)
        except (json.JSONDecodeError, TypeError):
            if path:
                return [path]
            return file_paths

        if isinstance(items, list):
            for item in items:
                item_path = item.get("path", "")
                item_type = item.get("type", "")
                
                if item_type == "file":
                    file_paths.append(item_path)
                elif item_type == "dir":
                    sub_files = await _fetch_tree_recursive(session, owner, repo, item_path)
                    file_paths.extend(sub_files)
        elif isinstance(items, dict):
            item_type = items.get("type", "")
            if item_type == "file":
                file_paths.append(items.get("path", path))
            elif item_type == "dir":
                sub_files = await _fetch_tree_recursive(session, owner, repo, items.get("path", path))
                file_paths.extend(sub_files)
    except Exception as e:
        print(f"Error fetching directory contents for path '{path}': {e}")
        
    return file_paths


async def _fetch_file_content(session: ClientSession, owner: str, repo: str, path: str) -> str:
    """Fetch content of a single file using MCP get_file_contents tool."""
    try:
        res = await session.call_tool("get_file_contents", arguments={"owner": owner, "repo": repo, "path": path})
        if not res or not res.content:
            return ""
        
        # 1. Search EmbeddedResource for text content
        for block in res.content:
            resource = getattr(block, "resource", None)
            if resource:
                text = getattr(resource, "text", None)
                if text is not None:
                    return text
        
        # 2. Fallback to TextContent text
        for block in res.content:
            text = getattr(block, "text", None)
            if text and not text.startswith("successfully downloaded"):
                return text
                
        return ""
    except Exception as e:
        # Zero-byte files or unparseable blocks return empty string content safely
        return ""




def _fetch_via_github_api(owner: str, repo: str) -> tuple[list[dict[str, str]], int, int]:
    """Fallback fetcher using GitHub REST API when Docker MCP is unavailable."""
    import urllib.request
    print(f"Using GitHub REST API fallback for {owner}/{repo}...")
    headers = {"User-Agent": "CodebaseExplainerAgent"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        tree_data = json.loads(resp.read().decode("utf-8"))
    
    tree = tree_data.get("tree", [])
    raw_paths = [item["path"] for item in tree if item.get("type") == "blob"]
    raw_count = len(raw_paths)
    
    filtered_paths = filter_files(raw_paths)
    filtered_count = len(filtered_paths)
    
    file_dicts = []
    for path in filtered_paths:
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}"
        try:
            r = urllib.request.Request(raw_url, headers=headers)
            with urllib.request.urlopen(r) as resp:
                content = resp.read().decode("utf-8", errors="replace")
        except Exception:
            content = ""
        _, ext = os.path.splitext(path)
        file_dicts.append({
            "path": path,
            "content": content,
            "extension": ext
        })
    
    cache_path = get_cache_path(owner, repo)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(file_dicts, f, indent=2)
        
    return file_dicts, raw_count, filtered_count


async def fetch_repository_files_async(repo_url: str) -> tuple[list[dict[str, str]], int, int]:
    """Fetch and filter repository files using GitHub MCP Server via stdio Docker transport,
    falling back to GitHub REST API if Docker MCP is unavailable.
    
    Returns:
        tuple of (file_dicts, raw_file_count, filtered_file_count)
        where file_dicts is a list of {"path": str, "content": str, "extension": str}
    """
    owner, repo = parse_github_url(repo_url)
    cache_path = get_cache_path(owner, repo)
    
    # Return cached data if present
    if os.path.exists(cache_path):
        print(f"Loading cached repository data from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_files = json.load(f)
        return cached_files, len(cached_files), len(cached_files)

    # Check if Docker is available and daemon is responsive
    docker_available = False
    try:
        import subprocess
        res = subprocess.run(["docker", "info"], capture_output=True, timeout=3)
        if res.returncode == 0:
            docker_available = True
    except Exception:
        docker_available = False

    if not docker_available:
        print(f"[INFO] Docker daemon is not running/available. Using GitHub REST API fallback for {owner}/{repo}...")
        return _fetch_via_github_api(owner, repo)

    server_params = StdioServerParameters(
        command="docker",
        args=[
            "run",
            "-i",
            "--rm",
            "-e",
            f"GITHUB_PERSONAL_ACCESS_TOKEN={GITHUB_TOKEN}",
            "ghcr.io/github/github-mcp-server",
        ],
        env=None,
    )
    
    print(f"Connecting to GitHub MCP server for {owner}/{repo}...")
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 1. Fetch full file tree
                raw_paths = await _fetch_tree_recursive(session, owner, repo, path="")
                raw_count = len(raw_paths)
                
                # 2. Filter paths
                filtered_paths = filter_files(raw_paths)
                filtered_count = len(filtered_paths)
                
                # 3. Fetch contents for filtered files
                file_dicts = []
                for path in filtered_paths:
                    content = await _fetch_file_content(session, owner, repo, path)
                    _, ext = os.path.splitext(path)
                    file_dicts.append({
                        "path": path,
                        "content": content,
                        "extension": ext
                    })
                    
                # 4. Cache results
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(file_dicts, f, indent=2)
                    
                return file_dicts, raw_count, filtered_count
    except Exception as e:
        print(f"[WARNING] GitHub MCP Docker connection failed: {e}. Falling back to GitHub REST API...")
        return _fetch_via_github_api(owner, repo)



def fetch_repository_files(repo_url: str) -> tuple[list[dict[str, str]], int, int]:
    """Synchronous wrapper for fetch_repository_files_async."""
    return asyncio.run(fetch_repository_files_async(repo_url))


