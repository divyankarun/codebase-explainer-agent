"""File filtering utilities for repo size, language extensions, and ignored paths."""

import os
from config import MAX_FILES

IGNORED_DIRS = {".git", "node_modules", "dist", "build", "__pycache__", ".venv"}
BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".woff", ".woff2", ".ico", ".svg"}
LOCKFILES = {"package-lock.json", "poetry.lock", "yarn.lock"}

def filter_files(file_paths: list[str]) -> list[str]:
    """Filter raw repository file paths based on directory exclusions, extensions, and limits.
    
    Args:
        file_paths: List of relative file paths in the repository.
        
    Returns:
        Filtered list of file paths capped at MAX_FILES.
    """
    filtered = []
    
    for path in file_paths:
        # Normalize path separators
        normalized_path = path.replace("\\", "/")
        parts = normalized_path.split("/")
        
        # 1. Directory exclusion: check if any path segment matches ignored directories
        if any(part in IGNORED_DIRS for part in parts[:-1]):
            continue
            
        filename = parts[-1]
        
        # 2. Lockfile exclusion
        if filename in LOCKFILES:
            continue
            
        # 3. Binary/image extension exclusion
        _, ext = os.path.splitext(filename)
        if ext.lower() in BINARY_EXTENSIONS:
            continue
            
        filtered.append(path)
        
        if len(filtered) >= MAX_FILES:
            break
            
    return filtered

