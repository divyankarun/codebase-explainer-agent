"""Language-aware code text splitter wrapping LangChain splitters."""

from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from config import CHUNK_SIZE, CHUNK_OVERLAP
from src.chunking.line_mapper import map_chunk_to_lines

CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".tsx"}


def get_splitter_for_extension(ext: str) -> Any:
    """Get the appropriate text splitter based on file extension.
    
    Args:
        ext: File extension (e.g. '.py', '.md', '')
        
    Returns:
        RecursiveCharacterTextSplitter instance or None for config files.
    """
    ext_lower = ext.lower() if ext else ""
    
    if ext_lower == ".py":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    elif ext_lower in {".js", ".ts", ".tsx"}:
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.JS,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    elif ext_lower in {".md", ".txt"}:
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    elif ext_lower in CONFIG_EXTENSIONS:
        return None
    else:
        # Any other extension, including empty string "" (dotfiles like .gitignore, .coveragerc)
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )


def chunk_file(file_obj: dict[str, str]) -> list[dict[str, Any]]:
    """Chunk a single file object into structured chunk dictionaries.
    
    Args:
        file_obj: Dict with keys 'path', 'content', 'extension'
        
    Returns:
        List of chunk dicts: {"text": str, "file_path": str, "start_line": int,
                             "end_line": int, "language": str, "file_type": str}
    """
    path = file_obj.get("path", "")
    content = file_obj.get("content", "")
    ext = file_obj.get("extension", "")
    ext_lower = ext.lower() if ext else ""

    if ext_lower in CONFIG_EXTENSIONS:
        lines = content.splitlines()
        total_lines = len(lines) if lines else 1
        return [{
            "text": content,
            "file_path": path,
            "start_line": 1,
            "end_line": total_lines,
            "language": "config",
            "file_type": "config"
        }]

    splitter = get_splitter_for_extension(ext_lower)
    if not content or not content.strip():
        return []

    raw_chunks = splitter.split_text(content)
    chunks = []

    lang = ext_lower.lstrip(".") if ext_lower else "plaintext"
    file_type = "code" if ext_lower in CODE_EXTENSIONS else "doc"

    for c in raw_chunks:
        start_line, end_line = map_chunk_to_lines(c, content)
        chunks.append({
            "text": c,
            "file_path": path,
            "start_line": start_line,
            "end_line": end_line,
            "language": lang,
            "file_type": file_type
        })

    return chunks


def chunk_all_files(files: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Process all file objects in a repository into a flat list of chunks."""
    all_chunks = []
    for f in files:
        all_chunks.extend(chunk_file(f))
    return all_chunks
