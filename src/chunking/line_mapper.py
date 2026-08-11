"""Utility to map split chunks back to precise source file line numbers."""

def map_chunk_to_lines(chunk_text: str, original_text: str) -> tuple[int, int]:
    """Find where chunk_text sits inside original_text using string search.
    
    Args:
        chunk_text: The string chunk extracted from original_text.
        original_text: The complete original file content.
        
    Returns:
        1-indexed tuple of (start_line, end_line). Returns (0, 0) if chunk_text
        cannot be found in original_text.
    """
    if not chunk_text or not original_text:
        return 0, 0

    start_idx = original_text.find(chunk_text)
    if start_idx == -1:
        return 0, 0

    start_line = original_text[:start_idx].count('\n') + 1
    end_idx = start_idx + len(chunk_text)
    
    # Calculate end_line based on newlines up to end_idx
    end_line = original_text[:end_idx].count('\n') + 1
    
    return start_line, end_line
