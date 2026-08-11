"""Configuration constants and environment settings for Codebase Explainer Agent."""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 4
MAX_FILES = 200
CACHE_DIR = "data/cache"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = "llama-3.3-70b-versatile"
