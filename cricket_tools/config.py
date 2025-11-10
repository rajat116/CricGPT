"""
cricket_tools/config.py
=======================
Central configuration for LLM providers and global options.
"""

import os
import logging

log = logging.getLogger("CricketConfig")

def get_llm_client():
    """Return a (provider, model, client) triple for use in agent."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model = None
    client = None

    if provider == "openai":
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Missing OPENAI_API_KEY in environment")
        client = OpenAI(api_key=key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    elif provider == "gemini":
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError("Please `pip install google-generativeai`") from e
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Missing GEMINI_API_KEY in environment")
        genai.configure(api_key=key)
        client = genai
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    elif provider == "ollama":
        import requests
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        client = requests.Session()
        client.base_url = base_url
        model = os.getenv("OLLAMA_MODEL", "llama3")

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    log.info(f"✅ Using LLM provider={provider}, model={model}")
    return provider, model, client