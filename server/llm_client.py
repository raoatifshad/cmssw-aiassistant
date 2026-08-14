"""CERN LiteLLM gateway client. Standalone — no imports from any other project."""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
DEFAULT_MODEL = os.environ.get("CERN_MODEL", "hf-qwen3-32b-awq")


def get_client() -> OpenAI:
    base_url = os.environ.get("CERN_API_BASE")
    api_key = os.environ.get("CERN_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("Set CERN_API_BASE and CERN_API_KEY in .env")
    return OpenAI(base_url=base_url, api_key=api_key)
