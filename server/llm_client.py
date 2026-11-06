"""CERN LiteLLM gateway client. Standalone — no imports from any other project."""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
DEFAULT_MODEL = os.environ.get("CERN_MODEL", "hf-qwen3-32b-awq")
# A second opinion from a different model: it checks what DEFAULT_MODEL concluded (user, 2026-11-05: "don't blindly trust the results"). Measured on
# pasted errors: glm-5.3-flash-int4 says "same problem" too easily, gpt-oss-20b too rarely, and they rarely err on the same item.
REVIEW_MODEL = os.environ.get("CERN_REVIEW_MODEL", "gpt-oss-20b")


# The openai SDK's own default timeout is very long (10 minutes) -- under
# real load (found live via scripts/stress_test_bot.py, 2026-09-22) a slow
# gateway can leave a request hanging well past what's reasonable for a chat
# response, with nothing surfaced to the user until it eventually resolves
# or the process itself times out. Bounding it here means every call through
# this client fails fast enough for callers (e.g. _run_llm_investigation) to
# degrade to a friendly, retryable message instead.
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("CERN_API_TIMEOUT_SECONDS", "25"))


def get_client() -> OpenAI:
    base_url = os.environ.get("CERN_API_BASE")
    api_key = os.environ.get("CERN_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("Set CERN_API_BASE and CERN_API_KEY in .env")
    # the SDK retries twice by default, which turned a 15 s limit into 45 s of silence (found 2026-11-05); each caller has its own way to go on without the reply
    return OpenAI(base_url=base_url, api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
