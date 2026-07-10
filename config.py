"""
config.py
─────────
Loads environment variables and creates singleton API client instances.
Both agents import from here so secrets are loaded from a single place.
"""

import os
from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import InferenceClient

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
HF_API_KEY: str = os.environ["HF_API_KEY"]

# ── Model Identifiers ─────────────────────────────────────────────────────────
INTERVIEWER_MODEL: str = os.getenv("INTERVIEWER_MODEL", "openai/gpt-oss-120b")
EVALUATOR_MODEL: str = os.getenv("EVALUATOR_MODEL", "openai/gpt-oss-120b")

# ── Tag history cap (guards against unbounded prompt growth) ──────────────────
MAX_TAGS_IN_PROMPT: int = int(os.getenv("MAX_TAGS_IN_PROMPT", "80"))

# ── Singleton clients ─────────────────────────────────────────────────────────
groq_client: Groq = Groq(api_key=GROQ_API_KEY)
hf_client: InferenceClient = InferenceClient(token=GROQ_API_KEY)
