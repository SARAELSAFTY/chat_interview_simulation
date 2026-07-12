
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]


# ── Model Identifiers ─────────────────────────────────────────────────────────
INTERVIEWER_MODEL: str = os.getenv("INTERVIEWER_MODEL", "openai/gpt-oss-120b")
EVALUATOR_MODEL: str = os.getenv("EVALUATOR_MODEL", "openai/gpt-oss-20b")

# ── Tag history cap (guards against unbounded prompt growth) ──────────────────
MAX_TAGS_IN_PROMPT: int = int(os.getenv("MAX_TAGS_IN_PROMPT", "10"))

# ── Singleton clients ─────────────────────────────────────────────────────────
groq_client: Groq = Groq(api_key=GROQ_API_KEY)

