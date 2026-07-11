# AI Interview Simulator

An intelligent interview simulation platform that uses LLMs (via Groq API) to conduct adaptive technical and soft-skill interviews, with real-time difficulty adjustment and comprehensive post-interview evaluations.

## 🎯 Overview

The AI Interview Simulator conducts multi-turn interviews with adaptive difficulty, persistent tag-based history, and AI-powered evaluation. It features:

- **Adaptive Difficulty**: Questions adjust in difficulty based on answer quality
- **Intelligent Question Generation**: Context-aware questions using full session and user history
- **Comprehensive Reviews**: Automated evaluation with strengths, weaknesses, and skill-level assessment
- **Tag-Based History**: Prevents topic repetition and enables skill-targeted follow-ups
- **Session Persistence**: Complete interview history with resumable review generation
- **Retry Resilience**: Graceful degradation when evaluation fails — sessions stay safe for retry

## 🏗️ Architecture

### Core Components

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI application, HTTP endpoints, session lifecycle |
| `interviewer.py` | Question generation service (Groq) with adaptive sequencing |
| `evaluator.py` | Post-interview review generation (Groq) with skill classification |
| `session_store.py` | In-memory session storage, tag indexing, history retrieval |

### API Endpoints

#### Interview Management
- `POST /start` — Create and start a new interview session
- `POST /session/{session_id}/answer` — Submit an answer, get next question
- `POST /session/{session_id}/retry-review` — Re-run evaluation if it failed
- `GET /session/{session_id}` — Fetch session details
- `GET /user/{user_id}/sessions` — List all sessions for a user

#### Debug/Admin
- `GET /debug/tags/{user_id}` — View tag index statistics
- `GET /health` — Liveness check

## 📋 API Examples

### Start an Interview
```bash
curl -X POST http://localhost:8000/start \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "field": "Backend Developer",
    "question_mix": {"technical": 5, "soft": 3},
    "level": "mid"
  }'
```

**Response** (first question):
```json
{
  "session_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "turn_id": 1,
  "question": "Describe your approach to handling database transactions in a high-throughput system.",
  "category": "technical",
  "tags": ["database", "concurrency", "performance"],
  "difficulty": "medium",
  "turns_done": 0,
  "total_questions": 8,
  "status": "in_progress"
}
```

### Submit an Answer
```bash
curl -X POST http://localhost:8000/session/{session_id}/answer \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "answer": "I would use ACID transactions with isolation levels to prevent dirty reads..."
  }'
```

**Response** (if more questions):
```json
{
  "session_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "status": "in_progress",
  "turns_done": 1,
  "total_questions": 8,
  "next_question": { ... }
}
```

**Response** (after final question):
```json
{
  "session_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "status": "complete",
  "turns_done": 8,
  "total_questions": 8,
  "review": {
    "summary": "Strong technical knowledge with solid communication...",
    "strengths": ["Problem decomposition", "System design thinking"],
    "weaknesses": ["Time management under pressure"],
    "skill_level": "senior",
    "level_up_gaps": ["Distributed consensus algorithms", "Advanced caching patterns"],
    "generated_at": "2025-07-10T14:32:15.123456Z"
  }
}
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Groq API key (free tier available at [console.groq.com](https://console.groq.com))
- Docker (optional)

### Local Development

1. **Clone and setup**:
   ```bash
   git clone <repo>
   cd ai-interview-simulator
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   export GROQ_API_KEY="your-groq-api-key-here"
   ```

4. **Run the server**:
   ```bash
   python main.py
   ```

   The API will be available at `http://localhost:8000`
   - OpenAPI docs: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### Docker Setup

1. **Build the image**:
   ```bash
   docker build -t ai-interview-simulator .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 \
     -e GROQ_API_KEY="your-groq-api-key-here" \
     ai-interview-simulator
   ```

3. **With Docker Compose** (if needed):
   ```yaml
   version: '3.8'
   services:
     api:
       build: .
       ports:
         - "8000:8000"
       environment:
         - GROQ_API_KEY=${GROQ_API_KEY}
       restart: unless-stopped
   ```

   ```bash
   docker compose up
   ```

## ⚙️ Configuration

Create a `config.py` file (template included):

```python
import os
from groq import Groq

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY)

# Model selection
INTERVIEWER_MODEL = "openai/gpt-oss-120b"  # Fast, cost-effective
EVALUATOR_MODEL = "openai/gpt-oss-20b"

# Agent tuning
INTERVIEWER_MAX_TOKENS = 1024
AGENT_MAX_ATTEMPTS = 3
AGENT_RETRY_MIN_WAIT_SECONDS = 2
AGENT_RETRY_MAX_WAIT_SECONDS = 15

# Prompt size management
MAX_TAGS_IN_PROMPT = 100  # Cap tag index to prevent prompt bloat
```

### Key Config Options

| Setting | Default | Purpose |
|---------|---------|---------|
| `GROQ_API_KEY` | — | **Required**. Groq API authentication |
| `INTERVIEWER_MODEL` | `openai/gpt-oss-120b` | LLM for question generation |
| `EVALUATOR_MODEL` | `openai/gpt-oss-20b` | LLM for review generation |
| `INTERVIEWER_MAX_TOKENS` | `1024` | Max output tokens per question |
| `MAX_TAGS_IN_PROMPT` | `100` | Tag index size limit (prevents prompt bloat) |
| `AGENT_MAX_ATTEMPTS` | `3` | Retry attempts on transient failures |

## 🔄 Session Lifecycle

```
User starts interview
       ↓
[GET first question from Interviewer service]
       ↓
User sees Q1, submits answer
       ↓
[Question recorded, tags merged into history]
       ↓
[GET next question with updated context]
       ↓
User sees Q2, submits answer
       ↓
... (repeat until all questions answered)
       ↓
[Call Evaluator service with full transcript]
       ↓
Review generated → Session marked complete
       ↓
(If Evaluator fails → Session marked "awaiting_review")
       ↓
User can call /retry-review to re-run evaluation
```

## 📊 Session Data Model

### Session Record
```json
{
  "session_id": "uuid",
  "user_id": "uuid",
  "field": "Backend Developer",
  "level": "mid",
  "question_mix": {"technical": 5, "soft": 3},
  "status": "complete",
  "starting_difficulty": 3,
  "current_difficulty": 4,
  "created_at": "2025-07-10T14:00:00Z",
  "completed_at": "2025-07-10T14:35:00Z",
  "turns": [
    {
      "turn_id": 1,
      "category": "technical",
      "question": "...",
      "tags": ["database", "transactions"],
      "difficulty": "medium",
      "difficulty_delta": 1,
      "answer": "...",
      "answered_at": "2025-07-10T14:02:00Z"
    },
    ...
  ],
  "review": {
    "summary": "...",
    "strengths": [...],
    "weaknesses": [...],
    "skill_level": "senior",
    "level_up_gaps": [...],
    "generated_at": "2025-07-10T14:35:00Z"
  }
}
```

## 🏥 Error Handling & Resilience

### Network Retries
- Transient failures (DNS, timeouts, connection errors) retry up to 3 times with exponential backoff
- Permanent errors (bad auth, validation) fail immediately
- Retry delays: 2s → 4s → 8s → 15s (capped)

### Evaluator Fallback
If the Evaluator fails after retries:
- Session is marked `awaiting_review` (not an error)
- User receives a message: *"Your final answer was saved... call POST /retry-review to try again"*
- Session data is **never lost** — answers are persisted
- Client can retry later via `POST /session/{id}/retry-review`

### Graceful Degradation
- Health check (`/health`) only checks app status, not external APIs
- Missing optional prompt templates don't crash the server
- Invalid LLM output (bad JSON) triggers a single retry before raising

## 📝 Prompt Templates

The system loads prompt templates from:
- `./prompts/interviewer.txt` — Question generation template
- `./prompts/evaluator.txt` — Review generation template

Templates use Python `.format()` placeholders:

**Interviewer template** variables:
```python
{field}, {level}, {category}, {current_difficulty}, {turns_done}, 
{total_questions}, {remaining_technical}, {remaining_soft},
{current_session_tags}, {all_time_tags}, {recent_sessions_json}, 
{difficulty_label}
```

**Evaluator template** variables:
```python
{field}, {level}, {technical_count}, {soft_count}, {total_turns},
{start_difficulty}, {end_difficulty}, {transcript}
```

## 🔐 Security Considerations

- **CORS**: Currently set to `allow_origins=["*"]` — tighten for production
- **API Key**: Never commit `config.py` or `.env` to version control
- **Session IDs**: UUIDs generated server-side (not predictable)
- **User Input**: All answers and questions are validated via Pydantic
- **Error Messages**: Generic messages to clients; detailed errors logged server-side

### Production Checklist
- [ ] Set `GROQ_API_KEY` via secrets manager or environment variable
- [ ] Update CORS `allow_origins` to specific domains
- [ ] Enable HTTPS on the reverse proxy
- [ ] Add rate limiting (per-user, per-IP)
- [ ] Persist sessions to a database (currently in-memory)
- [ ] Enable request logging and monitoring
- [ ] Set `DEBUG=False` in production

## 🧪 Testing

Example test flow:

```python
import requests

BASE_URL = "http://localhost:8000"

# Start
resp = requests.post(f"{BASE_URL}/start", json={
    "user_id": "test-user-1",
    "field": "Backend Developer",
    "question_mix": {"technical": 2, "soft": 1},
    "level": "mid"
})
session_id = resp.json()["session_id"]
question = resp.json()

# Answer
resp = requests.post(f"{BASE_URL}/session/{session_id}/answer", json={
    "session_id": session_id,
    "answer": "I would use a message queue..."
})

# Continue until complete
while resp.json()["status"] == "in_progress":
    question = resp.json()["next_question"]
    resp = requests.post(f"{BASE_URL}/session/{session_id}/answer", json={
        "session_id": session_id,
        "answer": "Another thoughtful answer..."
    })

# View review
review = resp.json()["review"]
print(f"Skill Level: {review['skill_level']}")
print(f"Strengths: {', '.join(review['strengths'])}")
```

## 📊 Logging

The server logs at `INFO` level by default with ISO 8601 timestamps:

```
2025-07-10T14:05:32.123Z [INFO] main: Session xxxxxxxx-xxxx started for user xxxxxxxx-xxxx
2025-07-10T14:05:35.456Z [INFO] interviewer: Interviewer → q='Describe your approach...' tags=['database', 'transactions'] diff=medium delta=1
2025-07-10T14:05:40.789Z [INFO] session_store: Answer recorded for session xxxxxxxx-xxxx turn 1/8
2025-07-10T14:35:50.012Z [INFO] evaluator: Evaluator → skill_level=senior, strengths=3, weaknesses=2, gaps=4
```

Adjust logging level in `main.py`:
```python
logging.basicConfig(level=logging.DEBUG)  # For verbose output
```

## 🐳 Docker Details

### Image Spec
- **Base**: `python:3.11-slim`
- **Size**: ~150 MB (optimized multi-stage build)
- **Port**: 8000 (exposed)
- **User**: `appuser` (non-root, for security)

### Environment Variables
| Variable | Required | Default |
|----------|----------|---------|
| `GROQ_API_KEY` | ✓ Yes | — |
| `LOG_LEVEL` | No | `INFO` |
| `MAX_TAGS_IN_PROMPT` | No | `100` |

### Build & Run
```bash
# Development (with live reload)
docker build -t ai-interview-simulator:dev -f Dockerfile.dev .
docker run -v $(pwd):/app -p 8000:8000 ai-interview-simulator:dev

# Production
docker build -t ai-interview-simulator:latest .
docker run -e GROQ_API_KEY=$GROQ_API_KEY -p 8000:8000 ai-interview-simulator:latest
```

### Health Check
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | ≥0.100 | Web framework |
| uvicorn | ≥0.23 | ASGI server |
| pydantic | ≥2.0 | Data validation |
| groq | ≥0.4.0 | Groq API client |
| requests | ≥2.31 | HTTP utilities |

See `requirements.txt` for full list.

## 🚀 Deployment

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-interview-simulator
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: api
        image: ai-interview-simulator:latest
        ports:
        - containerPort: 8000
        env:
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: groq-secret
              key: api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### AWS Lambda (with API Gateway)
```python
# handler.py
from mangum import Mangum
from main import app

handler = Mangum(app)
```

```bash
pip install -r requirements-lambda.txt
zip -r function.zip . -x "venv/*"
aws lambda create-function --function-name ai-interview-simulator \
  --runtime python3.11 --handler handler.handler \
  --zip-file fileb://function.zip
```

## 📚 References

- [FastAPI Docs](https://fastapi.tiangolo.com)
- [Groq API Guide](https://console.groq.com/docs)
- [Pydantic Docs](https://docs.pydantic.dev)

## 📄 License

MIT License — See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### "Evaluator returned malformed output"
- Check `EVALUATOR_MODEL` is set correctly
- Verify `prompts/evaluator.txt` exists and is valid
- Check Groq API key has access to the selected model

### "Session not found"
- Session IDs are case-sensitive UUIDs
- Sessions expire only when the app restarts (in-memory storage)
- Verify the `session_id` matches what was returned by `/start`

### "Timeout on Interviewer call"
- Groq API may be slow; increase `AGENT_RETRY_MAX_WAIT_SECONDS`
- Check network connectivity to `api.groq.com`
- Verify `GROQ_API_KEY` is valid (rate limits apply to free tier)

### "CORS errors in browser"
- Update `allow_origins` in `main.py` to include your frontend URL
- For local dev, set to `["http://localhost:3000"]`

---

**Version**: 1.2.0  
**Last Updated**: July 2025
