# Clinic Voice Agent

A self-hosted voice assistant that answers clinic questions and books, reschedules and cancels appointments by voice. It runs on open-source models (Qwen3, Whisper, Ming-omni-tts), PostgreSQL + pgvector, a Redis/arq job queue, and syncs to Google Calendar and Gmail.

> 🎥 **Demo video:** _link here_

---

## Features

- **Real-time voice over WebSocket.** The browser streams microphone audio, Silero VAD detects end of speech, and the reply streams back sentence by sentence.
- **LLM tool calling.** Qwen3 decides when to search clinic info, check availability, or book/reschedule/cancel.
- **Server-enforced confirmation.** Booking, rescheduling and cancelling only happen after the patient explicitly says "yes" to a read-back summary.
- **No double-booking.** PostgreSQL is the source of truth. A partial unique index and a short Redis slot hold prevent two patients getting the same slot.
- **Async side effects.** Google Calendar events and Gmail confirmations run in an arq worker with retries, so a Google outage never breaks a booking.
- **RAG over clinic documents.** Answers questions about fees, timings and doctors from your own documents, using pgvector.
- **Two hardware profiles, one codebase.** `dev` runs on a 6 GB laptop GPU and `gpu` runs the full models on a 24 GB card. Only `.env` changes.
- **Observability.** structlog JSON logs, Prometheus metrics with per-stage latency (STT, LLM, TTS), Grafana dashboards, and optional Langfuse traces.

---

## Architecture

```mermaid
flowchart LR
    B[Browser<br/>mic + speaker] <-- WebSocket --> N[Nginx]
    N <--> API[FastAPI api<br/>VAD · orchestrator · tools]
    API --> STT[STT service<br/>faster-whisper]
    API --> LLM[LLM<br/>vLLM / Ollama]
    API --> TTS[TTS service<br/>Ming / Piper]
    API <--> PG[(PostgreSQL<br/>+ pgvector)]
    API <--> R[(Redis)]
    R --> W[arq worker]
    W --> GC[Google Calendar]
    W --> GM[Gmail]
    W <--> PG
```

One voice turn works like this:

1. The patient speaks.
2. VAD detects the end of speech.
3. Whisper transcribes the segment.
4. Qwen3 reads the transcript and calls tools if needed.
5. The tools run against Postgres and RAG.
6. Qwen3 streams the reply.
7. Each sentence goes to TTS.
8. The audio streams back to the browser.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design: folder structure, WebSocket protocol, tools, data model, containers and design decisions.

---

## Tech stack

| Layer | Technology |
| --- | --- |
| Backend | Python 3.11, FastAPI, Pydantic v2, pydantic-settings |
| LLM | Qwen3-8B-AWQ on vLLM (`gpu`) · `qwen3:4b` on Ollama (`dev`) |
| Speech-to-text | faster-whisper: `large-v3` (`gpu`) · `small` int8 on CPU (`dev`) |
| Voice activity detection | Silero VAD |
| Text-to-speech | inclusionAI/Ming-omni-tts-0.5B on vLLM-Omni (`gpu`) · Piper (`dev`) |
| Embeddings | BGE-small-en-v1.5 (384-dim) |
| Database | PostgreSQL 17 + pgvector, SQLAlchemy 2.0 (async), asyncpg, Alembic |
| Queue | Redis 7 + arq |
| Integrations | Google Calendar API, Gmail API (OAuth) |
| Observability | structlog, Prometheus, Grafana, Langfuse (optional) |
| Frontend | HTML, CSS, JavaScript, AudioWorklet |
| Infra | Docker, Docker Compose, Nginx |
| Testing | pytest, pytest-asyncio, httpx, testcontainers |

---

## Hardware profiles

| | `dev` profile | `gpu` profile |
| --- | --- | --- |
| Target | Laptop with a 6 GB GPU (e.g. RTX 4050) and 16 GB RAM | Single 24 GB GPU (RTX 4090 / A5000 / L4, local or rented) |
| LLM | `qwen3:4b` on Ollama, running natively on the host | Qwen3-8B-AWQ on vLLM (container) |
| STT | faster-whisper `small`, int8, CPU | faster-whisper `large-v3`, CUDA |
| TTS | Piper (CPU) | Ming-omni-tts-0.5B on vLLM-Omni |
| Typical turn latency | A few seconds | ~1–2 s to first audio (target) |

The `dev` profile is for building and checking that everything works. Judge conversation quality and record the demo on the `gpu` profile.

---

## Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine + Compose v2 (Linux)
- Windows: WSL2 enabled. Cap its memory in `C:\Users\<you>\.wslconfig`:
  ```ini
  [wsl2]
  memory=8GB
  ```
- `dev` profile: [Ollama](https://ollama.com) installed on the host
- `gpu` profile: NVIDIA driver + NVIDIA Container Toolkit
- A Google account to act as the clinic account (Calendar + Gmail)

---

## Quick start: `dev` profile (laptop)

```bash
# 1. Clone
git clone https://github.com/<you>/clinic-voice-agent.git
cd clinic-voice-agent

# 2. Pull the local LLM (Ollama runs natively and uses your GPU)
ollama pull qwen3:4b

# 3. Configure
cp .env.dev.example .env
#    edit .env: POSTGRES_PASSWORD, CLINIC_NAME, CLINIC_EMAIL, etc.

# 4. Google OAuth (one time, see "Google setup" below)
pip install google-auth-oauthlib
python scripts/google_auth.py        # writes secrets/google_token.json

# 5. Start
docker compose --profile dev up -d --build

# 6. Database + seed data + knowledge base
docker compose exec api alembic upgrade head
docker compose exec api python -m scripts.seed
docker compose exec api python -m scripts.ingest_documents
```

Open **http://localhost**, click the mic and say something like *"I'd like to see a dermatologist tomorrow evening."*

The microphone works on `http://localhost` without HTTPS. Any other address needs HTTPS.

---

## Quick start: `gpu` profile (24 GB GPU or rented pod)

On a rented GPU (RunPod, Vast.ai, etc.), use a CUDA + Docker template and put the Hugging Face cache on a persistent volume so models aren't re-downloaded every session.

```bash
git clone https://github.com/<you>/clinic-voice-agent.git
cd clinic-voice-agent

cp .env.gpu.example .env
#    edit .env: HF_TOKEN (if needed), POSTGRES_PASSWORD, PUBLIC_BASE_URL

./scripts/download_models.sh         # Qwen3-8B-AWQ, whisper large-v3, Ming, BGE

# copy secrets/google_token.json from your laptop (never commit it)

docker compose --profile gpu --profile obs up -d --build
docker compose exec api alembic upgrade head
docker compose exec api python -m scripts.seed
docker compose exec api python -m scripts.ingest_documents
```

**Expose it over HTTPS.** The browser blocks the mic on plain `http://<ip>`. Use the provider's HTTPS proxy URL or a Cloudflare Tunnel:

```bash
cloudflared tunnel --url http://localhost:80
```

**Stop the pod when you're done.** It bills by the hour.

---

## Google setup (Calendar + Gmail)

1. Create a project in [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **Google Calendar API** and the **Gmail API**.
3. Configure the OAuth consent screen with user type **External**, publishing status **Testing**, and add the clinic Gmail account as a **test user**.
4. Create an OAuth client ID of type **Desktop app** and download it to `secrets/google_client.json`.
5. Run `python scripts/google_auth.py` and sign in as the clinic account. It saves `secrets/google_token.json` with the scopes:
   - `https://www.googleapis.com/auth/calendar.events`
   - `https://www.googleapis.com/auth/gmail.send`
6. In Google Calendar, create one calendar per doctor and put each calendar ID in `backend/scripts/seed_data/doctors.yaml`.

`secrets/` is git-ignored. Tokens in "Testing" mode expire after 7 days, so re-run step 5 if the worker logs `invalid_grant`.

---

## Using the demo

| Try saying | What happens |
| --- | --- |
| "What are your clinic timings?" | RAG answer from `clinic_info.txt` |
| "Which doctors do you have for skin problems?" | `list_doctors` |
| "Book me with Dr. Sen on Friday morning" | `check_availability` → bot offers slots → collects name, phone and email → reads back a summary → you say "yes" → `confirm_action` |
| "I want to move my appointment" | Asks for booking reference + phone → `find_appointment` → `reschedule_appointment` → confirm |
| "Cancel my booking" | Verifies booking reference + phone → `cancel_appointment` → confirm |

After a confirmed booking, check that the event appears in the doctor's Google Calendar and the confirmation email arrives in the patient's inbox (use your own email for testing).

Email addresses are hard to dictate, so the bot asks you to type the email into the on-screen field.

---

## Running tests

```bash
# Unit tests: no GPU, no network (FakeLLM + fake Google clients)
docker compose exec api pytest tests/unit

# Integration tests: real Postgres + Redis via testcontainers (run on the host, needs Docker)
cd backend && pip install -r requirements-dev.txt
pytest tests/integration

# Concurrency test: 20 parallel bookings for one slot, exactly 1 must succeed
pytest tests/integration/test_double_booking.py

# Conversation evals: scripted scenarios against the real LLM, asserting DB state
docker compose exec api python -m tests.evals.run --scenarios tests/evals/scenarios.yaml
```

---

## Observability

| Tool | URL (with `--profile obs`) | What to look at |
| --- | --- | --- |
| Grafana | http://localhost:3000 | "Voice turn latency" dashboard: STT, LLM first token, TTS first audio, total |
| Prometheus | http://localhost:9090 | Raw metrics: `voice_stage_latency_seconds`, `tool_calls_total`, `bookings_total` |
| Logs | `docker compose logs -f api worker` | JSON logs with `session_id` and `turn_id` |
| Langfuse (optional) | Langfuse Cloud or your own Langfuse deployment | Per-turn trace: prompt → tool calls → response |

Langfuse self-hosting needs several extra containers (it's too heavy for a 16 GB laptop). Use Langfuse Cloud's free tier, or self-host it only on the GPU box. Leave `LANGFUSE_*` empty to disable it.

---

## Configuration

Key variables (see `.env.dev.example` / `.env.gpu.example` for the full list):

| Variable | `dev` example | `gpu` example |
| --- | --- | --- |
| `LLM_BASE_URL` | `http://host.docker.internal:11434/v1` | `http://llm:8000/v1` |
| `LLM_MODEL` | `qwen3:4b` | `Qwen/Qwen3-8B-AWQ` |
| `STT_BASE_URL` | `http://stt-cpu:8003` | `http://stt-gpu:8003` |
| `STT_MODEL` / `STT_DEVICE` | `small` / `cpu` | `large-v3` / `cuda` |
| `TTS_BASE_URL` | `http://tts-piper:8002` | `http://tts-ming:8002` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | same |
| `DATABASE_URL` | `postgresql+asyncpg://clinic:***@postgres:5432/clinic` | same |
| `REDIS_URL` | `redis://redis:6379/0` | same |
| `CLINIC_TIMEZONE` | `Asia/Kolkata` | same |
| `STORE_AUDIO` | `false` | `false` |
| `LANGFUSE_PUBLIC_KEY` / `_SECRET_KEY` / `_HOST` | empty (disabled) | optional |

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Mic button does nothing | You're not on `localhost` or HTTPS. Use a tunnel or HTTPS URL. |
| `api` can't reach Ollama | Make sure Ollama is running on the host. On Linux, the compose file maps `host.docker.internal` via `extra_hosts: host-gateway`. |
| vLLM out of memory | Lower `--gpu-memory-utilization` or `--max-model-len` in `docker-compose.yml`. The LLM, STT and TTS share one GPU. |
| Bot "thinks" for a long time | Qwen3 thinking mode is on. It's disabled via `enable_thinking: false`. Check `LLM_DISABLE_THINKING=true`. |
| Booking OK but no Calendar event | Check `docker compose logs worker`. The appointment stays `calendar_sync_status=pending` and the retry sweep picks it up. |
| `invalid_grant` from Google | The OAuth token expired (Testing mode). Re-run `scripts/google_auth.py`. |

---

## Roadmap

- [ ] **M1:** Text chat agent with tools, booking logic, Postgres, Alembic, unit tests
- [ ] **M2:** arq worker, Google Calendar + Gmail sync, retry sweep, concurrency test
- [ ] **M3:** WebSocket voice pipeline on the `dev` profile (VAD, STT, Piper TTS)
- [ ] **M4:** RAG over clinic documents (pgvector)
- [ ] **M5:** `gpu` profile (vLLM, Whisper large-v3, Ming), Grafana dashboards, demo video
- [ ] Later: barge-in (interrupt the bot while it's speaking), Hindi/Bengali support, outbox table

---

## License

MIT
