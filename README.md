# Autonomous Multi-Vendor E-Commerce Dispute Resolution AI

> **Multi-Vendor Dispute Resolution System** — React.js + FastAPI + Hybrid GraphRAG (Neo4j + Qdrant) + LangGraph Multi-Agent Framework + LiveKit Voice AI + Langfuse Observability.

---

## Recent Changes

### Voice Pipeline — Production-Grade Upgrade

The voice path moved from a prototype to a first-class feature with measured sub-2-second latency, integrity-preserving memory on barge-in, and full-stack observability.

| Feature | Impact |
|---|---|
| **Voice fast path** (`achat_stream_fast`) | Bypasses the multi-agent graph for voice — single streaming LLM call. Drops perceived latency from ~32 s → ~1.5 s. |
| **Real token streaming** (`streaming=True` on the fast LLM) | First-token latency 2247 ms → 379 ms. Real streaming, not client-side chunking. |
| **Memory + barge-in integrity** | If the user interrupts mid-sentence, the *partial* answer (what was actually spoken) is saved to memory tagged `[interrupted]`. Long-term distillation skips interrupted turns. The agent's recollection matches the user's experience. |
| **Reactive UI bubble** (`VoiceBubble.tsx` + `VoiceRoom.tsx`) | WebAudio-driven SVG that pulses with mic + agent audio. Five states: idle / listening / thinking / speaking / error. Latency HUD beneath. |
| **LiveKit token endpoint** (`/voice/token`) | Browser-callable JWT minter so the UI joins LiveKit rooms without exposing the API secret. |
| **Sidebar split — Voice / Chat** | The session list now has two halves: voice calls on top, text chats below. Client-side partition by `voice-` prefix on `session_id`. |
| **Auto-generated session titles** | After 4 turns, a fast Groq LLM summarises the conversation into a 3–6 word title. Replaces `"Conversation 2026-05-17 12:30"` with something like `"Vendor Refund Dispute Resolution"`. Costs ~$0.00001 per session. |
| **Latency observability triangle** | Per-turn timings visible in **three places**: worker log, Langfuse `voice_pipeline` span with metadata, browser HUD via LiveKit data channel. |
| **Tuned VAD endpointing** | `silence_threshold_ms`: 500 → 300, `min_endpointing_delay`: 0.5 → 0.3. 400 ms saved per turn. Yaml + dataclass defaults now in sync. |
| **Worker pre-warming** (`warm_start`) | One tiny `llm_fast.ainvoke("hi")` at boot primes the HTTPS pool so call #1 doesn't pay TLS handshake cost. |
| **Langfuse v3+/v2 import fallback** | Tracing works whether your env has langfuse v2 or v4. |

---

### Deploy to AWS + CI/CD

| Feature | Impact |
|---|---|
| **AWS Copilot → ECS Fargate** | Three services (`api`, `worker`, `voice`) on serverless Graviton/arm64 containers behind an Application Load Balancer. No servers to manage. |
| **ElastiCache Redis + Arq worker** | Managed Redis as the broker; the Arq worker drains slow jobs (memory distillation, auto-titling) off the request path. |
| **SPA served from the API** | The React UI is built into the `api` image and served at `/`; FastAPI strips the `/api` prefix so one ALB hostname serves both the app and the API — no CORS, no second host. |
| **SSM Parameter Store secrets** | 21 secrets stored as Copilot-managed SecureStrings; the ECS task role is auto-granted read + decrypt. Nothing sensitive in the image or Git. |
| **On-demand voice** | The voice worker is kept at 0 tasks and scaled to 1 only for a call/demo — bursty cost instead of 24×7. |
| **GitHub Actions CI/CD (OIDC)** | Push to `dev` → GitHub authenticates via OIDC (no stored keys), builds the arm64 image, pushes to ECR, and force-rolls the `api` + `worker` ECS services until stable. See `.github/workflows/deploy.yml`. |
| **Cost guardrails** | AWS Budgets emails an alert every $20. Full cost breakdown, free-tier/credit analysis, and pause/teardown commands in `docs/DisputeAI_Cloud_Services_Cost_and_Decisions.docx`. |

### Deployment topology

```mermaid
flowchart TB
    Dev(["git push → dev"]) --> GHA["GitHub Actions · OIDC"]
    GHA --> ECR["Amazon ECR"]
    User(["Browser"]) --> ALB

    subgraph AWS["AWS · us-west-2"]
        ALB["Application Load Balancer"] --> API["Fargate · api ×2<br/>SPA + FastAPI"]
        WK["Fargate · worker · Arq"]
        VO["Fargate · voice · on-demand"]
        API --> REDIS[("ElastiCache Redis")]
        WK --> REDIS
        SSM[("SSM Parameter Store · secrets")] -.-> API
        SSM -.-> WK
        SSM -.-> VO
    end

    ECR --> API
    ECR --> WK
    ECR --> VO
    API --> EXT[("Supabase + Qdrant")]
    VO --> LKC["LiveKit Cloud"]

    classDef store fill:#eef3fb,stroke:#2E5496,color:#1F3864;
    class REDIS,SSM,EXT store;
```

Pre-setup (account, IAM user, CLI) is in `docs/AWS_From_Zero_Account_and_IAM_Setup.docx`. Deployment commands are in the **Deploying to AWS** section below.

---

## Architecture

Two entry surfaces, one orchestrator core.

```mermaid
flowchart TB
    subgraph TEXT["TEXT PATH"]
        B1["Browser"] --> API["FastAPI /chat"]
        API --> DG["decision_graph"]
        DG --> GR["Guardrail · Llama"]
        GR --> CAG["CAG cache · Qdrant"]
        CAG --> ACH["achat()"]
    end

    subgraph VOICE["VOICE PATH"]
        VB["Browser · React"] --> TOK["POST /voice/token → JWT"]
        TOK --> LK["LiveKit Cloud"]
        LK --> VW["Voice Worker"]
        VAD["Silero VAD"] --> STT["Deepgram STT"]
        STT --> ADP["LangGraphLLMAdapter"]
        VW --> ADP
        ADP --> FAST["achat_stream_fast()"]
        FAST --> GROQ["Groq llama-3.3-70b · real streaming"]
        GROQ --> TTS["ElevenLabs TTS → audio"]
        FAST --> BG["_save_voice_turn_async"]
        BG --> HUD["latency HUD → browser bubble"]
    end

    subgraph CORE["AgentOrchestrator · LangGraph"]
        ORCH["recall → supervisor → fan-out"]
        ORCH --> A1["admin"]
        ORCH --> A2["dispute"]
        ORCH --> A3["direct"]
        ORCH --> A4["web"]
        A1 --> MERGE["merge_responses → save_memory"]
        A2 --> MERGE
        A3 --> MERGE
        A4 --> MERGE
    end

    ACH --> ORCH
    MERGE --> MEM[("4-tier Memory · ST / LT / EP / Pr<br/>Supabase + pgvector + Qdrant")]
    BG --> MEM
    MEM --> LF[("Langfuse · per-turn trace + Sessions")]

    classDef store fill:#eef3fb,stroke:#2E5496,color:#1F3864;
    class MEM,LF store;
```

**Key boundary:** `src/voice/` and `ui/src/components/Voice*.tsx` are a self-contained vertical slice. The voice fast path is the only orchestrator addition (`achat_stream_fast`); the multi-agent text graph is untouched. Removing the voice layer leaves the text-path fully functional.

### Voice pipeline internals

```mermaid
flowchart TB
    AG["LiveKit Agent"] --> VAD["VAD · Silero<br/>activation 0.5 · min_silence 0.3s"]
    AG --> STT["STT · Deepgram nova-3 · streaming"]
    AG --> LLM["LLM · LangGraphLLMAdapter"]
    AG --> TTS["TTS · ElevenLabs turbo v2.5 · streaming"]
    AG --> OPT["allow_interruptions=True<br/>min_endpointing_delay=0.3s"]

    LLM --> RUN["LangGraphLLMStream._run()"]
    RUN --> SPAN["open Langfuse 'voice_pipeline' span"]
    RUN --> LOOP{"async for kind, payload<br/>in achat_stream_fast()"}
    LOOP -->|token| T1["emit ChatChunk → TTS"]
    LOOP -->|partial| T2["remember spoken · barge-in"]
    LOOP -->|final| T3["AgentResponse + metadata"]
    RUN --> CANCEL["on CancelledError · barge-in:<br/>save [interrupted] partial · span barge_in=true · re-raise"]

    classDef warn fill:#fdecea,stroke:#C00000,color:#7a1010;
    class CANCEL warn;
```

**EOU policy** is three-layered:
1. VAD (`vad_threshold=0.5`) — *"is this speech right now?"*
2. Silence persistence (`silence_threshold_ms=300`)
3. Confirmation buffer (`min_endpointing_delay=0.3`)

Perceived endpoint = layer 2 + layer 3 = **600 ms**.

### Latency budget (measured)

```mermaid
flowchart TB
    U(["User stops speaking"]) -->|300 ms| V["VAD endpointing"]
    V -->|200 ms| S["Deepgram STT · streaming"]
    S -->|"<150 ms"| O["Orchestrator pre-LLM<br/>time-boxed memory fetch · off-thread"]
    O -->|"200–400 ms"| G["Groq llama-3.3-70b · first token"]
    G -->|200 ms| E["ElevenLabs TTS · first byte"]
    E -->|200 ms| N["Network · browser ↔ region"]
    N --> R(["≈ 1.5 s perceived"])

    classDef goal fill:#e4ecda,stroke:#548235,color:#2c4214;
    class R goal;
```

### MCP integration layer

```mermaid
flowchart TB
    O["orchestrator.py · build_agent_mcp()"] --> C["MCP Client Layer<br/>langchain-mcp-adapters"]
    C --> S1["disputeai-crm · custom"]
    C --> S2["disputeai-memory · custom"]
    C --> S3["postgres MCP · off-the-shelf"]
    S1 --> CRM["CRMTool → Supabase"]
    S2 --> MO["MemoryOps → pgvector"]
    S3 --> PG["Supabase · raw SQL"]
```

Three MCP servers, three origins, one agent. The text path uses MCP-backed tools; the voice fast path skips MCP for latency reasons and calls Groq directly.

---

## Project Structure

```
Multi-Vendor Dispute AI/
│
├── src/
│   ├── voice/                                    # ← Voice side-car
│   │   ├── __init__.py
│   │   ├── config.py                             # VAD defaults 300/0.3
│   │   ├── stt.py                                # make_stt — Deepgram nova-3
│   │   ├── tts.py                                # make_tts — ElevenLabs / Deepgram
│   │   ├── adapter.py                            # LangGraphLLMStream._run rewritten
│   │   │                                         # (token streaming + barge-in + Langfuse span)
│   │   ├── pipeline.py                           # VoiceSession + SessionManager + event helpers
│   │   ├── agent.py                              # warm_start wired, latency data-channel publish
│   │   └── run.py                                # initialize_process_timeout=60s
│   │
│   ├── agents/
│   │   ├── orchestrator.py                       # achat_stream_fast + _save_voice_turn_async
│   │   ├── decision_graph.py                     # Text-path guardrail + CAG short-circuit
│   │   ├── guardrail.py
│   │   ├── router.py
│   │   ├── state.py
│   │   ├── prompts/agent_prompts.py
│   │   └── tools/{crm_tool.py, rag_tool.py, web_search_tool.py}
│   │
│   ├── mcp_servers/                              # CRM, memory, RAG, web, CAG, crawler MCP servers
│   │
│   ├── api/
│   │   ├── main.py                               # voice_router registered
│   │   ├── schemas.py
│   │   └── routers/
│   │       ├── chat.py                           # maybe_auto_title_sync at save sites
│   │       ├── chat_sessions.py                  # _is_default_title + maybe_auto_title_sync
│   │       ├── voice.py                          # /voice/token JWT endpoint
│   │       ├── health.py, customers.py
│   │       └── tools/{cag,crawl,crm,memory,rag,web}.py
│   │
│   ├── memory/                                   # 4-tier memory
│   │   ├── st_store.py, lt_store.py
│   │   ├── episodic_store.py, procedural_store.py
│   │   ├── memory_ops.py                         # MemoryDistiller + MemoryRecaller
│   │   ├── schemas.py, prompts.py
│   │
│   ├── services/{chat_service, crm_service, ingest_service}/
│   │
│   ├── workers/                                  # Arq background worker
│   │   ├── tasks.py                              #   WorkerSettings: save_chat_turn, auto_title, distill
│   │   └── enqueue.py                            #   ARQ_WORKER_ENABLED-gated job enqueue
│   │
│   └── infrastructure/
│       ├── config.py
│       ├── observability.py                      # v3+/v2 langfuse import fallback
│       ├── llm/
│       │   └── llm_provider.py                   # get_fast_chat_llm defaults streaming=True
│       ├── db/
│       └── log.py
│
├── ui/                                           # React + Vite + Tailwind + Framer Motion
│   └── src/
│       ├── App.tsx                               # Voice button + modal + sidebar-refresh hooks
│       ├── components/
│       │   ├── VoiceBubble.tsx                   # reactive SVG blob (5 states)
│       │   ├── VoiceRoom.tsx                     # LiveKit + WebAudio analysers
│       │   ├── Sidebar.tsx                       # split into Voice (top) + Chat (bottom)
│       │   ├── ChatWindow.tsx, InputBox.tsx, MessageBubble.tsx, …
│       │   └── …
│       └── hooks/{useChat, useChatStream, useSessions, useHealth, useCustomer}.ts
│
├── docker/
│   ├── api/Dockerfile                            # Node stage bundles the SPA into the image
│   ├── web/Dockerfile                            # nginx + built React (local compose only)
│   └── voice/Dockerfile                         # LiveKit voice worker
│
├── copilot/                                      # AWS Copilot manifests
│   ├── environments/dev/manifest.yml             #   VPC + ALB + ECS cluster
│   ├── api/manifest.yml                          #   Load Balanced Web Service (2 tasks, arm64)
│   ├── worker/manifest.yml                       #   Backend Service (Arq worker)
│   └── voice/manifest.yml                        #   Backend Service (on-demand, count 0)
│
├── .github/workflows/
│   └── deploy.yml                                # OIDC CI/CD — push to dev → AWS
│
├── scripts/
│   ├── seed_crm_unified.py, ingest_to_qdrant.py
│   ├── seed_procedures.py, rebuild_cag_cache.py
│   ├── init_supabase.py
│   └── aws/                                      # build_push_images.sh, deploy_redis.sh,
│       └── …                                     # push_secrets.sh, cfn/redis-cluster.yml
│
├── config/
│   └── param.yaml                                # VAD 300/0.3 defaults (yaml ↔ dataclass)
│
├── docs/
│   ├── AWS_From_Zero_Account_and_IAM_Setup.{docx,pdf}
│   └── DisputeAI_Cloud_Services_Cost_and_Decisions.{docx,pdf}
│
├── README.md                                     # ← this file
├── Makefile                                      # demo / voice / voice-test / voice-logs / …
├── docker-compose.yml                            # api + web (default), voice (profile), + redis + worker
├── compose.prod.yml                              # 2 api replicas + worker + voice + redis + web
├── pyproject.toml                                # Source of truth for dependencies
├── requirements.txt                              # Lock-step with pyproject.toml
├── .env.example                                  # Template — includes voice section
└── uv.lock
```

---

## Quick Start — Three Commands

```bash
# Terminal 1 — API + Web (Docker)
make demo

# Terminal 2 — Voice worker (foreground)
make voice

# Terminal 3 — UI (hot-reload dev server)
cd ui && npm install && npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173` or `5174`) in **Chrome** (Safari has WebAudio quirks). Log in as a customer, click **Voice** in the top bar, click **Start Call**.

**Readiness signals:**
- API ready when log says `Application startup complete` (~60 s — lifespan startup is gated on Supabase + Qdrant + MCP + CAG warm-up).
- Voice worker ready when log says **`LLM connection warm — first call took XXX ms`** AND `registered worker`.
- UI ready when Vite prints the local URL.

If you hit `ModuleNotFoundError: No module named 'langchain_core'` running uvicorn natively, your shell's `python` resolves to system Python, not your venv. Use:
```bash
PYTHONPATH=src python -m uvicorn api.main:app --reload --port 8000
```

---

## Compose targets

| Command | Brings up | Containers |
|---|---|---|
| `make demo` | Default text stack | `api` + `web` |
| `make demo-voice` | Text stack + voice worker | `api` + `web` + `voice` |
| `make voice` | Voice worker only (native, foreground) | — |
| `make voice-test` | Validate voice config + env vars | — |
| `make demo-down` | Stop default stack | — |
| `make demo-voice-down` | Stop voice profile | — |
| `make demo-logs` / `make voice-logs` | Tail logs | — |

The voice worker has no exposed port — it dials outbound to `LIVEKIT_URL` and registers as a worker. Use `make voice-logs` to confirm registration.

**Try these to see each route light up:**

| Channel | Query | What it exercises | Latency |
|---|---|---|---|
| Text | `What is your dispute policy?` | CAG cache → FAQ hit | ~290 ms |
| Text | `Show my active dispute for order #1234` | CRM → Supabase customer lookup | ~3–5 s |
| Voice | `I want to raise a dispute for my order` | Voice fast path | ~1.5 s |
| Voice | `Tell me about the refund process` *(interrupt mid-sentence)* | Barge-in + partial-answer memory | ~400 ms to silence |

---

## Voice configuration

`config/param.yaml`:

```yaml
voice:
  stt_provider: deepgram
  stt_model: nova-3
  stt_language: en

  tts_provider: elevenlabs            # or "deepgram"
  tts_model: eleven_turbo_v2_5
  tts_voice_id: l7kNoIfnJKPg7779LI2t  # ElevenLabs "Aria"

  # ── VAD + EOU policy ────────────────────────────────────────
  # Combined endpoint = silence_threshold_ms + min_endpointing_delay
  # 300 ms + 300 ms feels responsive without false interruptions.
  vad_threshold: 0.5
  silence_threshold_ms: 300           # was 500 — saved 200 ms per turn
  min_endpointing_delay: 0.3          # was 0.5 — saved 200 ms per turn

  interruption_enabled: true
  sample_rate: 16000
```

**Tuning knobs and their feel:**

| Knob | Lower | Higher |
|---|---|---|
| `vad_threshold` | More false positives (background noise becomes "speech") | Soft speakers get cut off |
| `silence_threshold_ms` | Snappy but interrupts people who pause | Polite but feels laggy |
| `min_endpointing_delay` | Fast turn-around | Better tolerates last-syllable trail-off |

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Send message, get reply (decision_graph → orchestrator). Background-schedules `touch_session` + `maybe_auto_title`. |
| `POST` | `/chat/stream` | SSE — node-by-node state updates. |
| `POST` | `/voice/token` | Mint a short-lived LiveKit JWT (10-min TTL) for the browser. |
| `GET` | `/chat_sessions?user_id=…` | List a customer's sessions for the sidebar. Voice + chat sessions in one list, partitioned client-side by `voice-` prefix. |
| `POST` | `/chat_sessions` | Create a new chat session. |
| `PATCH` | `/chat_sessions/{id}` | Rename or archive. |
| `DELETE` | `/chat_sessions/{id}` | Hard-delete (cascades ST turns). |
| `GET` | `/sessions/{id}/turns` | Fetch ST turn history for a session. |
| `GET` | `/health` | Liveness check + tool availability. |
| `GET` | `/graph` | LangGraph topology (Mermaid + structured). |
| `GET` | `/memory/{user_id}` | Inspect long-term semantic facts. |

### Examples

```bash
# Text chat — dispute query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_message": "What vendors are available?", "user_id": "94781030736", "session_id": "demo"}'

# Mint a voice token (the browser calls this)
curl -X POST http://localhost:8000/voice/token \
  -H "Content-Type: application/json" \
  -d '{"user_id": "94781030736"}'
# → {"url":"wss://...", "token":"eyJ...", "room":"voice-xxx", "identity":"94781030736"}

# List sessions for the sidebar
curl "http://localhost:8000/chat_sessions?user_id=94781030736"
```

The voice worker itself has no HTTP surface — it talks LiveKit's room protocol over WebRTC.

---

## Observability

Every `.chat()`, `.achat()`, and voice turn produces a Langfuse trace.

**Text path:**
```mermaid
flowchart TB
    T["trace: agent_chat"] --> R["node_recall · ST + LT retrieval"]
    T --> S["node_supervisor · router LLM"]
    T --> A["node_[agent] · tool call + synthesis"]
    T --> M["node_save_memory · ST store + LT distillation"]
```

**Voice path:**
```mermaid
flowchart TB
    T["trace: voice_turn"] --> SP["span: voice_pipeline<br/>user_id · session_id · tags[voice, fast_path]"]
    SP --> M1["first_token_ms"]
    SP --> M2["llm_total_ms"]
    SP --> M3["agent_total_ms"]
    SP --> M4["chunks"]
    SP --> M5["barge_in · true if interrupted"]
    SP --> GEN["generation · ChatGroq llama-3.3-70b<br/>input · output · token usage · cost"]
```

All turns of one call share `session_id="voice-<room>"` → grouped in the Langfuse **Sessions** view. Filter by `tags:["voice","fast_path"]` for the voice subset.

Worker log shows per-turn timings in real time:
```
⏱  achat_stream_fast: mem=87ms, prompt=1ms, pre_llm_total=92ms, first_llm_chunk=287ms
📊 Voice turn timings: first_token=326ms, llm_total=510ms, agent_total=623ms, chunks=8, route=voice_fast
```

The same numbers flow to the browser HUD via the LiveKit data channel.

---

## MCP integration details

```python
# orchestrator.py — build_agent_mcp()

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp_servers.mcp_config import build_mcp_server_config

mcp_client = MultiServerMCPClient(build_mcp_server_config())
tools = await mcp_client.get_tools()    # discovers tools from 3 servers
crm_tool = _MCPCRMToolAdapter(tools)    # same dispatch() interface
```

### MCP server architecture (stdio transport)

```mermaid
sequenceDiagram
    participant H as Host · LangGraph agent
    participant M as CRM MCP Server
    H->>M: spawn subprocess
    H->>M: stdin · JSON-RPC request
    M-->>H: stdout · JSON-RPC response
    M-->>H: stderr · logs only
```

> **Important:** Never `print()` inside an MCP server. stdout is reserved for the JSON-RPC protocol. Use `loguru` (defaults to stderr).

---

## External services

| Service | Purpose | Free tier |
|---|---|---|
| [Supabase](https://supabase.com) | PostgreSQL + pgvector (CRM + memory + chat_sessions) | Yes |
| [Qdrant Cloud](https://qdrant.tech) | Vector DB (RAG KB + CAG cache) | Yes (1 GB) |
| [Groq](https://groq.com) | llama-3.3-70b-versatile (voice fast path, router) | Pay-per-use |
| [OpenRouter](https://openrouter.ai) | Gemini 2.5 Flash (text synthesiser) | Pay-per-use |
| [Tavily](https://tavily.com) | Real-time web search | 1 000 free searches/mo |
| [Langfuse](https://langfuse.com) | Tracing, cost tracking, prompt versioning | Free (hobby) |
| [LiveKit Cloud](https://cloud.livekit.io) | WebRTC infra for voice rooms | Yes (dev tier) |
| [Deepgram](https://deepgram.com) | Streaming STT (Nova-3) | $200 credit |
| [ElevenLabs](https://elevenlabs.io) | Streaming TTS (Turbo v2.5) | 10 k chars/mo free |

---

## Dependency highlights

```
# Voice stack
livekit>=1.0.0
livekit-agents>=1.5.0
livekit-plugins-deepgram>=1.5.0
livekit-plugins-elevenlabs>=1.5.0
livekit-plugins-silero>=1.5.0
deepgram-sdk>=6.0.0
elevenlabs>=2.0.0
sounddevice>=0.5.0
onnxruntime>=1.17.0
torch>=2.0.0

# UI voice integration
livekit-client@^2.5.0           # (in ui/package.json)
framer-motion@^11.11.17         # (already present, used for bubble animation)

# MCP
mcp>=1.27.0
fastmcp>=3.0.0
langchain-mcp-adapters>=0.2.2
```

`pyproject.toml` and `requirements.txt` ship in lock-step. To add a dependency: edit both, or regenerate via `pip-compile pyproject.toml -o requirements.txt`.

---

## Deploying to AWS

Prerequisite: an AWS account + IAM user + CLI configured as profile `disputeai`
(region `us-west-2`). The full beginner walkthrough is
`docs/AWS_From_Zero_Account_and_IAM_Setup.docx`.

**First-time deploy** (one environment: VPC + ALB + cluster, then services):

```bash
# 1. Create the Copilot env (VPC, subnets, ALB, ECS cluster)
copilot env init  && copilot env deploy --name dev

# 2. Provision managed Redis (standalone CloudFormation)
./scripts/aws/deploy_redis.sh

# 3. Push the 21 secrets from .env into Copilot-managed SSM SecureStrings
#    (referenced in copilot/*/manifest.yml as /copilot/${APP}/${ENV}/secrets/KEY)
copilot secret init --cli-input-yaml /tmp/copilot-secrets.yml

# 4. Build arm64 images locally + push to ECR, then deploy each service
./scripts/aws/build_push_images.sh
copilot svc deploy --name api    --env dev
copilot svc deploy --name worker --env dev
copilot svc deploy --name voice  --env dev
```

> Images are tagged `:latest`, so a re-deploy of unchanged code needs
> `copilot svc deploy --force` (or push to `dev` and let CI/CD do it).

**Ongoing deploys — just push:**

```bash
git push origin dev      # GitHub Actions builds, pushes to ECR, rolls api+worker
```

**Pause / resume to save cost** (services bill per hour they exist):

```bash
CL=disputeai-dev-Cluster-KpIKFXAtIZpt
# pause (scale all services to 0 — stops compute)
for s in api worker voice; do
  SVC=$(aws ecs list-services --cluster $CL \
        --query "serviceArns[?contains(@,'-$s-')]" --output text --profile disputeai)
  aws ecs update-service --cluster $CL --service $SVC --desired-count 0 --profile disputeai
done
# resume (api→2, worker→1, voice→1 — voice only when needed)
```

> A paused stack still bills ~$3–5/day for the ALB + NAT + Redis. To reach
> ~$0/day, fully tear down with `copilot app delete` (code/images/secrets stay
> safe in Git/ECR/SSM; rebuild takes ~15–20 min). Cost details and the full
> teardown/rebuild commands are in
> `docs/DisputeAI_Cloud_Services_Cost_and_Decisions.docx`.

---

## Where to read next

| You want to … | Read |
|---|---|
| **Set up your own dev environment** | `SETUP_GUIDE.md` |
| **Set up AWS from zero (account + IAM + CLI)** | `docs/AWS_From_Zero_Account_and_IAM_Setup.docx` |
| **Understand AWS cost, free tier & decisions** | `docs/DisputeAI_Cloud_Services_Cost_and_Decisions.docx` |
| **Deploy to AWS** | the **Deploying to AWS** section above |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'langchain_core'` | Two Pythons on PATH. Use `PYTHONPATH=src python -m uvicorn api.main:app ...` |
| Voice worker loops with `error initializing process` | Already fixed via `initialize_process_timeout=60.0` in `voice/run.py`. If still failing, check `.env` has all LiveKit / Deepgram / Groq / ElevenLabs keys. |
| Voice "Start call" button does nothing | Open browser DevTools → Network — likely the API isn't fully booted yet (lifespan startup takes ~60 s). |
| Bubble freezes at IDLE | Use Chrome. If Chrome too, check mic permission in browser settings. |
| First voice turn takes 4–6 s | Confirm `LLM connection warm — first call took XXX ms` in worker boot log. Confirm `config/param.yaml` has `silence_threshold_ms: 300`. |
| Sidebar doesn't show voice session after hang-up | Check `App.tsx` has the polling `useEffect` while voice modal open. |
| Auto-title never fires | Need ≥ 4 ST turns and the title must still be the auto-generated default. Reset by creating a new session. |
| Langfuse traces missing | `pip install --upgrade 'langfuse>=3.0.0'`. The v2 fallback in `observability.py` works but newer is better. |

