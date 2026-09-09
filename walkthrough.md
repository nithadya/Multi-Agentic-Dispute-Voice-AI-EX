# Multi-Vendor Dispute Resolution AI — Walkthrough

## Summary of Changes

The system has been fully built as an **Autonomous Multi-Vendor E-Commerce Dispute Resolution AI**. All components (Database Schema, API Backend, Seeding Scripts, Frontend UI) are mapped to the dispute resolution domain.

### Database & ORM
- Created `customers`, `vendors`, `orders`, and `disputes` tables.
- Implemented `Customer`, `Vendor`, `Order`, and `Dispute` SQLAlchemy models in `src/infrastructure/db/crm_models.py`.

### API Routing
- `src/api/routers/customers.py` — handles customer `lookup`, `register`, and `update` endpoints with the `Customer` schema (`email`, `tier`, `dispute_count`, etc.).
- Updated `main.py` and `chat_sessions.py` to use `customer_id`.

### Seeding Logic (`scripts/seed_crm_unified.py`)
- Seeded with multi-vendor E-commerce entities (TechStore LK, FashionHub, etc.).
- Generates dummy orders and dispute records for the primary login customer.
- Creates the Demo Customer with the explicit phone number `94781030736`.

### Frontend Updates
- React components use `CustomerGate` and `useCustomer` hook.
- UI labels and API calls point to `/customers` routes.
- Tool Explorer UI looks up customers instead of legacy entities.

### Multi-Agent Dispute Graph
- `agents/orchestrator.py` — LangGraph fan-out with specialized sub-agents (admin, dispute, direct, web).
- `agents/decision_graph.py` — guardrail + CAG short-circuit for the text path.
- 4-tier memory (short-term, long-term, episodic, procedural) via Supabase + pgvector + Qdrant.

### Voice Path
- LiveKit + Deepgram STT + ElevenLabs TTS with real token streaming and barge-in memory integrity.
- `achat_stream_fast()` — sub-2-second latency voice path bypassing the multi-agent graph.
- Reactive `VoiceBubble.tsx` UI component with 5 states and latency HUD.

## Verification

Initialize the database and seed the data:

```bash
make init-supabase
make seed-crm-no-llm
```

After seeding, log in through the UI with the phone number `078 103 0736` (Demo Customer) and test:
- CRM lookups for orders and disputes
- Dispute creation and resolution flow via text chat
- Voice-based dispute queries via the Voice button
