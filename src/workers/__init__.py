"""
Background workers — Redis-backed, run by Arq.

Today's request path does all post-turn bookkeeping inline:

    LLM responds → memory write → distillation → auto-title → respond

Tomorrow's request path enqueues that bookkeeping and returns immediately:

    LLM responds → enqueue(bookkeeping) → respond
                         │
                         ▼
                 Redis queue (arq)
                         │
                         ▼
                 Worker process pulls jobs, runs them, retries on failure

Wins:
    - Faster — user waits for LLM only.
    - More reliable — if a memory write fails, the worker retries from the queue.
    - Independent scaling — workers scale on queue depth, API scales on RPS.
    - No request loses memory if it crashes mid-write — the queue persists.

Entry point:
    `arq src.workers.tasks.WorkerSettings`   (also wired into Makefile)
"""
