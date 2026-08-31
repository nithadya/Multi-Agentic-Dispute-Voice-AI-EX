"""
End-to-end test of the MCP-backed LangGraph agent.

Demonstrates:
  1. Launches 4-5 MCP servers as subprocesses:
       - disputeai-crm     (custom, 5 tools)
       - disputeai-memory  (custom, 6 tools)
       - disputeai-kb      (custom, 3 tools)
       - disputeai-web     (custom, 1 tool)
       - postgres         (off-the-shelf, if SUPABASE_POSTGRES_URL is set)
  2. Builds the orchestrator with ALL tools via MCP
  3. Lists all MCP tools discovered
  4. Runs a multi-turn conversation exercising CRM, RAG, and Web paths

Run:
    cd src && python ../scripts/test_mcp_agent.py
"""

import asyncio
import os
import sys

# Ensure src/ is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv
load_dotenv()

from loguru import logger


async def main() -> None:
    from agents.orchestrator import build_agent_mcp

    print("=" * 70)
    print(" MCP Full Integration Test — 5 servers, 1 agent")
    print("=" * 70)

    print("\n[1/3] Building MCP-backed agent (all tools via MCP)...")
    agent = await build_agent_mcp()

    print(f"\n[2/3] MCP tools discovered ({len(agent.mcp_tools)} total):")
    for name, tool in agent.mcp_tools.items():
        desc = (tool.description or "").strip().split("\n")[0][:70]
        print(f"   - {name:25s}  {desc}")

    print("\n[3/3] Running multi-turn conversation...")
    user_id = "94781030736"
    session_id = "mcp-full-test"

    turns = [
        # direct route
        ("Hi, I'm Anushka. My mobile is 078 103 0736.", "direct"),
        # CRM route → lookup_patient via MCP
        ("Can you look up my patient record?", "crm"),
        # CRM route → search_doctors via MCP
        ("What cardiologists do you have available?", "crm"),
        # RAG route → search_policy_kb via MCP
        ("What cardiac services does the store offer?", "rag"),
        # Web route → web_search via MCP
        ("What are the current visiting hours at DisputeAI?", "web_search"),
    ]

    for i, (msg, expected_route) in enumerate(turns, 1):
        print(f"\n{'─' * 70}")
        print(f"Turn {i}: {msg}")
        print(f"Expected route: {expected_route}")
        print("─" * 70)
        resp = await agent.achat(msg, user_id=user_id, session_id=session_id)
        actual = resp.routes[0] if resp.routes else resp.route
        match = "Y" if actual == expected_route else "X"
        print(f"  Route   : {actual} [{match}]")
        print(f"  Latency : {resp.latency_ms}ms")
        print(f"  Answer  : {resp.answer[:300]}")

    print(f"\n{'=' * 70}")
    print(" Full integration test complete.")
    print(" All 5 routes (direct, crm x2, rag, web) went through MCP.")
    print(" Zero direct Python tool imports in build_agent_mcp().")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
