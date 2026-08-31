"""
Rich Multi-Turn Test Script for LangGraph Multi-Agent System.

This script demonstrates:
1. Graph Visualization (Mermaid diagram).
2. Multi-turn conversation exercising all 3 specialized agents.
3. Selective memory injection and distillation.
"""

import os
import sys
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from agents.orchestrator import build_agent
from infrastructure.log import setup_logging

def run_demo():
    setup_logging("INFO")
    
    # 1. Build the Multi-Agent Orchestrator
    logger.info("Initializing LangGraph Multi-Agent System...")
    agent = build_agent()
    
    # 2. Visualize the Graph
    print("\n" + "="*80)
    print("🧠 LANGGRAPH ARCHITECTURE (MERMAID)")
    print("="*80)
    try:
        print(agent.graph.get_graph(xray=True).draw_mermaid())
    except Exception as e:
        logger.warning(f"Could not visualize graph: {e}")
    print("="*80 + "\n")

    # 3. Multi-Turn Conversation Demo
    user_id = "demo_user_w08"
    session_id = "session_langgraph_001"
    
    turns = [
        # Turn 1: Direct Agent (Greeting)
        "Hi! I'm David. Just arrived at DisputeAI for my checkup.",
        
        # Turn 2: Clinical Agent (Patient History / RAG)
        "I've been feeling some chest tightness lately. What are the common protocols for triage here?",
        
        # Turn 3: Clinical Agent (Memory Recall)
        "Oh, and I'm allergic to Penicillin. Please remember that.",
        
        # Turn 4: Administrative Agent (CRM - Doctor Search)
        "Who is the best cardiologist available tonight for a consultation?",
        
        # Turn 5: Administrative Agent (CRM - Appointment)
        "Can you book a slot with Dr. Tharindu for tomorrow morning?",
        
        # Turn 6: Clinical Agent (Selective Memory)
        "Wait, based on my allergy I mentioned earlier, are there any specific meds I should avoid if they admit me?",
        
        # Turn 7: Direct Agent (Exit)
        "Thanks for the help! I'll head to the Electronics now."
    ]

    for i, msg in enumerate(turns, 1):
        print(f"\n[Turn {i}] User: {msg}")
        resp = agent.chat(msg, user_id, session_id)
        
        print(f"--- [REPLY] Route: {resp.route} | Latency: {resp.latency_ms}ms ---")
        print(f"Assistant: {resp.answer}")
        
    logger.success("Multi-turn LangGraph demo completed!")

if __name__ == "__main__":
    run_demo()
