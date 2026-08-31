"""
Launch the DisputeAI E-Commerce Specialist API.

Usage:
    python -m api.run                    # from src/
    python src/api/run.py                # from project root

Options (env vars):
    API_HOST=0.0.0.0   (default)
    API_PORT=8000       (default)
    API_RELOAD=true     (default: true for dev)
"""

import os
import sys
import uvicorn

# Ensure src/ is on the path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)
os.chdir(src_dir)

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"

    print(f"\n  DisputeAI E-Commerce Specialist API")
    print(f"  {'─' * 40}")
    print(f"  Host    : {host}")
    print(f"  Port    : {port}")
    print(f"  Reload  : {reload}")
    print(f"  Docs    : http://localhost:{port}/docs")
    print(f"  {'─' * 40}\n")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
