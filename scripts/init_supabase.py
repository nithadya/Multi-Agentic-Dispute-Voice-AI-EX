#!/usr/bin/env python3
"""
Initialize Supabase schema - creates all Memory + CRM tables.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables
load_dotenv()

from loguru import logger
from infrastructure.log import setup_logging
from infrastructure.db.supabase_client import init_supabase_schema


def main():
    setup_logging()
    try:
        init_supabase_schema()
        return 0
    except Exception as e:
        logger.error(f"Failed to initialize schema: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
