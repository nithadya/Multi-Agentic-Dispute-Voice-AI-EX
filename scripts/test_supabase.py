#!/usr/bin/env python3
"""
Test Supabase connection and pgvector installation.
"""

import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from loguru import logger
from infrastructure.log import setup_logging
from infrastructure.db.supabase_client import (
    test_connection,
    check_pgvector_installed,
)


def main():
    setup_logging()

    logger.info("Testing database connection…")
    success = test_connection()

    logger.info("Checking pgvector extension…")
    pgvector = check_pgvector_installed()

    if success and pgvector:
        logger.success("All checks passed! Supabase is ready.")
        return 0
    else:
        logger.error("Some checks failed. See errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
