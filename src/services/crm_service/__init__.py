"""
CRM services — Supabase PostgreSQL client, LLM-powered data generation.
"""

from .crm_db_client import CRMDatabaseClient, get_crm_client
from .llm_data_generator import HealthcareDataGenerator, get_data_generator

__all__ = [
    "CRMDatabaseClient",
    "get_crm_client",
    "HealthcareDataGenerator",
    "get_data_generator",
]
