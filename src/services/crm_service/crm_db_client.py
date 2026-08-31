"""
CRM database client — reads bookings from Supabase PostgreSQL.

Provides CRMDatabaseClient with join-based queries for bookings,
patients, doctors, locations, and specialties.
"""

from loguru import logger
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy import text, and_
from sqlalchemy.orm import Session

from infrastructure.db import get_sql_engine
from infrastructure.db.crm_models import (
    Booking, Patient, Doctor, Location, Specialty
)
class CRMDatabaseClient:
    """
    CRM client backed by Supabase PostgreSQL.
    
    Queries the bookings table with proper joins and filtering.
    """
    
    def __init__(self):
        """Initialize CRM database client."""
        self.engine = get_sql_engine()
    
    def load_bookings(
        self, 
        user_id: str, 
        days_ahead: int = 14,
        status_filter: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Load bookings for a user (via external_user_id).
        
        Args:
            user_id: External user identifier (links to patients.external_user_id)
            days_ahead: Number of days to look ahead
            status_filter: Optional list of statuses to include
            
        Returns:
            List of booking dicts with related data
        """
        from sqlalchemy.orm import sessionmaker
        
        Session = sessionmaker(bind=self.engine)
        session = Session()
        
        try:
            # Calculate time window
            now = int(datetime.now().timestamp())
            cutoff = now + (days_ahead * 86400)
            
            # Build query
            query = (
                session.query(Booking)
                .join(Patient, Booking.patient_id == Patient.patient_id)
                .join(Doctor, Booking.doctor_id == Doctor.doctor_id)
                .join(Location, Booking.location_id == Location.location_id)
                .outerjoin(Specialty, Doctor.specialty_id == Specialty.specialty_id)
                .filter(Patient.external_user_id == user_id)
                .filter(and_(
                    Booking.start_at >= now,
                    Booking.start_at <= cutoff
                ))
                .order_by(Booking.start_at)
            )
            
            # Apply status filter if provided
            if status_filter:
                query = query.filter(Booking.status.in_(status_filter))
            
            # Execute and convert to dicts
            bookings = query.all()
            
            result = []
            for booking in bookings:
                booking_dict = booking.to_dict(include_relations=True)
                result.append(booking_dict)
            
            logger.info(f"Loaded {len(result)} bookings for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load bookings: {e}")
            return []
        
        finally:
            session.close()
    
    def get_booking(self, booking_id: str) -> Optional[Dict]:
        """
        Get a single booking by ID.
        
        Args:
            booking_id: Booking identifier
            
        Returns:
            Booking dict or None if not found
        """
        from sqlalchemy.orm import sessionmaker
        
        Session = sessionmaker(bind=self.engine)
        session = Session()
        
        try:
            booking = (
                session.query(Booking)
                .filter(Booking.booking_id == booking_id)
                .first()
            )
            
            if booking:
                return booking.to_dict(include_relations=True)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get booking {booking_id}: {e}")
            return None
        
        finally:
            session.close()
    
    def list_all(self, limit: int = 100) -> List[Dict]:
        """
        List all bookings (for debugging).
        
        Args:
            limit: Maximum number of bookings to return
            
        Returns:
            List of all booking dicts
        """
        from sqlalchemy.orm import sessionmaker
        
        Session = sessionmaker(bind=self.engine)
        session = Session()
        
        try:
            bookings = (
                session.query(Booking)
                .order_by(Booking.start_at.desc())
                .limit(limit)
                .all()
            )
            
            return [b.to_dict(include_relations=True) for b in bookings]
            
        except Exception as e:
            logger.error(f"Failed to list bookings: {e}")
            return []
        
        finally:
            session.close()
    
    def get_patient_by_user_id(self, external_user_id: str) -> Optional[Dict]:
        """
        Get patient record by external user ID.
        
        Args:
            external_user_id: External user identifier
            
        Returns:
            Patient dict or None if not found
        """
        from sqlalchemy.orm import sessionmaker
        
        Session = sessionmaker(bind=self.engine)
        session = Session()
        
        try:
            patient = (
                session.query(Patient)
                .filter(Patient.external_user_id == external_user_id)
                .first()
            )
            
            if patient:
                return patient.to_dict()
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get patient: {e}")
            return None
        
        finally:
            session.close()


# Singleton instance
_crm_client = None


def get_crm_client() -> CRMDatabaseClient:
    """
    Get singleton CRM database client.
    
    Returns:
        CRMDatabaseClient instance
    """
    global _crm_client
    
    if _crm_client is None:
        _crm_client = CRMDatabaseClient()
    
    return _crm_client

