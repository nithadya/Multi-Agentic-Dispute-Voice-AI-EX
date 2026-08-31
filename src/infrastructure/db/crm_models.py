"""
E-commerce database models (SQLAlchemy ORM).

Replaces the old CRM models with Customer, Vendor, Order, Dispute.
"""

import time
from sqlalchemy import Column, String, Integer, Float, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


class Customer(Base):
    """Customer record."""
    __tablename__ = "customers"
    
    id = Column(String, primary_key=True)
    external_user_id = Column(String, unique=True, nullable=False)  # Phone number
    name = Column(String, nullable=False)
    email = Column(String)
    tier = Column(String, default='standard')
    total_orders = Column(Integer, default=0)
    dispute_count = Column(Integer, default=0)
    active = Column(Integer, nullable=False, default=1)
    
    # We use string for timestamps in some legacy parts or let DB handle it.
    # The schema uses TIMESTAMPTZ, but for simplicity we can just map it loosely or rely on DB defaults.
    
    def to_dict(self):
        return {
            "id": self.id,
            "external_user_id": self.external_user_id,
            "name": self.name,
            "email": self.email,
            "tier": self.tier,
            "total_orders": self.total_orders,
            "dispute_count": self.dispute_count,
            "active": self.active,
        }


class Vendor(Base):
    """Vendor record."""
    __tablename__ = "vendors"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    type = Column(String)
    active = Column(Integer, nullable=False, default=1)
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "active": self.active,
        }


class Order(Base):
    """E-commerce order."""
    __tablename__ = "orders"
    
    id = Column(String, primary_key=True)
    order_number = Column(String, unique=True, nullable=False)
    customer_id = Column(String, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    vendor_id = Column(String, ForeignKey("vendors.id", ondelete="SET NULL"))
    status = Column(String, nullable=False)
    total_amount = Column(Float, nullable=False)
    currency = Column(String, default="LKR")
    purchase_date = Column(String)  # or DateTime, depending on dialect, using String for loose mapping
    items = Column(JSONB, default=lambda: [])
    shipping_address = Column(String)
    tracking_number = Column(String)
    
    def to_dict(self):
        return {
            "id": self.id,
            "order_number": self.order_number,
            "customer_id": self.customer_id,
            "vendor_id": self.vendor_id,
            "status": self.status,
            "total_amount": self.total_amount,
            "currency": self.currency,
            "purchase_date": self.purchase_date,
            "items": self.items,
            "shipping_address": self.shipping_address,
            "tracking_number": self.tracking_number,
        }


class Dispute(Base):
    """E-commerce dispute."""
    __tablename__ = "disputes"
    
    id = Column(String, primary_key=True)
    dispute_number = Column(String, unique=True, nullable=False)
    order_id = Column(String, nullable=False)
    customer_id = Column(String, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    complaint_text = Column(Text)
    evidence_urls = Column(JSONB, default=lambda: [])
    decision = Column(String)
    refund_amount = Column(Float)
    currency = Column(String, default="LKR")
    resolution_notes = Column(Text)
    customer_notes = Column(Text)
    
    def to_dict(self):
        return {
            "id": self.id,
            "dispute_number": self.dispute_number,
            "order_id": self.order_id,
            "customer_id": self.customer_id,
            "type": self.type,
            "status": self.status,
            "complaint_text": self.complaint_text,
            "evidence_urls": self.evidence_urls,
            "decision": self.decision,
            "refund_amount": self.refund_amount,
            "currency": self.currency,
            "resolution_notes": self.resolution_notes,
            "customer_notes": self.customer_notes,
        }


class ChatSession(Base):
    """ChatGPT-style conversation thread for a customer."""
    __tablename__ = "chat_sessions"

    session_id      = Column(String, primary_key=True)
    customer_id     = Column(String, nullable=False, index=True)
    title           = Column(String, nullable=False)
    last_message_at = Column(Integer)                                  # epoch seconds
    created_at      = Column(Integer, default=lambda: int(time.time()))
    updated_at      = Column(Integer, default=lambda: int(time.time()),
                              onupdate=lambda: int(time.time()))
    archived        = Column(Integer, nullable=False, default=0)

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "customer_id": self.customer_id,
            "title": self.title,
            "last_message_at": self.last_message_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archived": int(self.archived or 0),
        }
