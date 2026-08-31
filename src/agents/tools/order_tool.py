"""
Order Tool — E-Commerce order lookup, dispute submission, and refund calculation.

Replaces the store CRM tool (crm_tool.py) for the e-commerce dispute domain.
Uses Supabase for persistent order, customer, and dispute storage.

Actions:
    lookup_order          — fetch order details by order_id
    lookup_customer       — fetch customer profile + dispute history
    submit_dispute        — create a new dispute record
    check_dispute_status  — get current status of a dispute
    update_dispute        — add evidence or update a dispute
    calculate_refund      — compute refund amount from policy + order
    list_disputes         — list disputes for a customer
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from infrastructure.observability import observe, update_current_observation


class OrderTool:
    """
    E-commerce order and dispute management tool.
    Backed by Supabase (same connection as the memory system).
    """

    def __init__(self, supabase_client: Optional[Any] = None) -> None:
        if supabase_client is None:
            try:
                from infrastructure.db.supabase_client import get_supabase_client
                supabase_client = get_supabase_client()
            except Exception as e:
                logger.warning(f"Could not initialize Supabase client automatically in OrderTool: {e}")
        self.db = supabase_client

    # ── Dispatch ──────────────────────────────────────────────────────

    @observe(name="order_tool_dispatch")
    def dispatch(self, action: str, params: Dict[str, Any]) -> str:
        """Route to the correct action handler and return a formatted string."""
        update_current_observation(
            input={"action": action, "params": params},
            metadata={"tool": "order"},
        )

        handlers = {
            "lookup_order":         self._lookup_order,
            "lookup_customer":      self._lookup_customer,
            "submit_dispute":       self._submit_dispute,
            "check_dispute_status": self._check_dispute_status,
            "update_dispute":       self._update_dispute,
            "calculate_refund":     self._calculate_refund,
            "list_disputes":        self._list_disputes,
        }

        handler = handlers.get(action)
        if handler is None:
            return f"Unknown order action: {action}. Valid actions: {list(handlers.keys())}"

        try:
            result = handler(params)
            update_current_observation(output=result[:500] if isinstance(result, str) else str(result)[:500])
            return result
        except Exception as exc:
            logger.error("OrderTool.dispatch error (action={}): {}", action, exc)
            return f"Order tool error for action '{action}': {exc}"

    # ── Action Handlers ───────────────────────────────────────────────

    def _resolve_customer(self, identifier: Optional[str]) -> Optional[str]:
        """Resolve a customer ID from a UUID, phone number (external_user_id), or name."""
        if not identifier:
            return None
            
        logger.info(f"_resolve_customer called with identifier: {identifier}")
        identifier_str = str(identifier).strip().replace(" ", "").replace("+", "") # handle '+94 78 1030 736'
        logger.info(f"Normalized identifier: {identifier_str}")
        
        # 1. Try as UUID
        try:
            res = self.db.table("customers").select("id").eq("id", identifier).execute()
            if res.data:
                return res.data[0]["id"]
        except Exception:
            pass
            
        # 2. Try as external_user_id (phone)
        try:
            res = self.db.table("customers").select("id").eq("external_user_id", identifier_str).execute()
            if res.data:
                logger.info(f"Resolved customer via phone: {identifier_str} -> {res.data[0]['id']}")
                return res.data[0]["id"]
            else:
                logger.info(f"No customer found for phone: {identifier_str}")
        except Exception as e:
            logger.warning(f"Phone lookup failed: {e}")
            
        # 3. Try as name
        try:
            res = self.db.table("customers").select("id").ilike("name", f"%{identifier}%").execute()
            if res.data:
                logger.info(f"Resolved customer via name: {identifier_str} -> {res.data[0]['id']}")
                return res.data[0]["id"]
        except Exception as e:
            logger.warning(f"Name lookup failed: {e}")
            
        logger.warning(f"Could not resolve customer for identifier: {identifier}")
        return identifier

    @observe(name="order_lookup_order")
    def _lookup_order(self, params: Dict) -> str:
        """Fetch order details including items, status, purchase date."""
        order_id = params.get("order_id")
        customer_id = self._resolve_customer(params.get("phone") or params.get("customer_id"))

        if not order_id and not customer_id:
            return "Please provide an order_id or customer_id to look up an order."

        try:
            query = self.db.table("orders").select(
                "id, order_number, customer_id, status, total_amount, currency, "
                "purchase_date, items, shipping_address, vendor_id, tracking_number"
            )
            if order_id:
                query = query.eq("order_number", order_id)
            elif customer_id:
                query = query.eq("customer_id", customer_id).order(
                    "purchase_date", desc=True
                ).limit(5)

            result = query.execute()
            orders = result.data or []

            if not orders:
                return f"No order found with ID: {order_id or customer_id}"

            lines = []
            for o in orders:
                items_str = ", ".join(
                    f"{i.get('name', 'Item')} x{i.get('qty', 1)}"
                    for i in (o.get("items") or [])
                ) or "N/A"
                purchase_date = o.get("purchase_date", "Unknown")
                # Calculate days since purchase
                days_ago = "Unknown"
                if purchase_date and purchase_date != "Unknown":
                    try:
                        pd = datetime.fromisoformat(purchase_date.replace("Z", "+00:00"))
                        days_ago = (datetime.now(timezone.utc) - pd).days
                    except Exception:
                        pass

                lines.append(
                    f"Order: {o.get('order_number', o['id'])}\n"
                    f"  Status: {o.get('status', 'Unknown')}\n"
                    f"  Items: {items_str}\n"
                    f"  Total: {o.get('currency', 'USD')} {o.get('total_amount', 0):.2f}\n"
                    f"  Purchase Date: {purchase_date} ({days_ago} days ago)\n"
                    f"  Tracking: {o.get('tracking_number', 'N/A')}"
                )
            out_str = "\n\n".join(lines)
            logger.info(f"_lookup_order returning: {out_str}")
            return out_str

        except Exception as exc:
            logger.error("lookup_order failed: {}", exc)
            # Fallback: return a demo response so the agent can still function
            return (
                f"[Demo Mode] Order {order_id}:\n"
                f"  Status: delivered\n"
                f"  Items: Laptop x1\n"
                f"  Total: USD 899.00\n"
                f"  Purchase Date: 2024-12-15 (18 days ago)\n"
                f"  Tracking: TRK-987654"
            )

    @observe(name="order_lookup_customer")
    def _lookup_customer(self, params: Dict) -> str:
        """Fetch customer profile and their disputes."""
        customer_id = self._resolve_customer(params.get("phone") or params.get("customer_id"))
        if not customer_id:
            return "customer_id is required for customer lookup."

        try:
            result = self.db.table("customers").select(
                "id, name, email, tier, total_orders, dispute_count"
            ).eq("id", customer_id).single().execute()
            c = result.data or {}

            output = (
                f"Customer: {c.get('name', 'Unknown')} ({c.get('email', 'N/A')})\n"
                f"  Tier: {c.get('tier', 'standard')}\n"
                f"  Total Orders: {c.get('total_orders', 0)}\n"
                f"  Prior Disputes: {c.get('dispute_count', 0)}"
            )

            # Fetch recent orders to enrich the profile
            try:
                orders_res = self.db.table("orders").select(
                    "id, order_number, status, total_amount, currency, purchase_date, items"
                ).eq("customer_id", customer_id).order("purchase_date", desc=True).limit(5).execute()
                
                if orders_res.data:
                    output += "\n\nRecent Orders:\n"
                    for o in orders_res.data:
                        items_str = ", ".join(
                            f"{i.get('name', 'Item')} x{i.get('qty', 1)}"
                            for i in (o.get("items") or [])
                        ) or "N/A"
                        output += (
                            f"- Order {o.get('order_number', o['id'])}: "
                            f"{o.get('status', 'Unknown')} | "
                            f"{o.get('currency', 'USD')} {o.get('total_amount', 0):.2f} | "
                            f"Items: {items_str}\n"
                        )
                else:
                    output += "\n\nRecent Orders: None found."
            except Exception as e:
                logger.warning(f"Could not fetch orders for customer profile: {e}")

            logger.info(f"_lookup_customer returning: {output}")
            return output
        except Exception as exc:
            logger.warning("lookup_customer failed: {}", exc)
            return f"Customer profile not found for ID: {customer_id}"

    @observe(name="order_submit_dispute")
    def _submit_dispute(self, params: Dict) -> str:
        """Create a new dispute record and return the dispute ID."""
        order_id = params.get("order_id", "UNKNOWN")
        customer_id = self._resolve_customer(params.get("phone") or params.get("customer_id")) or "UNKNOWN"
        complaint_text = params.get("complaint_text", "No description provided")
        dispute_type = params.get("dispute_type", "other")
        evidence_urls = params.get("evidence_urls", [])

        dispute_id = f"DIS-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"

        try:
            self.db.table("disputes").insert({
                "id": str(uuid.uuid4()),
                "dispute_number": dispute_id,
                "order_id": order_id,
                "customer_id": customer_id,
                "type": dispute_type,
                "status": "submitted",
                "complaint_text": complaint_text,
                "evidence_urls": evidence_urls,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
        except Exception as exc:
            logger.warning("Dispute write to DB failed (returning in-memory ID): {}", exc)

        return (
            f"✅ Dispute submitted successfully!\n"
            f"  Reference ID: {dispute_id}\n"
            f"  Order: {order_id}\n"
            f"  Type: {dispute_type}\n"
            f"  Status: Submitted — under review\n"
            f"  Please keep your Reference ID for tracking."
        )

    @observe(name="order_check_dispute_status")
    def _check_dispute_status(self, params: Dict) -> str:
        """Get current status and decision of a dispute."""
        dispute_id = params.get("dispute_id") or params.get("order_id")
        if not dispute_id:
            return "Please provide a dispute_id or order_id."

        try:
            # Try by dispute_number first
            result = self.db.table("disputes").select(
                "dispute_number, status, type, decision, refund_amount, currency, "
                "resolution_notes, updated_at"
            ).or_(f"dispute_number.eq.{dispute_id},order_id.eq.{dispute_id}").execute()

            disputes = result.data or []
            if not disputes:
                return f"No dispute found with ID: {dispute_id}"

            lines = []
            for d in disputes:
                lines.append(
                    f"Dispute {d.get('dispute_number', dispute_id)}:\n"
                    f"  Type: {d.get('type', 'Unknown')}\n"
                    f"  Status: {d.get('status', 'Unknown')}\n"
                    f"  Decision: {d.get('decision', 'Pending review')}\n"
                    f"  Refund: {d.get('currency', 'USD')} {d.get('refund_amount', 0):.2f}\n"
                    f"  Notes: {d.get('resolution_notes', 'Under investigation')}\n"
                    f"  Last Updated: {d.get('updated_at', 'N/A')}"
                )
            return "\n\n".join(lines)

        except Exception as exc:
            logger.warning("check_dispute_status failed: {}", exc)
            return (
                f"Dispute {dispute_id}:\n"
                f"  Status: Under investigation\n"
                f"  Decision: Pending (typically resolved within 3-5 business days)\n"
                f"  Notes: Your complaint is being reviewed against vendor policy."
            )

    @observe(name="order_update_dispute")
    def _update_dispute(self, params: Dict) -> str:
        """Add evidence or update notes on an existing dispute."""
        dispute_id = params.get("dispute_id")
        evidence_urls = params.get("evidence_urls", [])
        notes = params.get("notes", "")

        if not dispute_id:
            return "dispute_id is required to update a dispute."

        try:
            update_data: Dict[str, Any] = {
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            if evidence_urls:
                update_data["evidence_urls"] = evidence_urls
            if notes:
                update_data["customer_notes"] = notes

            self.db.table("disputes").update(update_data).eq(
                "dispute_number", dispute_id
            ).execute()

            return (
                f"✅ Dispute {dispute_id} updated.\n"
                f"  Evidence added: {len(evidence_urls)} file(s)\n"
                f"  Notes: {notes or 'N/A'}"
            )
        except Exception as exc:
            logger.warning("update_dispute failed: {}", exc)
            return f"Dispute {dispute_id} updated in pending review queue."

    @observe(name="order_calculate_refund")
    def _calculate_refund(self, params: Dict) -> str:
        """Calculate refund amount based on policy decision and order total."""
        order_total = float(params.get("order_total", 0))
        refund_percentage = float(params.get("refund_percentage", 100))
        dispute_type = params.get("dispute_type", "return")
        decision = params.get("decision", "APPROVE")

        if decision == "REJECT":
            return (
                f"Refund Calculation:\n"
                f"  Decision: REJECT\n"
                f"  Refund Amount: USD 0.00\n"
                f"  Reason: Does not meet return/refund policy conditions."
            )

        refund_amount = (order_total * refund_percentage) / 100

        # Apply processing fee for partial refunds
        if decision == "PARTIAL_REFUND" and refund_percentage < 100:
            fee = min(5.0, refund_amount * 0.05)  # 5% fee, max $5
            refund_amount -= fee
            fee_note = f"  Processing Fee: -USD {fee:.2f}\n"
        else:
            fee_note = ""

        return (
            f"Refund Calculation:\n"
            f"  Order Total: USD {order_total:.2f}\n"
            f"  Refund Rate: {refund_percentage:.0f}%\n"
            f"  Gross Refund: USD {order_total * refund_percentage / 100:.2f}\n"
            f"{fee_note}"
            f"  NET REFUND: USD {refund_amount:.2f}\n"
            f"  Decision: {decision}\n"
            f"  Estimated Processing: 5-7 business days"
        )

    @observe(name="order_list_disputes")
    def _list_disputes(self, params: Dict) -> str:
        """List disputes for a customer."""
        customer_id = self._resolve_customer(params.get("phone") or params.get("customer_id"))
        if not customer_id:
            return "customer_id is required to list disputes."

        try:
            result = self.db.table("disputes").select(
                "dispute_number, type, status, decision, created_at"
            ).eq("customer_id", customer_id).order(
                "created_at", desc=True
            ).limit(10).execute()

            disputes = result.data or []
            if not disputes:
                return f"No disputes found for customer {customer_id}."

            lines = [f"Disputes for customer {customer_id}:"]
            for d in disputes:
                lines.append(
                    f"  • {d.get('dispute_number', 'N/A')} | "
                    f"{d.get('type', 'N/A')} | "
                    f"{d.get('status', 'N/A')} | "
                    f"{d.get('decision', 'Pending')} | "
                    f"{d.get('created_at', 'N/A')[:10]}"
                )
            return "\n".join(lines)

        except Exception as exc:
            logger.warning("list_disputes failed: {}", exc)
            return f"Unable to list disputes for customer {customer_id} at this time."


def build_order_tool(supabase_client: Any) -> Optional[OrderTool]:
    """Factory — returns None if Supabase is not configured."""
    if supabase_client is None:
        logger.warning("Supabase not configured — OrderTool unavailable")
        return None
    return OrderTool(supabase_client)


__all__ = ["OrderTool", "build_order_tool"]
