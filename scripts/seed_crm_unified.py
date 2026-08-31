"""
Unified E-Commerce Data Seeder - Single script for seeding Dispute AI.

Features:
- Fallback to LLM / Template generation.
- Switch between storage backends (Database vs JSONL).
- Configurable scale parameters and batch generation with progress tracking.
"""

import argparse
import random
import uuid
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass
import pytz
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from loguru import logger
from infrastructure.log import setup_logging
from infrastructure.db.sql_client import get_sql_engine
from infrastructure.db.crm_models import Customer, Vendor, Order, Dispute

# ============================================================================
# CONFIGURATION ENUMS
# ============================================================================
class DataGenerationMode(Enum):
    """Data generation mode."""
    LLM = "llm"          # Use LLM to generate realistic names/notes
    TEMPLATE = "template"  # Use predefined templates (fast, free)

class StorageBackend(Enum):
    """Storage backend for CRM data."""
    DATABASE = "database"  # Database
    JSONL = "jsonl"       # JSON Lines file (simple, portable)

# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================
@dataclass
class CRMSeederConfig:
    """Configuration for E-Commerce data seeder."""
    generation_mode: DataGenerationMode = DataGenerationMode.TEMPLATE
    storage_backend: StorageBackend = StorageBackend.DATABASE
    
    # Scale parameters
    n_vendors: int = 5
    n_customers: int = 20
    n_orders_per_customer: int = 3
    dispute_probability: float = 0.2  # 20% of orders get a dispute
    
    # Scheduling parameters
    start_date: str = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    timezone: str = "Asia/Colombo"
    
    # Other parameters
    rand_seed: int = 42
    output_file: Optional[Path] = None  # For JSONL mode
    
    def __post_init__(self):
        """Validate configuration."""
        if self.storage_backend == StorageBackend.JSONL and not self.output_file:
            self.output_file = Path("data/ecommerce_seed.jsonl")


# ============================================================================
# BASE CLASSES
# ============================================================================
class DataGenerator:
    """Base class for data generators."""
    
    def generate_vendors(self, n: int) -> List[Dict]:
        raise NotImplementedError
        
    def generate_customers(self, n: int) -> List[Dict]:
        raise NotImplementedError
        
    def generate_products(self, n: int, vendor_type: str) -> List[Dict]:
        raise NotImplementedError
        
    def generate_dispute_complaints(self, n: int) -> List[str]:
        raise NotImplementedError


class StorageAdapter:
    """Base class for storage adapters."""
    
    def initialize(self):
        raise NotImplementedError
        
    def store_data(self, data: Dict):
        raise NotImplementedError
        
    def flush(self):
        pass
        
    def finalize(self):
        raise NotImplementedError


# ============================================================================
# TEMPLATE DATA GENERATOR
# ============================================================================
class TemplateDataGenerator(DataGenerator):
    """Generate data using predefined templates (fast, free)."""
    
    def __init__(self):
        self.logger = logger
        
    def generate_vendors(self, n: int) -> List[Dict]:
        base_vendors = [
            ("TechStore LK", "Electronics"),
            ("FashionHub", "Clothing"),
            ("HomeEssentials", "Home"),
            ("SportsGear", "Sports"),
            ("BookWorld", "Books"),
            ("AutoParts LK", "Automotive"),
            ("BeautyCare", "Beauty"),
        ]
        vendors = []
        for i in range(n):
            name, v_type = base_vendors[i % len(base_vendors)]
            if i >= len(base_vendors):
                name = f"{name} {i+1}"
            vendors.append({"name": name, "type": v_type})
        return vendors

    def generate_customers(self, n: int) -> List[Dict]:
        base_names = [
            ("Amal Perera", "M"),
            ("Kamal Jayasuriya", "M"),
            ("Nethmi Wijesinghe", "F"),
            ("Sunil Fernando", "M"),
            ("Madhavi Silva", "F"),
            ("Nuwan Pradeep", "M"),
            ("Kavindi Silva", "F"),
        ]
        customers = []
        for i in range(n):
            name, gender = base_names[i % len(base_names)]
            if i >= len(base_names):
                name = f"{name} {i+1}"
            customers.append({"full_name": name, "gender": gender})
        return customers

    def generate_products(self, n: int, vendor_type: str) -> List[Dict]:
        products_by_type = {
            "Electronics": [("Wireless Mouse", 3500.0), ("Bluetooth Headphones", 12000.0), ("Laptop Charger", 8500.0), ("Smartphone Case", 1500.0)],
            "Clothing": [("Cotton T-Shirt", 2500.0), ("Denim Jeans", 5500.0), ("Running Shoes", 15000.0), ("Jacket", 8000.0)],
            "Home": [("Table Lamp", 4500.0), ("Bed Sheet Set", 6500.0), ("Blender", 18000.0), ("Wall Clock", 3000.0)],
        }
        fallback = [("Generic Item", 1000.0), ("Premium Item", 10000.0)]
        base = products_by_type.get(vendor_type, fallback)
        
        products = []
        for i in range(n):
            name, price = base[i % len(base)]
            products.append({"name": name, "price": price})
        return products

    def generate_dispute_complaints(self, n: int) -> List[str]:
        complaints = [
            "The item arrived damaged with scratched edges.",
            "I received the wrong color. I ordered black but got blue.",
            "The product stopped working after 2 days of use.",
            "Missing parts in the box.",
            "Quality is much poorer than described on the website.",
            "Package never arrived despite tracking saying delivered.",
        ]
        return [complaints[i % len(complaints)] for i in range(n)]

# ============================================================================
# LLM DATA GENERATOR
# ============================================================================
class LLMDataGenerator(DataGenerator):
    """Generate data using LLM (OpenAI)."""
    
    def __init__(self):
        from infrastructure.llm import get_chat_llm
        self.llm = get_chat_llm()
        self._cache = {}
        self.logger = logger
        self.template_fallback = TemplateDataGenerator()
    
    def _invoke_llm_json(self, prompt: str, cache_key: str, n: int, fallback_method, fallback_args=None):
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        self.logger.info(f"🤖 Generating {n} items via LLM for {cache_key}...")
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found")
            
            self._cache[cache_key] = data[:n]
            return data[:n]
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            if fallback_args is not None:
                return fallback_method(*fallback_args)
            return fallback_method(n)

    def generate_vendors(self, n: int) -> List[Dict]:
        prompt = f"""Generate {n} realistic Sri Lankan E-commerce Vendor/Store names and categories.
Output as JSON array: [{{"name": "TechLK", "type": "Electronics"}}, ...]"""
        return self._invoke_llm_json(prompt, f"vendors_{n}", n, self.template_fallback.generate_vendors, (n,))

    def generate_customers(self, n: int) -> List[Dict]:
        prompt = f"""Generate {n} realistic Sri Lankan customer names. Mix Sinhala and Tamil, Male and Female.
Output as JSON array: [{{"full_name": "Amal Perera", "gender": "M"}}, ...]"""
        return self._invoke_llm_json(prompt, f"customers_{n}", n, self.template_fallback.generate_customers, (n,))

    def generate_products(self, n: int, vendor_type: str) -> List[Dict]:
        prompt = f"""Generate {n} realistic product names and prices (in LKR) for a vendor selling {vendor_type}.
Output as JSON array: [{{"name": "Wireless Mouse", "price": 3500.0}}, ...]"""
        return self._invoke_llm_json(prompt, f"products_{vendor_type}_{n}", n, self.template_fallback.generate_products, (n, vendor_type))

    def generate_dispute_complaints(self, n: int) -> List[str]:
        prompt = f"""Generate {n} distinct and realistic e-commerce customer complaint sentences.
Output as JSON array of strings: ["Item arrived damaged.", ...]"""
        return self._invoke_llm_json(prompt, f"complaints_{n}", n, self.template_fallback.generate_dispute_complaints, (n,))

# ============================================================================
# STORAGE ADAPTERS
# ============================================================================
class DatabaseStorageAdapter(StorageAdapter):
    """Store data in SQLite/Postgres database."""
    
    def __init__(self, config: CRMSeederConfig):
        self.config = config
        self.logger = logger
        self.engine = get_sql_engine()
        self.Session = sessionmaker(bind=self.engine)
        self.models = {
            'Vendor': Vendor,
            'Customer': Customer,
            'Order': Order,
            'Dispute': Dispute,
        }
    
    def initialize(self):
        """Initialize database connection and clear existing data."""
        # Initialize schema
        from infrastructure.db.supabase_schema import generate_supabase_schema
        schema_sql = generate_supabase_schema()
        with self.engine.connect() as conn:
            for stmt in schema_sql.split(';'):
                if stmt.strip():
                    try:
                        conn.execute(text(stmt))
                    except Exception:
                        pass
            conn.commit()
            
        self.session = self.Session()
        
        # Clear existing data
        self.logger.info("Clearing existing E-Commerce data...")
        self.session.execute(text("DELETE FROM disputes"))
        self.session.execute(text("DELETE FROM orders"))
        self.session.execute(text("DELETE FROM vendors"))
        self.session.execute(text("DELETE FROM customers"))
        self.session.commit()
    
    def store_data(self, data: Dict):
        model_class = self.models[data['type']]
        instance = model_class(**data['data'])
        self.session.add(instance)
    
    def flush(self):
        self.session.flush()
    
    def finalize(self):
        self.session.commit()
        self.session.close()
        self.logger.info("✓ Data committed to database")


class JSONLStorageAdapter(StorageAdapter):
    """Store data in JSONL file."""
    
    def __init__(self, config: CRMSeederConfig):
        self.config = config
        self.logger = logger
        self.records = []
    
    def initialize(self):
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
        
    def store_data(self, data: Dict):
        self.records.append(data)
    
    def finalize(self):
        with open(self.config.output_file, 'w') as f:
            for rec in self.records:
                f.write(json.dumps(rec) + '\n')
        self.logger.info(f"✓ Wrote {len(self.records)} records to {self.config.output_file}")


# ============================================================================
# UNIFIED E-COMMERCE SEEDER
# ============================================================================
class UnifiedCRMSeeder:
    """Unified E-Commerce data seeder with all capabilities."""
    
    def __init__(self, config: CRMSeederConfig):
        self.config = config
        self.logger = logger
        
        self.generator = self._create_generator()
        self.storage = self._create_storage()
        
        random.seed(config.rand_seed)
    
    def _create_generator(self) -> DataGenerator:
        if self.config.generation_mode == DataGenerationMode.LLM:
            self.logger.info("🤖 Using LLM data generator")
            return LLMDataGenerator()
        else:
            self.logger.info("📋 Using template data generator")
            return TemplateDataGenerator()
    
    def _create_storage(self) -> StorageAdapter:
        if self.config.storage_backend == StorageBackend.DATABASE:
            self.logger.info("🗄️  Using database storage")
            return DatabaseStorageAdapter(self.config)
        else:
            self.logger.info("📄 Using JSONL storage")
            return JSONLStorageAdapter(self.config)
    
    def seed(self):
        self.logger.info("=" * 70)
        self.logger.info("🌱 Starting E-Commerce data seeding")
        self.logger.info("=" * 70)
        
        start_time = time.time()
        self.storage.initialize()
        
        # 1. Generate & Store Vendors
        self.logger.info(f"Seeding {self.config.n_vendors} Vendors...")
        vendors_data = self.generator.generate_vendors(self.config.n_vendors)
        vendors = []
        for vd in vendors_data:
            v_id = str(uuid.uuid4())
            self.storage.store_data({
                'type': 'Vendor',
                'data': {
                    'id': v_id,
                    'name': vd['name'],
                    'type': vd.get('type', 'General'),
                    'active': 1
                }
            })
            vendors.append({'id': v_id, **vd})
            
        self.storage.flush()
            
        # 2. Generate & Store Customers
        self.logger.info(f"Seeding {self.config.n_customers} Customers...")
        customers_data = self.generator.generate_customers(self.config.n_customers)
        customers = []
        
        # Ensure Demo Customer exists
        demo_phone = "94781030736"
        c_id = "0c00f438-76b1-4480-b57c-614f107b8bf8"
        self.storage.store_data({
            'type': 'Customer',
            'data': {
                'id': c_id,
                'external_user_id': demo_phone,
                'name': "Demo Customer",
                'email': "demo@example.com",
                'tier': "premium",
                'total_orders': 0, # Will be updated
                'dispute_count': 0, # Will be updated
                'active': 1
            }
        })
        customers.append({'id': c_id, 'external_user_id': demo_phone, 'name': "Demo Customer"})
        
        for i, cd in enumerate(customers_data):
            # Skip if we already hit n_customers (1 is Demo)
            if i >= self.config.n_customers - 1:
                break
            random_c_id = str(uuid.uuid4())
            phone = f"947{random.randint(10000000, 99999999)}"
            email = f"{cd['full_name'].lower().replace(' ', '.')}@gmail.com"
            tier = random.choices(["standard", "premium", "vip"], weights=[0.7, 0.2, 0.1])[0]
            
            self.storage.store_data({
                'type': 'Customer',
                'data': {
                    'id': random_c_id,
                    'external_user_id': phone,
                    'name': cd['full_name'],
                    'email': email,
                    'tier': tier,
                    'total_orders': 0,
                    'dispute_count': 0,
                    'active': 1
                }
            })
            customers.append({'id': random_c_id, 'external_user_id': phone, 'name': cd['full_name']})
            
        self.storage.flush()
            
        # 3. Generate & Store Orders and Disputes
        self.logger.info(f"Seeding Orders and Disputes...")
        tz = pytz.timezone(self.config.timezone)
        start_dt = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        start_dt = tz.localize(start_dt)
        now_dt = datetime.now(tz)
        
        total_orders = 0
        total_disputes = 0
        
        # Pre-generate products for vendors
        vendor_products = {}
        for v in vendors:
            vendor_products[v['id']] = self.generator.generate_products(5, v.get('type', 'General'))
            
        complaints = self.generator.generate_dispute_complaints(max(10, int(self.config.n_customers * self.config.n_orders_per_customer * self.config.dispute_probability)))

        for c in customers:
            customer_orders = random.randint(1, self.config.n_orders_per_customer)
            customer_disputes = 0
            
            # If Demo Customer, inject specific hardcoded test scenarios
            if c['id'] == "0c00f438-76b1-4480-b57c-614f107b8bf8":
                # Smart TV Return
                self.storage.store_data({
                    'type': 'Order',
                    'data': {
                        'id': str(uuid.uuid4()),
                        'order_number': f"ORD-SMARTTV",
                        'customer_id': c['id'],
                        'vendor_id': vendors[0]['id'],
                        'status': "delivered",
                        'total_amount': 85000.0,
                        'currency': "LKR",
                        'purchase_date': (now_dt - timedelta(days=45)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        'items': [
                            {"name": "65-Inch Smart TV", "price": 75000.0, "quantity": 1},
                            {"name": "Premium Warranty (2 Years)", "price": 10000.0, "quantity": 1}
                        ],
                        'shipping_address': "123 Demo St, Colombo",
                        'tracking_number': "TRK-SMARTTV123"
                    }
                })
                # Missing Laptop
                self.storage.store_data({
                    'type': 'Order',
                    'data': {
                        'id': str(uuid.uuid4()),
                        'order_number': f"ORD-LAPTOP",
                        'customer_id': c['id'],
                        'vendor_id': vendors[0]['id'],
                        'status': "delivered", # marked delivered but missing
                        'total_amount': 250000.0,
                        'currency': "LKR",
                        'purchase_date': (now_dt - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        'items': [
                            {"name": "Gaming Laptop Pro", "price": 250000.0, "quantity": 1}
                        ],
                        'shipping_address': "123 Demo St, Colombo",
                        'tracking_number': "TRK-LAPTOP456"
                    }
                })
                total_orders += 2
                
            for _ in range(customer_orders):
                v = random.choice(vendors)
                products = vendor_products[v['id']]
                
                # Pick 1-3 items for this order
                num_items = random.randint(1, 3)
                items = []
                order_total = 0.0
                for _ in range(num_items):
                    prod = random.choice(products)
                    qty = random.randint(1, 2)
                    items.append({"name": prod['name'], "price": prod['price'], "quantity": qty})
                    order_total += prod['price'] * qty
                
                order_id = str(uuid.uuid4())
                order_date = start_dt + timedelta(days=random.randint(0, (now_dt - start_dt).days))
                
                status = random.choices(
                    ["processing", "shipped", "delivered", "cancelled"], 
                    weights=[0.1, 0.2, 0.6, 0.1]
                )[0]
                
                self.storage.store_data({
                    'type': 'Order',
                    'data': {
                        'id': order_id,
                        'order_number': f"ORD-{uuid.uuid4().hex[:8].upper()}",
                        'customer_id': c['id'],
                        'vendor_id': v['id'],
                        'status': status,
                        'total_amount': order_total,
                        'currency': "LKR",
                        'purchase_date': order_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        'items': items,
                        'shipping_address': f"{random.randint(1, 999)} Random Street, Colombo",
                        'tracking_number': f"TRK-{random.randint(100000, 999999)}" if status in ["shipped", "delivered"] else None
                    }
                })
                total_orders += 1
                
                # Chance of dispute if delivered
                if status == "delivered" and random.random() < self.config.dispute_probability:
                    dispute_id = str(uuid.uuid4())
                    complaint = random.choice(complaints)
                    d_status = random.choices(["investigating", "resolved", "refunded"], weights=[0.4, 0.4, 0.2])[0]
                    
                    self.storage.store_data({
                        'type': 'Dispute',
                        'data': {
                            'id': dispute_id,
                            'dispute_number': f"DSP-{uuid.uuid4().hex[:8].upper()}",
                            'order_id': order_id,
                            'customer_id': c['id'],
                            'type': random.choice(["damaged", "missing_item", "wrong_item", "not_received"]),
                            'status': d_status,
                            'complaint_text': complaint,
                            'evidence_urls': [f"http://example.com/evidence_{random.randint(1,100)}.jpg"],
                            'decision': "APPROVED" if d_status == "refunded" else "PENDING",
                            'refund_amount': order_total if d_status == "refunded" else 0.0,
                            'currency': "LKR",
                            'resolution_notes': "Refund processed." if d_status == "refunded" else "",
                            'customer_notes': "Please resolve ASAP."
                        }
                    })
                    total_disputes += 1
                    customer_disputes += 1
            
            # Update customer counts (If we were doing raw SQL updates we'd do it here, but we set to 0 initially.
            # In a real ORM setup we might update the object before commit, but since this is dummy data seed, 
            # we can run a bulk update at the end or just let them be 0. We'll run a raw SQL update in finalize).
        
        # Finalize
        self.storage.finalize()
        
        # Bulk update counts if using database
        if isinstance(self.storage, DatabaseStorageAdapter):
            with self.storage.engine.connect() as conn:
                conn.execute(text("""
                    UPDATE customers 
                    SET total_orders = (SELECT COUNT(*) FROM orders WHERE orders.customer_id = customers.id),
                        dispute_count = (SELECT COUNT(*) FROM disputes WHERE disputes.customer_id = customers.id)
                """))
                conn.commit()
        
        elapsed = time.time() - start_time
        
        self.logger.info("=" * 70)
        self.logger.info("✅ Seeding complete!")
        self.logger.info(f"   Time: {elapsed:.1f}s")
        self.logger.info(f"   Vendors: {len(vendors)}")
        self.logger.info(f"   Customers: {len(customers)}")
        self.logger.info(f"   Orders: {total_orders}")
        self.logger.info(f"   Disputes: {total_disputes}")
        self.logger.info("=" * 70)

# ============================================================================
# CLI INTERFACE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified E-Commerce Data Seeder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LLM + Database (default)
  python seed_crm_unified.py --mode llm --n-customers 20
  
  # Template + Database (fast, free)
  python seed_crm_unified.py --mode template --n-customers 50 --n-vendors 10
  
  # Template + JSONL file
  python seed_crm_unified.py --mode template --storage jsonl --output data/ecommerce.jsonl
        """
    )
    
    parser.add_argument('--mode', choices=['llm', 'template'], default='template', help='Data generation mode')
    parser.add_argument('--storage', choices=['database', 'jsonl'], default='database', help='Storage backend')
    
    parser.add_argument('--n-vendors', type=int, default=5, help='Number of vendors')
    parser.add_argument('--n-customers', type=int, default=20, help='Number of customers')
    parser.add_argument('--n-orders', type=int, default=3, help='Max orders per customer')
    parser.add_argument('--dispute-prob', type=float, default=0.2, help='Probability of a delivered order having a dispute')
    
    parser.add_argument('--start', default=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"), help='Start date (YYYY-MM-DD)')
    parser.add_argument('--tz', default='Asia/Colombo', help='Timezone')
    parser.add_argument('--output', type=Path, help='Output file (for JSONL mode)')
    parser.add_argument('--rand-seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    setup_logging()
    
    config = CRMSeederConfig(
        generation_mode=DataGenerationMode(args.mode),
        storage_backend=StorageBackend(args.storage),
        n_vendors=args.n_vendors,
        n_customers=args.n_customers,
        n_orders_per_customer=args.n_orders,
        dispute_probability=args.dispute_prob,
        start_date=args.start,
        timezone=args.tz,
        rand_seed=args.rand_seed,
        output_file=args.output,
    )
    
    seeder = UnifiedCRMSeeder(config)
    seeder.seed()

if __name__ == "__main__":
    main()
