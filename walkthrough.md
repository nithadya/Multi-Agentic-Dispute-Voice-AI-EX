# E-commerce AI Refactoring Walkthrough

## Summary of Changes

The legacy Nawaloka hospital/clinic CRM has been completely replaced with an E-commerce Dispute AI system, allowing you to test the AI agent with domain-specific mock data. All components (Database Schema, API Backend, Seeding Scripts, Frontend UI) have been successfully mapped to the new architecture.

### Database & ORM
- Created `customers`, `vendors`, `orders`, and `disputes` tables.
- Replaced `Patient`, `Doctor`, and `Booking` models in `src/infrastructure/db/crm_models.py` with `Customer`, `Vendor`, `Order`, and `Dispute` SQLAlchemy models.

### API Routing
- Refactored `src/api/routers/patients.py` to `customers.py`. 
- Updated `lookup`, `register`, and `update` endpoints to use the new `Customer` schema (`email`, `tier`, `dispute_count`, etc.).
- Updated `main.py` and `chat_sessions.py` to use `customer_id`.

### Seeding Logic (`scripts/seed_crm_unified.py`)
- Swapped hospital entities for Dummy E-commerce Stores (TechStore LK, FashionHub).
- Generates dummy orders and a dispute record for the primary login user.
- Creates the Demo Customer with the explicit phone number `94781030736`. 

### Frontend Updates
- Migrated React components: `PatientGate` → `CustomerGate`, `usePatient` → `useCustomer`.
- Swapped UI labels and API calls to point to `/customers` routes.
- Updated the Tool Explorer UI to lookup customers instead of patients.

## Verification
You are now ready to initialize the database and seed the data! 
As requested, you can now run the following commands in your own terminal to test the full AI:

```bash
make init-supabase
make seed-crm-no-llm
```

After doing so, you can log in through the UI with the phone number `078 103 0736` (which Maps to Demo Customer) and test the CRM capabilities for orders and disputes.
