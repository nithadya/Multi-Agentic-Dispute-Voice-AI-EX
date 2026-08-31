import os

file_path = 'src/api/routers/chat.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    '_patient_to_dict': '_customer_to_dict',
    '_fetch_patient_sync': '_fetch_customer_sync',
    '_fetch_upcoming_bookings_sync': '_fetch_recent_orders_sync',
    'patient_id': 'customer_id',
    'patient_task': 'customer_task',
    'cached_patient': 'cached_customer',
    'fallback_patient_id': 'fallback_customer_id',
    '_patient_task': '_customer_task',
    'patient_loaded': 'customer_loaded',
    'patient': 'customer',
    'Patient': 'Customer',
    'PATIENT': 'CUSTOMER',
    'booking_id': 'order_id',
    'booking': 'order',
    'Booking': 'Order',
    'bookings': 'orders',
    'doctor_name': 'items',
    'specialty': 'total_amount',
    'upcoming': 'recent',
    'clinic': 'store',
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Renamed healthcare terms to e-commerce terms in chat.py!")
