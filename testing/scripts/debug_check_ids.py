
import sqlite3

db_path = r"d:\AI Agent Projects\AI Disgnostic\vehicle_info\vehicle_architecture.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

ids_to_check = ['0001', '1', '01', '0x1', '0x0001', '0019', '19']

print("Checking IDs in ecu_info:")
for eid in ids_to_check:
    cursor.execute("SELECT * FROM ecu_info WHERE ecu_address = ?", (eid,))
    res = cursor.fetchone()
    if res:
        print(f"Found {eid}: {res}")
    else:
        print(f"Not found: {eid}")

print("\nChecking IDs in topology (main_ecu):")
for eid in ids_to_check:
    cursor.execute("SELECT * FROM topology WHERE main_ecu = ? LIMIT 1", (eid,))
    res = cursor.fetchone()
    if res:
        print(f"Found {eid}: {res}")

conn.close()
