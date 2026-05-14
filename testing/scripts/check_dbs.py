import sqlite3

print("=== vehicle_architecture.db ===")
conn = sqlite3.connect("vehicle_info/vehicle_architecture.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", tables)
cursor.execute("SELECT COUNT(*) FROM ecu_info")
print("ecu_info count:", cursor.fetchone()[0])
cursor.execute("SELECT * FROM ecu_info LIMIT 5")
for r in cursor.fetchall():
    print(r)
cursor.execute("SELECT COUNT(*) FROM topology")
print("topology count:", cursor.fetchone()[0])
cursor.execute("SELECT * FROM topology LIMIT 3")
for r in cursor.fetchall():
    print(r)
conn.close()

print("\n=== dtcs_complete.db ===")
conn = sqlite3.connect("vehicle_info/dtcs_complete.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", tables)
cursor.execute("SELECT COUNT(*) FROM dtc_info")
print("dtc_info count:", cursor.fetchone()[0])
cursor.execute("PRAGMA table_info(dtc_info)")
cols = cursor.fetchall()
print("Schema:")
for c in cols:
    print(f"  {c[1]} ({c[2]})")
cursor.execute("SELECT * FROM dtc_info LIMIT 3")
for r in cursor.fetchall():
    print(r)
conn.close()