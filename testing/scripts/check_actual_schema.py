import sqlite3

# Check vehicle_architecture.db
print("=== vehicle_architecture.db ===")
conn = sqlite3.connect("vehicle_info/vehicle_architecture.db")
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"Tables: {tables}")

for table in tables:
    tname = table[0]
    cursor.execute(f"PRAGMA table_info({tname})")
    cols = cursor.fetchall()
    print(f"\n--- {tname} ---")
    for c in cols:
        print(f"  {c[1]:30s} {c[2]}")
    
    cursor.execute(f"SELECT * FROM {tname} LIMIT 2")
    for row in cursor.fetchall():
        print(f"  Row: {row}")

conn.close()

# Check topology.sqlite
print("\n=== topology.sqlite (if exists) ===")
import os
if os.path.exists("logs/topology.sqlite"):
    conn = sqlite3.connect("logs/topology.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables: {tables}")
    conn.close()
else:
    print("File not found")
