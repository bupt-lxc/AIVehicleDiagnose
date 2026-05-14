
import sqlite3
import os

db_path = r"d:\AI Agent Projects\AI Disgnostic\vehicle_info\vehicle_architecture.db"

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=== ecu_info schema ===")
    cursor.execute("PRAGMA table_info(ecu_info)")
    for col in cursor.fetchall():
        print(col)
        
    print("\n=== topology schema ===")
    cursor.execute("PRAGMA table_info(topology)")
    for col in cursor.fetchall():
        print(col)
        
    print("\n=== ecu_info sample ===")
    cursor.execute("SELECT * FROM ecu_info LIMIT 3")
    for row in cursor.fetchall():
        print(row)

    conn.close()
except Exception as e:
    print(f"Error: {e}")
