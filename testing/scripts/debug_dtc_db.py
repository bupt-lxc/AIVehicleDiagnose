
import sqlite3
import os

db_path = r"d:\AI Agent Projects\AI Disgnostic\vehicle_info\dtcs_complete.db"

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=== ecus schema ===")
    cursor.execute("PRAGMA table_info(ecus)")
    for col in cursor.fetchall():
        print(col)
        
    print("\n=== ecus sample ===")
    cursor.execute("SELECT * FROM ecus LIMIT 5")
    for row in cursor.fetchall():
        print(row)

    conn.close()
except Exception as e:
    print(f"Error: {e}")
